import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import os
import GPUtil
from datetime import datetime
from collections import OrderedDict
import math
import argparse
import numpy as np
from tqdm import tqdm

from pytorch_utils import idx2onehot, normalizevector, crossentropy, kldivergence, eval_one_epoch
from pytorch_cvae import cVAE
from pytorch_vat import Net

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--no-CUDA', action='store_true')
# parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--lr-decay-epoch', type=int, default=21)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--coef-vat1', type=float, default=1)
parser.add_argument('--coef-vat2', type=float, default=1)
parser.add_argument('--coef-ent', type=float, default=1)
parser.add_argument('--zeta', type=float, default=0.001)
parser.add_argument('--epsilon1', type=float, default=0.5)
parser.add_argument('--epsilon2', type=float, default=0.05)
parser.add_argument('--num-l', type=int, default=200)
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--logdir', type=str, default='./cvaeVATz_logs')

args = parser.parse_args()

use_CUDA = True
if args.no_CUDA or (not torch.cuda.is_available()):
    use_CUDA = False
if use_CUDA:
    torch.cuda.set_device(torch.device('cuda:{}'.format(args.gpu)))

if __name__ == '__main__':
    trainset_ul = datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.ToTensor())
    train_ul_loader = torch.utils.data.DataLoader(
        trainset_ul, batch_size=args.batch_size_ul, shuffle=True)

    trainset_l = torch.utils.data.Subset(trainset_ul, [i for i in range(args.num_l)])
    train_l_loader = torch.utils.data.DataLoader(
        trainset_l, batch_size=args.batch_size_l, shuffle=True)
    train_l_loader_list = list(train_l_loader)
    list_len = len(train_l_loader_list)

    testset = datasets.MNIST('../data', train=False, download=True,
                             transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size_ul, shuffle=True)

    vae = cVAE(args.latent_dim)
    model = Net()
    if use_CUDA:
        vae = vae.cuda()
        model = model.cuda()

    for i in range(args.epoch):
        model.train()

        if i < args.lr_decay_epoch:
            decayed_lr = args.lr
        else:
            decayed_lr = args.lr * (args.epoch - i) / float(args.epoch - args.lr_decay_epoch)

        optimizer = torch.optim.Adam(model.parameters(), lr=decayed_lr)
        vae_optimizer = torch.optim.Adam(vae.parameters(), lr=decayed_lr)

        pbar = tqdm(enumerate(train_ul_loader))
        for j, (x_ul, _) in pbar:
            x_l, y_l = train_l_loader_list[j % list_len]

            if use_CUDA:
                x_ul, x_l, y_l = x_ul.cuda(), x_l.cuda(), y_l.cuda()

            out_l = model(x_l)  # (32, 10)
            # supervised_loss = crossentropy(out_l, idx2onehot(y_l, 10))
            supervised_loss = F.nll_loss(torch.log(out_l), y_l)

            out_ul = model(x_ul)  # (128,10)
            entropy_loss = crossentropy(out_ul, out_ul)

            # conditional vae graph
            # x_recon, mu, logvar = vae(x_ul, out_ul)
            mu, logvar = vae.encode(x_ul, out_ul)
            z = vae.reparamenterize(mu, logvar)
            x_recon = vae.decode(z, out_ul)
            x_gen = vae.decode(torch.randn((out_ul.shape[0], args.latent_dim)), out_ul)  # ？
            vae_loss = vae.BCE(x_recon, x_ul) + vae.KLD(mu, logvar)

            # TNAR graph
            r0 = torch.zeros_like(z).requires_grad_()
            x_recon_r0 = vae.decode(z + r0, out_ul)
            diff2 = 0.5 * torch.sum((x_recon - x_recon_r0) ** 2, dim=1)  #
            diffJaco = torch.autograd.grad(diff2, r0)[0]  #

            # power method: compute tagent adv & loss
            r_adv = normalizevector(torch.randn(z.shape)).requires_grad_()
            for j in range(1):
                r_adv *= 1e-6
                x_r = vae.decode(z + r_adv, out_ul)
                out_r = model(x_r - x_recon + x_ul)  # x_r - x_recon 是原文里的 r(eta)
                kl = kldivergence(out_r, out_ul)  # 原文里的F(x, r(eta), theta)
                r_adv = torch.autograd.grad(kl, r_adv.detach())[0]
                r_adv.detach_()#?
                r_adv = normalizevector(r_adv)

                # begin cg
                rk = r_adv + 0.  #
                pk = rk + 0.  # pk是原文里的mu？
                xk = torch.zeros_like(rk)  #
                for k in range(4):
                    Apk = torch.autograd.grad(diffJaco * pk, r0)[0].detach_()  #
                    pkApk = torch.sum(pk * Apk, dim=1, keepdim=True)
                    rk2 = torch.sum(rk * rk, dim=1, keepdim=True)
                    mask = rk2 > 1e-8
                    alphak = (rk2 / (pkApk + 1e-8)) * mask.float()
                    xk += alphak * pk
                    rk -= alphak * Apk
                    betak = torch.sum(rk * rk, dim=1, keepdim=True) / (rk2 + 1e-8)
                    pk = rk + betak * pk
                # end cg
                r_adv = normalizevector(xk)

            x_adv = vae.decode(z + r_adv * args.epsilon1, out_ul)
            r_x = x_adv - x_recon
            out_adv = model(x_ul + r_x)#.detach_()
            vat_tangent_loss = kldivergence(out_adv, out_ul.detach())#为啥detach？

            # 计算 normal regularization
            r_x = normalizevector(r_x)
            r_adv_orth = normalizevector(torch.randn(x_ul.shape))
            for j in range(1):
                r_adv_orth1 = 1e-6 * r_adv_orth
                out_r = model(x_ul + r_adv_orth1)
                kl = kldivergence(out_r, out_ul)
                r_adv_orth1 = torch.autograd.grad(kl, r_adv_orth1)[0] / 1e-6
                r_adv_orth1.detach_()
                # 这里的args.zeta是原文里的lambda
                r_adv_orth = r_adv_orth1 \
                             - args.zeta * (torch.sum(r_x * r_adv_orth, dim=1, keepdims=True) * r_x) \
                             + args.zeta * r_adv_orth
                r_adv_orth = normalizevector(r_adv_orth)
            out_adv_orth = model(x_ul + r_adv_orth * args.epsilon2)
            vat_orth_loss = kldivergence(out_adv_orth, out_ul.detach())

            total_loss = supervised_loss + args.coef_ent * entropy_loss \
                         + args.coef_vat1 * vat_tangent_loss + args.coef_vat2 * vat_orth_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            vae_optimizer.zero_grad()
            vae_loss.backward()
            vae_optimizer.step()

            pbar_dic = OrderedDict()
            pbar_dic['supervised loss'] = '{:.2f}'.format(supervised_loss)
            pbar_dic['vat tangent loss'] = '{:.2f}'.format(vat_tangent_loss)
            pbar_dic['vat orth loss'] = '{:.2f}'.format(vat_orth_loss)
            pbar_dic['entropy loss'] = '{:.2f}'.format(entropy_loss)
            pbar_dic['total loss'] = '{:.2f}'.format(total_loss)
            pbar.set_postfix(pbar_dic)

        acc = eval_one_epoch(model, test_loader)
        print('epoch:', i, 'total loss:', total_loss, 'acc:', acc)