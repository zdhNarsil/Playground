import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import argparse
from tqdm import tqdm
import pdb
import numpy as np
import math
from collections import OrderedDict

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--no-CUDA', action='store_true')
parser.add_argument('--gpu', type=int, default=0, help='Which gpu to use')
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=31)
parser.add_argument('--lr-decay-epoch', type=int, default=21)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size-l', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--epsilon', type=float, default=3.0)
parser.add_argument('--num_l', type=int, default=200)
parser.add_argument('--logdir', type=str, default='./vat_logs')

args = parser.parse_args()

use_CUDA = True
if args.no_CUDA or (not torch.cuda.is_available()):
    use_CUDA = False
if use_CUDA:
    torch.cuda.set_device(torch.device('cuda:{}'.format(args.gpu)))


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=False):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if self.ACROSS_CHANNELS:
            self.average = nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                                        stride=1,
                                        padding=(int((local_size - 1.0) / 2), 0, 0))
        else:
            self.average = nn.AvgPool2d(kernel_size=local_size,
                                        stride=1,
                                        padding=int((local_size - 1.0) / 2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.lrn1 = LRN()
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.lrn2 = LRN()
        self.fc1 = nn.Linear(49 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.reshape((-1, 1, 28, 28))
        x = F.relu_(self.conv1(x))
        x = F.relu_(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.lrn1(x)
        x = F.relu_(self.conv3(x))
        x = F.relu_(self.conv4(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.lrn2(x)
        x = x.reshape((-1, 49 * 64))
        x = F.relu_(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x)
        return x


def idx2onehot(idx, n):
    idx = idx.reshape(idx.size(0), 1)

    assert idx.size(1) == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)  # 注意scatter_要求得是torch的longint数组

    return onehot


def normalizevector(r):  ###
    n_batch = r.shape[0]

    denominator = torch.norm(r.reshape(n_batch, -1), p=float('inf'), dim=1, keepdim=True).reshape(n_batch, 1, 1, 1)
    r /= (denominator + 1e-12)

    denominator = 1e-12 + torch.norm(r.reshape(n_batch, -1), p='fro', dim=1).reshape(n_batch, 1, 1, 1)
    return r / denominator


def crossentropy(logits, label):
    # both input is (N_batch, Class)
    return -torch.mean(torch.sum(label * torch.log(logits + 1e-8), dim=1))


def kldivergence(logits, label):
    return torch.mean(torch.sum(label * (torch.log(label + 1e-8) - torch.log(logits + 1e-8)), dim=1))


def eval_one_epoch(net, batch_generator):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()

    pbar.set_description('Evaluating')  #
    for (data, label) in pbar:
        if use_CUDA:
            data = data.cuda()
            label = label.cuda()

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar.set_postfix(pbar_dic)
    return clean_accuracy.mean


def torch_accuracy(output, target, topk=(1,)):
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, largest=True, sorted=True)  # torch.topk()
    pred = pred.t()  # 转置

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


class AvgMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


'''
一个epoch后Acc=93.04, 四个epoch后Acc=97.11，再往后突然nan
'''
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

    model = Net()
    if use_CUDA:
        model = model.cuda()
    # pdb.set_trace()
    # x, y = trainset_ul[0]
    # y = idx2onehot(y.reshape((1,)), 10)
    # print(model(x,))

    for i in range(args.epoch):
        model.train()

        if i < args.lr_decay_epoch:
            decayed_lr = args.lr
        else:
            decayed_lr = args.lr * (args.epoch - i) / float(args.epoch - args.lr_decay_epoch)

        optimizer = torch.optim.Adam(model.parameters(), lr=decayed_lr)

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

            ###################################
            # power method - begin
            r_adv = normalizevector(torch.randn(x_ul.shape))
            for k in range(1):
                r_adv *= 1e-6  # ?
                r_adv.requires_grad_()

                out_r = model(x_ul + r_adv)
                kl = kldivergence(out_r, out_ul)  # 两个输入反过来其实也可以？
                r_adv = torch.autograd.grad(kl, r_adv)[0]  #

                r_adv = r_adv.detach()#?
                r_adv = normalizevector(r_adv)

            out_adv = model(x_ul + args.epsilon * r_adv)
            out_ul.detach_()  # ?
            vat_loss = kldivergence(out_adv, out_ul)
            # power method - end
            ###################################

            loss = supervised_loss + args.beta * entropy_loss + args.alpha * vat_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_dic = OrderedDict()
            pbar_dic['supervised loss'] = '{:.2f}'.format(supervised_loss)
            pbar_dic['vat loss'] = '{:.2f}'.format(vat_loss)
            pbar_dic['entropy loss'] = '{:.2f}'.format(entropy_loss)
            pbar_dic['total loss'] = '{:.2f}'.format(loss)
            pbar.set_postfix(pbar_dic)

        acc = eval_one_epoch(model, test_loader)
        print('epoch:', i, 'supervised loss:', supervised_loss, 'vat loss:', vat_loss,
              'entropy loss', entropy_loss, 'total loss:', loss, 'acc:', acc)
