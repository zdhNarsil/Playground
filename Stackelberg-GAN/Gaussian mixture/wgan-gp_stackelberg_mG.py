import argparse
import os
import numpy as np
import math
from datetime import datetime
import pdb

from torch.autograd import Variable

import torch.nn as nn
import torch
import shutil

import matplotlib.pyplot as plt

plt.switch_backend('agg')
import seaborn as sns
import matplotlib

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=2, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
parser.add_argument('--n_paths_D', type=int, default=1, help='number of paths of discriminator')
parser.add_argument('--n_paths_G', type=int, default=8, help='number of paths of generator')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--no-output', action='store_true')
parser.add_argument('--data-var', type=float, default=0.02, help='the variance used when sampling data')
parser.add_argument('--lambda_gp', type=float, default=2., help='coefficient of gradient penalty')
opt = parser.parse_args()
print(opt)

# np.random.seed(1)
# torch.manual_seed(1)
# if torch.cuda.is_available():
#    torch.cuda.manual_seed(1)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        modules = nn.ModuleList()
        for _ in range(opt.n_paths_G):
            modules.append(nn.Sequential(
                *block(opt.latent_dim, 32, normalize=False),
                nn.Linear(32, 2),
                nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        for path in self.paths:
            img.append(path(z))
        img = torch.cat(img, dim=1)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        modules = nn.ModuleList()
        for _ in range(opt.n_paths_D):
            modules.append(nn.Sequential(
                nn.Linear(2, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ))
        self.paths = modules

    def forward(self, img):
        img_flat = img
        validity = []
        for path in self.paths:
            validity.append(path(img_flat))
        validity = torch.cat(validity, dim=1)
        return validity


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1,)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,  # 不是要算高阶导数才要True？
        retain_graph=True,
        only_inputs=True,
    )[0]  # autograd.grad 输出是个tuple所以要加 [0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if not opt.no_output:
    path = 'images_ensemble_wgan_gp_mG' + "{0:%Y-%m-%d}_{0:%H-%M-%S}".format(datetime.now())
    os.makedirs(path, exist_ok=True)
    shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    n_mixture = 8
    radius = 1
    std = 0.01
    thetas = np.linspace(0, 2 * (1 - 1 / n_mixture) * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    data_size = 1000 * n_mixture
    data = torch.zeros(data_size, 2)
    for i in range(data_size):
        coin = np.random.randint(0, n_mixture)
        data[i, :] = torch.normal(mean=torch.Tensor([xs[coin], ys[coin]]), std=std * torch.ones(1, 2))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    images = []
    n_batch = math.ceil(data_size / opt.batch_size)

    for epoch in range(opt.n_epochs):
        # colors = matplotlib.cm.rainbow(np.linspace(0, 1, 1 + opt.n_paths_G))
        # plt.plot(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), color=colors[0], marker='.', linestyle='None')
        # z = Variable(Tensor(np.random.normal(0, 1, (8000 // opt.n_paths_G, opt.latent_dim))))
        #
        # # 看下此时generator的8个path能把隐空间分别输出成啥样
        # for k in range(opt.n_paths_G):
        #     gen_data = generator.paths[k](z).detach()
        #     plt.plot(gen_data[:, 0].cpu().numpy(), gen_data[:, 1].cpu().numpy(), color=colors[1 + k], marker='.',
        #              linestyle='None')

        for i in range(n_batch):

            imgs = data[i * opt.batch_size:min((i + 1) * opt.batch_size, data_size - 1), :]

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), opt.n_paths_D).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), opt.n_paths_D).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------
            if i % opt.n_critic == 0:
                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                g_loss = 0
                for k in range(opt.n_paths_G):
                    # Generate a batch of images
                    gen_imgs = generator.paths[k](z)
                    g_loss += -torch.mean(discriminator(gen_imgs))

                g_loss.backward()
                optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            d_loss = 0
            real_loss = -torch.mean(discriminator(real_imgs))
            for k in range(opt.n_paths_G):
                # Generate a batch of images
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
                gen_imgs = generator.paths[k](z)

                # Measure discriminator's ability to classify real from generated samples
                fake_loss = torch.mean(discriminator(gen_imgs.detach()))
                d_loss += (real_loss + fake_loss) / 2

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data)
            d_loss += opt.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D value: %f]" % (
                epoch + 1, opt.n_epochs, i, n_batch,
                d_loss.item(), g_loss.item(), discriminator(real_imgs).mean(0).mean().item()))

        colors = matplotlib.cm.rainbow(np.linspace(0, 1, 1 + opt.n_paths_G))
        plt.plot(data[:, 0].cpu().numpy(), data[:, 1].cpu().numpy(), color=colors[0], marker='.', linestyle='None')

        z = Variable(Tensor(np.random.normal(0, 1, (8000 // opt.n_paths_G, opt.latent_dim))))
        temp = []
        for k in range(opt.n_paths_G):
            # temp.append(generator.paths[k](z).detach())
            # gen_data = torch.cat(temp, dim=0)
            # sns.jointplot(gen_data[:, 0].cpu().numpy(), gen_data[:, 1].cpu().numpy(), kind='kde', stat_func=None)
            gen_data = generator.paths[k](z).detach()
            plt.plot(gen_data[:, 0].cpu().numpy(), gen_data[:, 1].cpu().numpy(), color=colors[1 + k], marker='.',
                     linestyle='None')

        if not opt.no_output:
            plt.savefig(os.path.join(path, '%d.png' % epoch))
            plt.close('all')
