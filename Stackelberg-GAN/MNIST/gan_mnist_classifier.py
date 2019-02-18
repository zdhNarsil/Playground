import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval betwen image samples')
parser.add_argument('--n_paths_G', type=int, default=10, help='number of paths of generator')
parser.add_argument('--classifier_para', type=float, default=1.0, help='regularization parameter for classifier')
parser.add_argument('--path', type=str,)
opt = parser.parse_args()
print(opt)

os.makedirs(opt.path, exist_ok=True)
shutil.rmtree(opt.path)
os.makedirs(opt.path, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8)) #0.8是啥？？
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        modules = nn.ModuleList()
        for _ in range(opt.n_paths_G):
            modules.append(nn.Sequential(
                *block(opt.latent_dim, 128),
                *block(128, 512),
                # *block(256, 512),
                # *block(512, 512),
                # *block(512, 1024),
                nn.Linear(512, int(np.prod(img_shape))),
                nn.Tanh()
            ))
        self.paths = modules

    def forward(self, z):
        img = []
        for path in self.paths:
            img.append(path(z).view(img.size(0), *img_shape)) # img一个list怎么有size()方法？？
        img = torch.cat(img, dim=0)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(int(np.prod(img_shape)), 512)
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2 = nn.Linear(512, 256)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)
        modules = nn.ModuleList()
        modules.append(nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ))
        modules.append(nn.Sequential(
            nn.Linear(256, 10),
        ))
        self.paths = modules

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        img_flat = self.lr2(self.fc2(self.lr1(self.fc1(img_flat))))
        validity = self.paths[0](img_flat)
        classifier = F.log_softmax(self.paths[1](img_flat), dim=1)
        return validity, classifier


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
os.makedirs('../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

# print("--------Loading Model--------")
# checkpoint = torch.load(os.oath.join(opt.path, '0checkpoint.tar'))
# generator.load_state_dict(checkpoint['g_state_dict'])
# discriminator.load_state_dict(checkpoint['d_state_dict'])

for epoch in tqdm(range(opt.n_epochs)):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        g_loss = 0
        for k in range(opt.n_paths_G):
            # Generate a batch of images
            gen_imgs = generator.paths[k](z)

            # Loss measures generator's ability to fool the discriminator
            validity, classifier = discriminator(gen_imgs)
            g_loss += adversarial_loss(validity, valid)

            # 加了一项分类loss，用的居然是generator编号作为标签。。。
            # Loss measures classifier's ability to classify various generators
            target = Variable(Tensor(imgs.size(0)).fill_(k), requires_grad=False)
            target = target.type(torch.cuda.LongTensor)
            g_loss += F.nll_loss(classifier, target) * opt.classifier_para

        g_loss.backward()
        optimizer_G.step()

        # ------------------------------------
        #  Train Discriminator and Classifier
        # ------------------------------------

        optimizer_D.zero_grad()

        d_loss = 0
        validity, classifier = discriminator(real_imgs)
        real_loss = adversarial_loss(validity, valid)
        temp = []
        for k in range(opt.n_paths_G):
            # Generate a batch of images
            gen_imgs = generator.paths[k](z).view(imgs.shape[0], *img_shape)
            temp.append(gen_imgs[0:(100 // opt.n_paths_G), :])

            # Loss measures discriminator's ability to classify real from generated samples
            validity, classifier = discriminator(gen_imgs.detach())
            fake_loss = adversarial_loss(validity, fake)
            d_loss += (real_loss + fake_loss) / 2

            # Loss measures classifier's ability to classify various generators
            target = Variable(Tensor(imgs.size(0)).fill_(k), requires_grad=False)
            target = target.type(torch.cuda.LongTensor)
            d_loss += F.nll_loss(classifier, target) * opt.classifier_para

        plot_imgs = torch.cat(temp, dim=0)
        plot_imgs.detach()

        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                         d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(plot_imgs[:100], os.path.join(opt.path, '%d.png' % batches_done), nrow=10,
                       normalize=True)

    if epoch % 50 == 0:
        torch.save({
        'epoch': epoch + 1,
        'g_state_dict': generator.state_dict(),
        'd_state_dict': discriminator.state_dict(),
        }, os.path.join(opt.path, '0checkpoint.tar'))
