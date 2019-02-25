import numpy as np
import os
import argparse
import shutil

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch


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
parser.add_argument('--path', type=str, default='MNIST_DCGAN')
opt = parser.parse_args()
print(opt)

os.makedirs(opt.path, exist_ok=True)
shutil.rmtree(opt.path)
os.makedirs(opt.path, exist_ok=True)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class DCGenerator(nn.Module):
    def __init__(self, d=128):
        super(DCGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d * 4, 4, 1, 0)  # (n, 4d, 4, 4)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 3, 2, 1)  # (n, 2d, 7, 7)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)
        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)  # (n, d, 14, 14)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)  # (n, 1, 28, 28)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, d=128):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)  # (n, d, 14, 14)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)  # (n, 2d, 7, 7)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1)  # (n, 4d, 4, 4)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)  # (n, 1, 1, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x


dcd = DCDiscriminator()
dcg = DCGenerator()
dcd.weight_init(mean=0.0, std=0.02)
dcg.weight_init(mean=0.0, std=0.02)

os.makedirs('../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(dcg.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(dcd.parameters(), lr=2e-4, betas=(0.5, 0.999))

Tensor = torch.cuda.FloatTensor

adversarial_loss = torch.nn.BCELoss()
adversarial_loss.cuda()
dcg.cuda()
dcd.cuda()

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

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        z = z.reshape(opt.batch_size, opt.latent_dim, 1, 1)

        gen_imgs = dcg(z)
        validity = dcd(gen_imgs)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # ------------------------------------
        #  Train Discriminator and Classifier
        # ------------------------------------

        optimizer_D.zero_grad()

        d_loss = 0
        validity = dcd(real_imgs)
        real_loss = adversarial_loss(validity, valid)
        gen_imgs = dcg(z)

        validity = dcd(gen_imgs.detach())
        fake_loss = adversarial_loss(validity, fake)

        d_loss += (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                                         d_loss.item(), g_loss.item()))

        temp = []
        for j in range(opt.n_paths_G):
            temp.append(gen_imgs[10 * j: 10 * (j + 1), :])
        plot_imgs = torch.cat(temp, dim=0)
        plot_imgs.detach()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(plot_imgs[:100], os.path.join(opt.path, '%d.png' % batches_done), nrow=10,
                       normalize=True)

    if epoch % 50 == 0:
        torch.save({
            'epoch': epoch + 1,
            'g_state_dict': dcg.state_dict(),
            'd_state_dict': dcd.state_dict(),
        }, os.path.join(opt.path, '0checkpoint.tar'))