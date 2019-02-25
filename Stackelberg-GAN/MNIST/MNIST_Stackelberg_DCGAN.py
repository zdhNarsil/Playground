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
parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=100, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=600, help='interval betwen image samples')
parser.add_argument('--n_paths_G', type=int, default=10, help='number of paths of generator')
parser.add_argument('--classifier-para', type=float, default=0.0, help='regularization parameter for classifier')
parser.add_argument('--path', type=str, default='MNIST_Stackelberg_DCGAN')
opt = parser.parse_args()
print(opt)

os.makedirs(opt.path, exist_ok=True)
shutil.rmtree(opt.path)
os.makedirs(opt.path, exist_ok=True)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# TODO: 现在的单个generator还是表达能力太强（三层卷积）
# TODO: 下次试一试两层卷积能不能generate一类mnist
# 简化过的generator
class DCGenerator(nn.Module):
    def __init__(self, d=128):
        super(DCGenerator, self).__init__()

        modules = nn.ModuleList()
        for _ in range(opt.n_paths_G):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(100, 4 * d, 9, 1, 0),
                nn.BatchNorm2d(d * 4),
                nn.ConvTranspose2d(4 * d, 1, 6, 3, 1),
            ))

            # 这个表达能力太强，会串
            # modules.append(nn.Sequential(
            #     nn.ConvTranspose2d(100, d * 4, 5, 1, 0),
            #     nn.BatchNorm2d(d * 4),
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(d * 4, d, 8, 2, 1),
            #     nn.BatchNorm2d(d),
            #     nn.ReLU(),
            #     nn.ConvTranspose2d(d, 1, 4, 2, 1),
            #     nn.Tanh()
            # ))
        self.paths = modules

    # 我不确定这样会不会报错
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        pass
        # x = F.relu(self.deconv1_bn(self.deconv1(input)))
        # x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # x = F.tanh(self.deconv3(x))
        # return x


class DCDiscriminator(nn.Module):
    def __init__(self, d=128):
        super(DCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)  # (n, d, 14, 14)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)  # (n, 2d, 7, 7)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 3, 2, 1)  # (n, 4d, 4, 4)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)  # (n, 1, 1, 1)
        self.fc = nn.Linear(4*d * 4 * 4, 10)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        validity = torch.sigmoid(self.conv4(x))
        classifier = F.log_softmax(self.fc(x.reshape(x.shape[0], -1)))
        return validity, classifier

# 简化版的discriminator
# class DCDiscriminator(nn.Module):
#     def __init__(self, d=128):
#         super(DCDiscriminator, self).__init__()  # (n, 1, 28, 28)
#         self.conv1 = nn.Conv2d(1, d, 4, 2, 1)  # (n, d, 14, 14)
#         self.conv1_bn = nn.BatchNorm2d(d)
#         self.conv2 = nn.Conv2d(d, d*4, 8, 2, 1)  # (n, 4d, 5, 5)
#         self.conv2_bn = nn.BatchNorm2d(d*4)
#         self.conv3 = nn.Conv2d(d*4, 1, 5, 1, 0)  # (n, 1, 1, 1)
#
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     def forward(self, input):
#         x = F.leaky_relu(self.conv1_bn(self.conv1(input)), 0.2)
#         x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
#         x = F.sigmoid(self.conv3(x))
#         return x


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

# TODO: 在第一个epoch之后使用classifier_para，并且对其调参
cls_para = opt.classifier_para
for epoch in tqdm(range(opt.n_epochs)):
    if epoch > 1:
        opt.classifier_para = cls_para
    else:
        opt.classifier_para = 0.

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

        g_loss = 0.
        for k in range(opt.n_paths_G):
            gen_imgs = dcg.paths[k](z)
            validity, classifier = dcd(gen_imgs)
            g_loss += adversarial_loss(validity, valid)

            target = Tensor(opt.batch_size).fill_(k)
            target = target.type(torch.cuda.LongTensor)
            g_loss += opt.classifier_para * F.nll_loss(classifier, target)
        g_loss.backward()
        optimizer_G.step()

        # ------------------------------------
        #  Train Discriminator and Classifier
        # ------------------------------------

        optimizer_D.zero_grad()

        d_loss = 0
        validity, _ = dcd(real_imgs)
        real_loss = adversarial_loss(validity, valid)
        temp = []
        for k in range(opt.n_paths_G):
            gen_imgs = dcg.paths[k](z)
            temp.append(gen_imgs[0:(100 // opt.n_paths_G), :])

            validity, classifier = dcd(gen_imgs.detach())
            fake_loss = adversarial_loss(validity, fake)
            d_loss += (real_loss + fake_loss) / 2

            target = Variable(Tensor(imgs.size(0)).fill_(k), requires_grad=False)
            target = target.type(torch.cuda.LongTensor)
            d_loss += opt.classifier_para * F.nll_loss(classifier, target)

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

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch + 1,
            'g_state_dict': dcg.state_dict(),
            'd_state_dict': dcd.state_dict(),
        }, os.path.join(opt.path, '0checkpoint.tar'))
