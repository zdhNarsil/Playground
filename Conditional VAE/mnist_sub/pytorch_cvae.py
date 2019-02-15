# use pytorch to implement cvae.py
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=121)
parser.add_argument('--lr-decay-epoch', type=int, default=81)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--latent-dim', type=int, default=100)
parser.add_argument('--kld-coef', type=float, default=1.0)
parser.add_argument('--logdir', type=str, default='./cvae_logs')

args = parser.parse_args()


class cVAE(nn.Module):
    def __init__(self, latent_dim):
        super(cVAE, self).__init__()
        self.latent_dim = latent_dim

        # encode to process x
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=1)
        # encode to process y
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        # encode to produce mu & logvar
        self.fc3 = nn.Linear(49 * 32 + 128, self.latent_dim)
        self.fc4 = nn.Linear(49 * 32 + 128, self.latent_dim)

        # decode to process (z, y)
        self.fc5 = nn.Linear(self.latent_dim + 10, 49 * 32)
        #这里wjf用tf的转置卷积实现的，但是pytorch的转置卷积怎么换参数也不能达到想要的形状……哭哭
        # (-1, 32, 7, 7) -> (-1, 16, 14, 14)
        # self.transconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, padding=2, stride=2)
        self.transconv1 = nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=12)
        # (-1, 16, 14, 14) -> (-1, 1, 28, 28)
        # self.transconv2 = nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2, stride=2)
        self.transconv2 = nn.Conv2d(16, 1, kernel_size=5, stride=1, padding=9)

    def encode(self, x, y):
        x = x.reshape((-1, 1, 28, 28))
        x = self.conv1(x)
        x = F.relu_(F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.conv2(x)
        x = F.relu_(F.max_pool2d(x, kernel_size=2, stride=2))

        y = F.relu_(self.fc1(y))
        y = F.relu_(self.fc2(y))

        x = x.reshape((-1, 49 * 32))
        x = torch.cat((x, y), 1)
        mu = self.fc3(x)
        logvar = self.fc4(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):  # 随机sample
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # torch.mul 是 element wise

    def decode(self, z, y):
        z = torch.cat((z, y), 1)
        z = F.relu_(self.fc5(z))
        z = z.reshape((-1, 32, 7, 7))
        z = F.relu_(self.transconv1(z))
        # print(z.shape)
        z = F.sigmoid(self.transconv2(z))
        # print(z.shape)
        z = z.reshape((-1, 784))  # 输出伯努利分布的参数
        return z

    def forward(self, x, label):
        mu, logvar = self.encode(x, label)  #
        z = self.reparameterize(mu, logvar)
        return self.decode(z, label), mu, logvar

    def BCE(self, recon_x, x):
        # 希望这里效果是对batch内平均?
        return torch.mean(F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='mean')) #?
        # x.view(-1, 784) 应该在[0, 1]之间？

    def KLD(self, mu, logvar):
        # 对batch内求平均?
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))  #?


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def idx2onehot(idx, n):
    idx = idx.reshape(idx.size(0), 1)

    assert idx.size(1) == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)  # 注意scatter_要求得是torch的longint数组

    return onehot


if __name__ == '__main__':
    trainset = datasets.MNIST('../data', train=True, download=True,
                              transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True)

    x, y = trainset[0]
    y = idx2onehot(y.reshape((1,)), 10)

    model = cVAE(args.latent_dim)
    print(model(x, y))
