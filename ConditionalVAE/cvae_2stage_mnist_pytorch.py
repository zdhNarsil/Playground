from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import GPUtil
import os
import math
from IPython import embed


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--noTrain', action="store_true", default=False, help='if train or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

#deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1, maxMemory=0.1, excludeID=[], excludeUUID=[])
deviceIDs = GPUtil.getAvailable()
#deviceIDs = [3]
print('Unloaded gpu:', deviceIDs)
embed()

# 设置显卡
os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceIDs[0])
#torch.cuda.set_device(0)

#embed()


class VAE(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, d_label):
        # 784, 400, 20, 10
        # 20, 15, 10, 10
        super(VAE, self).__init__()

        self.d_in = d_in

        self.fc1 = nn.Linear(d_in + d_label, d_hidden)
        self.fc21 = nn.Linear(d_hidden, d_out)
        self.fc22 = nn.Linear(d_hidden, d_out)
        self.fc3 = nn.Linear(d_out + d_label, d_hidden)
        self.fc4 = nn.Linear(d_hidden, d_in)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)  # 只传递了shape
        return eps.mul(std).add_(mu)  # torch.mul 是 element wise

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, label):
        mu, logvar = self.encode(torch.cat((x.view(-1, self.d_in), label), 1))
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z, label), 1)), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, dim_x):
    #embed()
    target = x.view(-1, dim_x)
    #target = target.requires_grad_(requires_grad=False)

    BCE = F.binary_cross_entropy(recon_x, target, reduction='sum')
    #BCE = F.binary_cross_entropy(x.view(-1, dim_x), recon_x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(model, optimizer, epoch, loader):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(loader):
        # data shape: torch.Size([128, 1, 28, 28]) label shape: torch.Size([128])
        # label 是 0～9

        # 重要！！
        data.detach_()
        label.detach_()

        data = data.to(device)
        label = idx2onehot(label, 10).to(device)  # label: [128, 10]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, label)

        dim_x = data.reshape(data.shape[0], -1).shape[1]
        loss = loss_function(recon_batch, data, mu, logvar, dim_x)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader.dataset),
                100. * batch_idx / len(loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(loader.dataset)))


def test(model, epoch, loader, isPrint=False):
    model.eval()
    test_loss = 0
    makedirs('results/cvae_2stage/')
    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data = data.to(device)
            label = idx2onehot(label, 10).to(device)

            recon_batch, mu, logvar = model(data, label)
            dim_x = data.reshape(data.shape[0], -1).shape[1]
            test_loss += loss_function(recon_batch, data, mu, logvar, dim_x).item()

            # 展示原图和vae重建出来的图像
            if i == 0 and isPrint:
                n = min(data.size(0), 8)
                dim_x = int(math.sqrt(dim_x))
                comparison = torch.cat([data[:n],
                                      recon_batch.view(args.batch_size, 1, dim_x, dim_x)[:n]])
                save_image(comparison.cpu(),
                         'results/cvae_2stage/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def idx2onehot(idx, n):
    idx = idx.reshape(idx.size(0), 1)

    assert idx.size(1) == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1) # 注意scatter_要求得是torch的longint数组

    return onehot


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class HiddenDataset(torch.utils.data.Dataset):
    def __init__(self, origin_dataset, model):
        self.origin_dataset = origin_dataset
        self.model = model

    def __len__(self):
        return len(self.origin_dataset)

    def __getitem__(self, idx):
        image, label = self.origin_dataset[idx]
        new_label = torch.tensor([label], dtype=torch.int64)
        new_label = idx2onehot(new_label, 10).to(device)
        mu, logvar = self.model.encode(torch.cat((image.view(-1, 784), new_label), 1))
        z = self.model.reparameterize(mu, logvar)
        return (z, label)

if __name__ == "__main__":
    # ？？？
    # torch.multiprocessing.set_start_method("spawn", force=True)

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model1 = VAE(784, 400, 20, 10).to(device)
    optimizer1 = optim.Adam(model1.parameters(), lr=1e-3)
    PATH = "cvae_2stage.pt"

    # 先炼一个vae
    print("begin to train the 1st vae")
    if not args.noTrain:
        for epoch in range(1, args.epochs + 1):
            train(model1, optimizer1, epoch, train_loader)
            test(model1, epoch, test_loader)
        torch.save(model1.state_dict(), PATH)

    # 加载之前训好的第一步的模型
    model1.load_state_dict(torch.load(PATH))

    # 获取hidden数据
    print("begin to make hidden data...")
    train_loader2 = torch.utils.data.DataLoader(
        HiddenDataset(datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()), model1),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader2 = torch.utils.data.DataLoader(
        HiddenDataset(datasets.MNIST('../data', train=False,
                       transform=transforms.ToTensor()), model1),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # 再炼一个vae
    print("begin to train the 2nd vae")
    model2 = VAE(20, 15, 10, 10).to(device)
    optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        train(model2, optimizer2, epoch, train_loader2)
        test(model2, epoch, test_loader2, isPrint=False)

        with torch.no_grad():
            makedirs('results/cvae_2stage/')
            sample = torch.randn(64, 10).to(device)  # 这个小batch里只有64个
            label_sample = torch.zeros(64, 1, dtype=torch.int64)
            label_sample = idx2onehot(label_sample, 10)
            #from IPython import embed; embed()
            if args.cuda:
                label_sample = label_sample.cuda()
            sample = model2.decode(torch.cat((sample, label_sample), 1)).cpu()
            sample = model1.decode(torch.cat((sample, label_sample), 1)).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/cvae_2stage/sample_' + str(epoch) + '.png')
