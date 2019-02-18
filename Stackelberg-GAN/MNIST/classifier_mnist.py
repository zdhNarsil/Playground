import argparse
import os
import numpy as np
import math
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import shutil

import sys
sys.path.append('../..')
from pytorch_data import load_MNIST

parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true")
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs of training')
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

CUDA = torch.cuda.is_available() and (not opt.no_cuda)
img_shape = (opt.channels, opt.img_size, opt.img_size)


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


class Classifier(nn.Module):
    def __init__(self, hidden_dim=784):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))


if __name__ == '__main__':
    tsfs = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])
    train_loader, test_loader = load_MNIST(opt.batch_size, data_dir='../data/mnist',
                                           transform_train=tsfs, transform_test=tsfs)

    print("--------Loading Model--------")
    checkpoint = torch.load(os.path.join(opt.path, '0checkpoint.tar'))
    generator = Generator()
    generator.load_state_dict(checkpoint['g_state_dict'])

    model = Classifier()
    criterion = torch.nn.CrossEntropyLoss()

    if CUDA:
        deviceIDs = [0]
        # deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1,
        #                                 maxMemory=0.1, excludeID=[], excludeUUID=[])
        print('available cuda device ID(s):', deviceIDs)
        torch.cuda.set_device(deviceIDs[0])
        model = model.cuda()
        if generator is not None:
            generator = generator.cuda()
        criterion = criterion.cuda()

    model.load_state_dict(torch.load('classifier.tar'))

    for i in range(1):
        z = torch.Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim)))
        if CUDA:
            z = z.cuda()

        for k in range(opt.n_paths_G):
            # Generate a batch of images
            gen_imgs = generator.paths[k](z)
            pred = model(gen_imgs)
            _, predicted = torch.max(pred.data, 1)
            print('path:', k, 'predict labels:', predicted)

    '''
    optimizer = torch.optim.Adam(model.parameters(), )

    for i in range(opt.n_epochs):
        correct = 0
        total = 0
        train_loss = 0.
        for j, data in enumerate(tqdm(train_loader)):
            images, labels = data
            if CUDA:
                images, labels = images.cuda(), labels.cuda()

            pred = model(images)
            optimizer.zero_grad()
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            correct += predicted.eq(labels.data).sum().item()
            total += len(labels)
            acc = correct / total
            print('\nepoch:', i, 'train loss:', train_loss / (j + 1), 'accuracy:', acc)

        # test or attack
        correct = 0
        total = 0
        for j, data in enumerate(tqdm(test_loader)):
            images, labels = data
            if CUDA:
                images, labels = images.cuda(), labels.cuda()

            pred = model(images)  # test
            _, predicted = torch.max(pred.data, 1)
            correct += predicted.eq(labels.data).sum().item()
            total += len(labels)
            acc = correct / total
            print('correct:', correct, 'total:', total, 'accuracy:', acc)

        print('epoch:', i, 'save model\'s state dict')
        torch.save(model.state_dict(), 'classifier.tar')
    '''



