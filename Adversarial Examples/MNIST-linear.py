import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from tqdm import tqdm
import argparse
import GPUtil

from attack import FGSM, IPGD
import sys
sys.path.append('..')
from pytorch_data import load_MNIST
'''
Results: 
For 2-linear layers model the test accuracy can get somewhat 95%.
Even for a simple model like this (MNIST, 2 linear layers)
the test accuracy deduces to 2/10000, 
sometimes 0/10000 (test set has 10000 images) under the simplest attack (FGSM). 
'''

parser = argparse.ArgumentParser()

parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--save-path", type=str, default='./MNIST-linear.pt')
parser.add_argument("--load", action="store_true")
parser.add_argument("--load-path", type=str, default='./MNIST-linear.pt')
# parser.add_argument("--test", action="store_true")
parser.add_argument("--attack", choices=['FGSM', 'IPGD'], default=None)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--train-epoch", type=int, default=3)
parser.add_argument("--adv-ratio", type=float, default=0., help="ratio of advrserial examples in whole training dataset")

args = parser.parse_args()

CUDA = torch.cuda.is_available() and (not args.no_cuda)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x))


if __name__ == '__main__':
    model = Net()
    train_loader, test_loader = load_MNIST(args.batch_size, './data/mnist')
    criterion = torch.nn.CrossEntropyLoss()

    if CUDA:
        deviceIDs = [2, ]
        # deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1,
        #                                 maxMemory=0.1, excludeID=[], excludeUUID=[])
        print('available cuda device ID(s):', deviceIDs)
        DEVICE = torch.device('cuda:{}'.format(deviceIDs[0]))
        torch.cuda.set_device(deviceIDs[0])
        # model = model.cuda()
        model = model.to(DEVICE)
        # criterion = criterion.cuda()
        criterion = criterion.to(DEVICE)

    if args.load:
        # load state_dict
        model.load_state_dict(torch.load(args.load_path))
        # for param in model.fc1.parameters():
        # 看rank和一范

    else:
        # train
        optimizer = torch.optim.Adam(model.parameters(), )

        for i in range(args.train_epoch):
            correct = 0
            total = 0
            train_loss = 0.
            for j, data in enumerate(tqdm(train_loader)):
                images, labels = data

                if CUDA:
                    images, labels = images.cuda(), labels.cuda()

                if torch.rand(1).item() < args.adv_ratio:
                    images = IPGD(model, images, labels, criterion=criterion, CUDA=CUDA)

                images = images.reshape((-1, 784))
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

        # save state_dict
        if args.save:
            torch.save(model.state_dict(), args.save_path)

    # test or attack
    correct = 0
    total = 0
    for j, data in enumerate(tqdm(test_loader)):
        images, labels = data
        if CUDA:
            images, labels = images.cuda(), labels.cuda()

        if args.attack:
            if args.attack == 'FGSM':
                adv_inp = FGSM(model, images, labels, criterion=criterion, CUDA=CUDA)
                # print(torch.sum(images - adv_inp))
                pred = model(adv_inp)
            elif args.attack == 'IPGD':
                adv_inp = IPGD(model, images, labels, criterion=criterion, CUDA=CUDA)
                # print(torch.sum(images - adv_inp))
                pred = model(adv_inp)
        else:
            pred = model(images)  # test

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(labels.data).sum().item()
        total += len(labels)
        acc = correct / total
        print('correct:', correct, 'total:', total, 'accuracy:', acc)