import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm
import argparse
import GPUtil
import pdb

from attack import FGSM, IPGD

'''
Results: 
For  model the test accuracy can get somewhat %.
the test accuracy deduces to /10000, 
sometimes 0/10000 (test set has 10000 images) under the simplest attack (FGSM). 
'''

parser = argparse.ArgumentParser()

parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--save-path", type=str, default='./CIFAR10-conv.pt')
parser.add_argument("--load", action="store_true")
parser.add_argument("--load-path", type=str, default='./CIFAR10-conv.pt')
parser.add_argument("--test", action="store_true")  # default: attack
parser.add_argument("--attack", choices=['FGSM', 'IPGD'], default='FGSM')
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--train-epoch", type=int, default=10)
parser.add_argument("--optimizer", type=str, default='SGD', choices=['Adam', 'SGD'])
parser.add_argument("--adv-ratio", type=float, default=0., help="ratio of advrserial examples in whole training dataset")

args = parser.parse_args()

CUDA = torch.cuda.is_available() and (not args.no_cuda)


def load_CIFAR10(batch_size):
    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), #?
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 全体cifar10的均值和方差
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True
                                              , download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False
                                             , download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def train(model, train_loader, optimizer, adv_ratio, criterion, attack_method):
    correct = 0
    total = 0
    train_loss = 0.
    scheduler.step()
    for j, data in enumerate(tqdm(train_loader)):
        images, labels = data  # 分别是(N_batch, 3, 32, 32)和（N_batch, 32）

        if CUDA:
            images, labels = images.cuda(), labels.cuda()

        if torch.rand(1).item() < adv_ratio:
            images = IPGD(model, images, labels, criterion=criterion, CUDA=CUDA)

        pred = model(images)

        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)  # dim=1  predicted是最大值的下标
        correct += predicted.eq(labels.data).sum().item()
        total += len(labels)
        acc = correct / total
        print('\nepoch:', i, 'train loss:', train_loss / (j + 1), 'accuracy:', acc)


if __name__ == '__main__':
    model = ResNet(BasicBlock, [2, 2, 2, 2])

    train_loader, test_loader = load_CIFAR10(args.batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    if CUDA:
        deviceIDs = [0]
        # deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1,
        #                                 maxMemory=0.1, excludeID=[], excludeUUID=[])
        print('available cuda device ID(s):', deviceIDs)
        torch.cuda.set_device(deviceIDs[0])
        model.cuda()
        criterion = criterion.cuda()

    if args.load:
        # load state_dict
        model.load_state_dict(torch.load(args.load_path))
    else:
        # train
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), )
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)

        for i in range(args.train_epoch):
            train(model=model, train_loader=train_loader, optimizer=optimizer,
                  criterion=criterion, adv_ratio=args.adv_ratio)

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

        if args.test:
            pred = model(images)  # test
        else:
            if args.attack == 'FGSM':
                pred = model(FGSM(model, images, labels, criterion=criterion, CUDA=CUDA))  # attack
            elif args.attack == 'IPGD':
                pred = model(IPGD(model, images, labels, criterion=criterion, CUDA=CUDA))

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(labels.data).sum().item()
        total += len(labels)
        acc = correct / total
        print('correct:', correct, 'total:', total, 'accuracy:', acc)
