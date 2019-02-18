import torch
import torchvision
from torchvision import transforms


def load_CIFAR10(batch_size, data_dir='./data/cifar10'):
    # transform_train = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # ?
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 全体cifar10的均值和方差
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True
                                              , download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False
                                             , download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def load_MNIST(batch_size, data_dir='./data/mnist',
               transform_train=transforms.Compose([transforms.ToTensor(), ]),
               transform_test=transforms.Compose([transforms.ToTensor(), ])
               ):
    train_data = torchvision.datasets.MNIST(root=data_dir, train=True
                                            , download=True, transform=transform_train)
    test_data = torchvision.datasets.MNIST(root=data_dir, train=False
                                           , download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader
