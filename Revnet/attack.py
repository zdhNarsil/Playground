'''
Results: Revnet防不住FGSM，正确率会从90%掉到10%

'''

import torch
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import revnet
from tqdm import tqdm
import GPUtil
import pdb

parser = argparse.ArgumentParser()

parser.add_argument("--model", metavar="NAME", default='revnet38', help="what model to use")
parser.add_argument("--load", metavar="PATH", help="load a previous model state")
parser.add_argument("--batch-size", default=128, type=int,
                    help="size of the mini-batches")
parser.add_argument("--no-cuda", action="store_true")

args = parser.parse_args()

CUDA = torch.cuda.is_available() and (not args.no_cuda)


def FGSM(model, input, label, eta=1.):
    model.eval()

    input.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss().cuda()

    model.zero_grad()
    loss = criterion(model(input), label)
    loss.backward()

    grad_sign = input.grad.sign()
    input.detach_()  # 没用
    return input + eta * grad_sign


def IPGD(model, input, label, eta=1.):
    pass


def load_CIFAR10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True,
        download=True, transform=transform_train
    )

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False,
        download=True, transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == '__main__':
    model = getattr(revnet, args.model)()

    if CUDA:
        deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1,
                                        maxMemory=0.1, excludeID=[], excludeUUID=[])
        print('available cuda device ID(s):', deviceIDs)
        torch.cuda.set_device(deviceIDs[0])
        model.cuda()

    # load_path = torch.load(args.load)

    exp_id = "cifar_revnet38_2019-01-22_20-56-24"
    load_path = os.path.join("experiments", exp_id,
                             "checkpoints", "cifar_revnet38_2019-01-22_22-58-57.dat")
    model.load_state_dict(torch.load(load_path))

    _, testloader = load_CIFAR10()

    # total_num = 一个testloader的epoch里的图片数量 = 10000
    total_num = total_attack_fail = total_pred_success = 0.
    t = tqdm(testloader, ascii=True)
    for i, data in enumerate(t):
        inputs, labels = data  # labels: (128, )

        if CUDA:
            inputs, labels = inputs.cuda(), labels.cuda()

        attack_pred = model(FGSM(model, inputs, labels))  # (128, 10)
        attack_pred = attack_pred.argmax(dim=1)  # (128,)
        attack_fail = (attack_pred == labels).sum().item()

        pred_success = (model(inputs).argmax(dim=1) == labels).sum().item()

        total_num += len(labels)
        total_attack_fail += attack_fail
        total_pred_success += pred_success

        torch.cuda.empty_cache()  # 没用
        del inputs  # 没用
        # Free the memory used to store activations
        if type(model) is revnet.RevNet:
            model.free()  # 有用！

        print("iter =", i, "batch_size =", args.batch_size)
        print("pred_success_num =", pred_success, "attack_fail_num =", attack_fail)

    print("Original test accuracy:", total_pred_success/total_num,
          "Total attack fail ration: ", total_attack_fail/total_num)
    print("epoch capacity:", total_num)
