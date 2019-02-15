import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import math
from tqdm import tqdm
from collections import OrderedDict


def idx2onehot(idx, n):
    idx = idx.reshape(idx.size(0), 1)

    assert idx.size(1) == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx, 1)  # 注意scatter_要求得是torch的longint数组

    return onehot


def normalizevector(r):  ###
    n_batch = r.shape[0]

    denominator = torch.norm(r.reshape(n_batch, -1), p=float('inf'), dim=1, keepdim=True).reshape(n_batch, 1, 1, 1)
    r /= (denominator + 1e-12)

    denominator = 1e-12 + torch.norm(r.reshape(n_batch, -1), p='fro', dim=1).reshape(n_batch, 1, 1, 1)
    return r / denominator


def crossentropy(logits, label):
    # both input is (N_batch, Class)
    return -torch.mean(torch.sum(label * torch.log(logits + 1e-8), dim=1))


def kldivergence(logits, label):
    return torch.mean(torch.sum(label * (torch.log(label + 1e-8) - torch.log(logits + 1e-8)), dim=1))


def eval_one_epoch(net, batch_generator, use_CUDA):
    net.eval()
    pbar = tqdm(batch_generator)
    clean_accuracy = AvgMeter()

    pbar.set_description('Evaluating')#
    for (data, label) in pbar:
        if use_CUDA:
            data = data.cuda()
            label = label.cuda()

        with torch.no_grad():
            pred = net(data)
            acc = torch_accuracy(pred, label, (1,))
            clean_accuracy.update(acc[0].item())

        pbar_dic = OrderedDict()
        pbar_dic['CleanAcc'] = '{:.2f}'.format(clean_accuracy.mean)
        pbar.set_postfix(pbar_dic)
    return clean_accuracy.mean


def torch_accuracy(output, target, topk=(1,)):
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, largest=True, sorted=True)  # torch.topk()
    pred = pred.t()  # 转置

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


class AvgMeter(object):
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num

