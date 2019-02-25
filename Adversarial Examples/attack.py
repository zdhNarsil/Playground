import torch
import torchvision
import numpy as np


def FGSM(model, input, label, eta=1., criterion=torch.nn.CrossEntropyLoss(),
         CUDA=False, isIPGD=False):
    model.eval()

    if not isIPGD:
        input.requires_grad = True  # 单独使用FGSM的话就设为True
    if CUDA:
        model = model.cuda()
        input = input.cuda()
        criterion = criterion.cuda()

    model.zero_grad()
    loss = criterion(model(input), label)
    # loss.backward()

    grad_sign = torch.autograd.grad(loss, input, only_inputs=False)[0].sign()
    # grad_sign = input.grad.sign()
    # input.detach()
    return input + eta * grad_sign


def clip(perturb, norm, eps, CUDA=False):
    avoid_zero_div = torch.tensor(1e-12)
    one = torch.tensor(1.0)
    if CUDA:
        # eps.cuda()
        avoid_zero_div = avoid_zero_div.cuda()
        one = one.cuda()

    if norm == np.inf:
        perturb = torch.clamp(perturb, -eps, eps)
    else:
        batch_size = perturb.shape[0]
        normalize = torch.norm(perturb.reshape(batch_size, -1), p=norm, dim=1, keepdim=True) # (N_batch, 1)
        normalize = torch.max(normalize, avoid_zero_div)
        normalize.unsqueeze_(dim=-1)  # (N_batch, 1, 1)
        normalize.unsqueeze_(dim=-1)  # (N_batch, 1, 1, 1)
        perturb *= torch.min(one, eps / normalize)

    return perturb


def IPGD(model, input, label, eta=1., eps=6, norm=np.inf, criterion=torch.nn.CrossEntropyLoss(),
         CUDA=False, num_attack=15):
    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    model.eval()
    perturb = torch.zeros_like(input)
    if CUDA:
        model = model.cuda()
        perturb = perturb.cuda()
        criterion = criterion.cuda()

    input.requires_grad = True
    for i in range(num_attack):
        output = FGSM(model, input + perturb, label, eta=eta, criterion=criterion, CUDA=CUDA, isIPGD=True)
        perturb = output - input
        perturb = clip(perturb, norm=norm, eps=eps, CUDA=CUDA)

    return input + perturb