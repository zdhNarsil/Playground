import torch
import torchvision


def FGSM(model, input, label, eta=1., CUDA=False):
    model.eval()

    input.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    if CUDA:
        criterion = criterion.cuda()

    model.zero_grad()
    loss = criterion(model(input), label)
    loss.backward()

    grad_sign = input.grad.sign()
    input.detach()
    return input + eta * grad_sign


def PGD(model, input, label, eta=1., CUDA=False):
