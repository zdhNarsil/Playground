import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import argparse
import GPUtil

from attack import FGSM

'''
Results: 
For 2-linear layers model the test accuracy can get somewhat 95%.
Even for a simple model like this, the test accuracy deduces to 2/10000, 
sometimes 1/10000 (test set has 10000 images) under the simplest attack (FGSM). 
'''

parser = argparse.ArgumentParser()

parser.add_argument("--no-cuda", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--save-path", type=str, default='./MNIST-linear.pt')
parser.add_argument("--load", action="store_true")
parser.add_argument("--load-path", type=str, default='./MNIST-linear.pt')
parser.add_argument("--test", action="store_true")  # default: attack

args = parser.parse_args()

CUDA = torch.cuda.is_available() and (not args.no_cuda)


def load_MNIST():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True
                                            , download=True, transform=transform_train)
    test_data = torchvision.datasets.MNIST(root='./data', train=False
                                           , download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

    return train_loader, test_loader


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 10),
        torch.nn.Softmax(),
    )

    train_loader, test_loader = load_MNIST()

    if CUDA:
        deviceIDs = [0]
        # deviceIDs = GPUtil.getAvailable(order='first', limit=4, maxLoad=0.1,
        #                                 maxMemory=0.1, excludeID=[], excludeUUID=[])
        print('available cuda device ID(s):', deviceIDs)
        torch.cuda.set_device(deviceIDs[0])
        model.cuda()

    if args.load:
        # load state_dict
        model.load_state_dict(torch.load(args.load_path))
    else:
        # train
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), )

        if CUDA:
            criterion = criterion.cuda()

        for i in range(3):
            correct = 0
            total = 0
            train_loss = 0.
            for j, data in enumerate(tqdm(train_loader)):
                images, labels = data

                if CUDA:
                    images, labels = images.cuda(), labels.cuda()
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
                print('\ntrain loss:', train_loss / (j + 1), 'accuracy:', acc)

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

        images = images.reshape((-1, 784))

        if args.test:
            pred = model(images)  # test
        else:
            pred = model(FGSM(model, images, labels, CUDA=CUDA))  # attack

        _, predicted = torch.max(pred.data, 1)
        correct += predicted.eq(labels.data).sum().item()
        total += len(labels)
        acc = correct / total
        print('correct:', correct, 'total:', total, 'accuracy:', acc)