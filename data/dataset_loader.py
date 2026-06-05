import torch
from torchvision import datasets, transforms


def load_dataset(name, root='./datasets'):
    name = name.lower()
    if name in ['mnist', 'fashionmnist', 'fashion-mnist']:
        tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        if name == 'mnist':
            train = datasets.MNIST(root, train=True, download=True, transform=tfm)
            test = datasets.MNIST(root, train=False, download=True, transform=tfm)
            in_channels = 1
        else:
            train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
            test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
            in_channels = 1
    elif name in ['cifar10', 'cifar-10']:
        tfm_train = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        tfm_test = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])
        train = datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
        in_channels = 3
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return train, test, in_channels, 10
