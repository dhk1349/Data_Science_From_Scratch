import torch
import torchvision
import torchvision.transforms as transforms


def get_train_loader(path="", batch_size_train=64):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)
    return train_loader


def get_test_loader(path="", batch_size_test=1000):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(path, train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    return test_loader


def cifar10_loader(path, train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if train == True:
        dataset = torchvision.datasets.CIFAR10(root=path, train=True,
                                               download=True, transform=transform)
    else:
        dataset = torchvision.datasets.CIFAR10(root=path, train=False,
                                               download=True, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    return loader


def cifar10_get_class():
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes


