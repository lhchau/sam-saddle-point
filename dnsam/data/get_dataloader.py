import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

def get_cifar10(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train, data_val = random_split(
        dataset=torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'num_classes': len(data_test.classes)
    }

def get_cifar100(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train, data_val = random_split(
        dataset=torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'num_classes': len(data_test.classes)
    }

def get_mnist(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    data_train, data_val = random_split(
        dataset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'num_classes': len(data_test.classes)
    }

def get_fashion_mnist(
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # normalize inputs
        Cutout()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # normalize inputs
    ])
    data_train, data_val = random_split(
        dataset=torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train),
        lengths=split,
        generator=torch.Generator().manual_seed(42)
    )
    data_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    
    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'test_dataloader': test_dataloader,
        'num_classes': len(data_test.classes)
    }

# Data
def get_dataloader(
    dataset='cifar10',
    batch_size=128,
    num_workers=4,
    split=(0.8, 0.2)    
):
    print('==> Preparing data..')
    
    if dataset == 'cifar10':
        return get_cifar10(batch_size, num_workers, split)
    elif dataset == 'cifar100':
        return get_cifar100(batch_size, num_workers, split)
    elif dataset == 'mnist':
        return get_mnist(batch_size, num_workers, split)
    elif dataset == 'fashion_mnist':
        return get_fashion_mnist(batch_size, num_workers, split)
