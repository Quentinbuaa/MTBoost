import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2

class MyDataset():
    def __init__(self, idx = 'cifar10'):
        self.idx = idx
        self.batch_size = 128

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_loaders(self):   # need to be implemented.
        return None

    def get_labels(self):   # need to be implemented.
        return None

class SVHNDatset(MyDataset):
    def __init__(self):
        super().__init__('svhn')
        self.train_transforms = transforms.ToTensor()
        self.test_transforms = transforms.ToTensor()

    def set_augment_mode(self, train_transforms_mode):
        if train_transforms_mode=='auto_augment':
            self.train_transforms = self.__get_autoaugment_transforms()
        if train_transforms_mode=='augmix':
            self.train_transforms=self.__get_automix_transforms()
    def __get_autoaugment_transforms(self):
        TRANSFORMS = transforms.Compose([
            v2.AutoAugment(v2.AutoAugmentPolicy.SVHN),
            transforms.ToTensor()
        ])
        return TRANSFORMS
    def __get_automix_transforms(self):
        TRANSFORMS =  transforms.Compose([
            v2.AugMix(),
            transforms.ToTensor()
        ])
        return TRANSFORMS

    def get_loaders(self):
        # setup data loader
        self.trainset = datasets.SVHN('./data', split='train', download=True, transform=self.train_transforms)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testset = datasets.SVHN('./data', split='test', download=True, transform=self.test_transforms)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader
    def get_labels(self):
        return self.trainset.labels

class CIFAR10Dataset(MyDataset):
    def __init__(self):
        super().__init__('cifar10')
        # setup data loader
        self.mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 Normalization Mean
        self.std = (0.2023, 0.1994, 0.2010)  # CIFAR-10 Normalization Std
        # setup data loader
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def set_augment_mode(self, train_transforms_mode):
        if train_transforms_mode=='auto_augment':
            self.train_transforms = self.__get_autoaugment_transforms()
        if train_transforms_mode=='augmix':
            self.train_transforms=self.__get_automix_transforms()

    def __get_autoaugment_transforms(self):
        TRANSFORMS = transforms.Compose([
            v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return TRANSFORMS

    def __get_automix_transforms(self):
        TRANSFORMS = transforms.Compose([
            v2.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return TRANSFORMS

    def get_loaders(self):
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader

    def get_labels(self):
        return self.trainset.targets

class GTSRBDataset(MyDataset):
    def __init__(self):
        super().__init__('gtsrb')
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Image resize to 32x32
            transforms.ToTensor(),  # Numpy to Tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalization
        ])

    def get_loaders(self):

        self.trainset = datasets.GTSRB(root='./data', split='train', download=True, transform=self.transform)
        # 加载测试集
        self.testset = datasets.GTSRB(root='./data', split='test', download=True, transform=self.transform)
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader
    def get_labels(self):
        return self.trainset

class FashionMNISTDataset(MyDataset):
    def __init__(self):
        super().__init__('fashion_mnist')
        # Define transform for single channel including normalization.
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # resize to 32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalization for single channel.
        ])

    def get_loaders(self):
        #Loading  FashionMNIST datasets.
        self.trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testset = datasets.FashionMNIST('./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader

    def get_labels(self):
        return self.trainset

class ThreeChannelledFashionMNISTDataset(FashionMNISTDataset):
    def __init__(self):
        super().__init__()
        # Define the changes for each channel including normalization.
        # Since VGG11 takes RGB images, here we transfer Gray-image to RGB-image.
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Input of VGG11 is 3x224x224
            transforms.Grayscale(num_output_channels=3),  # for the sake of VGG, converting single channel to 3 channels.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # The normalization parameters are normally used during VGG11 pretraining.
        ])
