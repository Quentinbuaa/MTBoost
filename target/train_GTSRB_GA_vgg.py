from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
from deap import base, creator, tools, algorithms
from torchvision.transforms import v2
import imgaug.augmenters as iaa
import time
import numpy as np
from torchvision.models import vgg11
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
#from train_SVHN_GA_cnn import generate_individual,select_cross_mutate
parser = argparse.ArgumentParser(description='PyTorch Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test_pgd-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--retrain-epochs', type=int, default=100, metavar='N',
                    help='number of epochs to retrain')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./GA/GTSRB',
                    help='directory of model for saving checkpoint')
args = parser.parse_args()
# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.empty_cache()
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图像大小调整为32x32
    transforms.ToTensor(),        # 将图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

trainset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
# 加载测试集
testset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform)
# 创建数据加载器
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
def VGG():
    # 加载预训练的 VGG11 模型
    model = vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 43)
    return model
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 55:
        lr = args.lr * 0.1
    if epoch >= 75:
        lr = args.lr * 0.01
    if epoch >= 90:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def transform(image,a):
     image = TF.affine(image, angle=a[0]*0.1 * 30, translate=[a[1]*0.1 * 3, 0],
                     scale=a[2]*0.1 * 0.1+1, shear=[a[3] * 0.01, 0])
     # image = TF.adjust_contrast(image,a[4]*0.02 +1) #范围太大GA训练特别不好的图片
     # image = TF.adjust_brightness(image,a[5]*0.02 +1)#0.8-1.2
     return image


def transform_snow(data,a):
    # 创建增强器的序列
    seq = iaa.Sequential([
        iaa.FastSnowyLandscape(
            lightness_threshold=140,
            lightness_multiplier=2.5
        )
    ])

    # 假设输入张量形状为 (128, 128, 4, 4)
    # 先将 data 转换为 NumPy 数组
    data_np = data.cpu().numpy()  # shape: (128, 128, 4, 4)

    # 初始化一个空的数组来存储变换后的图像
    batch_transformed = np.zeros_like(data_np)  # (128, 128, 4, 4)

    # 遍历每个图像
    for i in range(data_np.shape[0]):  # 128
        for j in range(data_np.shape[1]):  # 128
            single_image = data_np[i, j]  # 选取 (4, 4) 的图像

            # 将图像数据类型转换为 uint8
            if single_image.max() <= 1.0:
                single_image = (single_image * 255).astype(np.uint8)
            else:
                single_image = single_image.astype(np.uint8)

                # 应用增强
            transformed_image = seq.augment_image(image=single_image)

            # 将增强后的图像转换为 float32 并归一化到 [0, 1]
            transformed_image = transformed_image.astype(np.float32) / 255.0

            # 将变换后的图像存回 batch_transformed
            batch_transformed[i, j] = transformed_image

            # 转换回 PyTorch 张量
    transformed_data = torch.from_numpy(batch_transformed).to(data.device)

    return transformed_data

class my_transoform():
    def __init__(self, strategy):
        super().__init__()
        self.strategy = strategy
        self.p = 0.5
    def __call__(self, image):
        if torch.rand(1) < self.p:
            return TF.affine(image, angle=self.strategy[0], translate=[self.strategy[1], self.strategy[2]],
                         scale=self.strategy[3], shear=[self.strategy[4], self.strategy[5]])
        else:
            return image

def train_adv(args, model, device,optimizer, epoch,strategy):
    model.train()
    transform_train = transforms.Compose([
        my_transoform(strategy),
        transforms.ToTensor(),
    ])
    trainset1 = datasets.GTSRb('./data', split='train', download=True, transform=transform_train)
    train_loader1 = torch.utils.data.DataLoader(trainset1, batch_size=args.batch_size, shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))
def train_adv_epoch_ll(args, model, device, optimizer, epoch):
    criterion_kl = nn.KLDivLoss(size_average=False)
    # strategy = worstof10_ll(model)
    kl = False
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        strategy = select_cross_mutate(model, data, target, kl)

        model.train()
        optimizer.zero_grad()

        output = model(data)
        output_adv = model(transform_MR6(data, strategy))
        loss1 = F.cross_entropy(output, target)
        # loss2 = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
        loss3 = F.cross_entropy(output_adv, target)
        loss = loss1 + loss3

        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))
def train_adv_epoch_kl(args, model, device, optimizer, epoch):
    criterion_kl = nn.KLDivLoss(size_average=False)
    strategy = worstof10_ll(model)
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output_adv = model(transform_MR6(data, strategy))
        loss = F.cross_entropy(output, target) + (1.0 / args.batch_size) * \
               criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
        # loss = F.cross_entropy(output, target) + (1.0 / args.batch_size) * \
        #        criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss.item()))


def augmixtrain():
    # init model, Net() can be also used here for training
    model = VGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.retrain_epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        TRANSFORMS =  transforms.Compose([
            v2.AugMix(),
            transforms.ToTensor()
        ])
        trainset = datasets.GTSRb('./data',  split='train', download=True,transform=TRANSFORMS)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train(args, model, device, train_loader, optimizer, epoch)
        # evaluation on natural examples
        print('================================================================')
        # eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save(model.state_dict(),
               os.path.join(model_dir, 'vgg_AUGMIX_model-nn-epoch{}.pt'.format(epoch)))
def autoaugmenttrain():
    # init model, Net() can be also used here for training
    model = VGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.retrain_epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        TRANSFORMS = transforms.Compose([
            v2.AutoAugment(v2.AutoAugmentPolicy.SVHN),
            transforms.ToTensor()
        ])
        trainset = datasets.GTSRb('./data',  split='train', download=True,transform=TRANSFORMS)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train(args, model, device, train_loader, optimizer, epoch)
        # evaluation on natural examples
        print('================================================================')
        # eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save(model.state_dict(),
               os.path.join(model_dir, 'cnn_AUTOAUGMENT_model-nn-epoch{}.pt'.format(epoch)))



def normaltrain(model):
    # init model, Net() can be also used here for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # normal train
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        # evaluation on natural examples
        print('================================================================')
        # eval_test(model, device, test_loader)
        print('================================================================')
    torch.save(model.state_dict(),
               os.path.join(model_dir, 'vgg_model-nn-epoch{}.pt'.format(epoch)))

def randomtrain():
    model = VGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # normal train
    for epoch in range(1, args.retrain_epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        TRANSFORMS = transforms.Compose([
            transforms.RandomAffine(30,[0.1,0.1],[0.9,1.1],[1,1]),
            # transforms.ColorJitter([0.1,1.9],[0.8,1.2]),
            transforms.ToTensor()
        ])
        trainset = datasets.GTSRb('./data', split='train', download=True, transform=TRANSFORMS)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        # eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save(model.state_dict(),
               os.path.join(model_dir, 'RANDOM_6_model-nn-epoch{}.pt'.format(epoch)))

def GAretrain():
    model = VGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.retrain_epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        strategy = select_cross_mutate(model)
        train_adv(args, model, device, train_loader, optimizer, epoch,strategy)
        # train_adv_CORRECT(args, model, device, train_loader, optimizer, epoch,strategy)
        # evaluation on natural examples
        print('============================GA==================================')
        # eval_test_adv(model, device, test_loader)
        print('================================================================')


    torch.save(model.state_dict(),
                os.path.join(model_dir, 'GA_losskl-acc-rob_train_model-nn-epoch{}.pt'.format(epoch)))
    # torch.save(optimizer.state_dict(),
    #             os.path.join(model_dir, 'GA_retrain_opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
def GAretrain_ll(model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # -----------------------------------------------------------------------
        train_adv_epoch_ll(args, model, device,optimizer, epoch)
        # evaluation on natural examples
        print('===========================GA==================================')
        eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, 'vgg_GA_l_ll_10_5_retrain_opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
def GAretrain_kl(model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # -----------------------------------------------------------------------
        train_adv_epoch_kl(args, model, device,optimizer, epoch)
        # evaluation on natural examples
        print('===========================GA==================================')
        eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_dir, 'vgg_GA_kl_retrain_opt-nn-checkpoint_epoch{}.tar'.format(epoch)))
#------------------------------------w-10-------------------------------------------------------

def eval_train_adv1(model, device, train_loader,a):
    model.eval()
    correct = 0
    correct_rob = 0
    correct_mts = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_adv = model(transform_MR6(data,a))
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            correct_mts += pred_adv.eq(pred.view_as(pred_adv)).sum().item()

    training_accuracy = correct / len(train_loader.dataset)
    training_accuracy_rob = correct_rob / len(train_loader.dataset)
    training_accuracy_mts = correct_mts / len(train_loader.dataset)
    # print('Training Accuracy: {}'.format(training_accuracy))
    # print('Training Accuracy (Robust): {}'.format(training_accuracy_rob))
    # print('Training Accuracy (MTS): {}'.format(training_accuracy_mts))
    return training_accuracy,training_accuracy_rob,training_accuracy_mts
def transform_MR6(image, a):
    image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                      scale=a[3], shear=[a[4], a[5]])
    return image
def worstof10_ll(model):
    k = 10
    rob_list = []
    a10 = []
    for i in range(k):
        a0 = round(random.uniform(-30, 30), 2)
        a1 = round(random.uniform(-3, 3), 2)
        a2 = round(random.uniform(-3, 3), 2)
        a3 = round(random.uniform(0.8, 1.2), 2)
        a4 = round(random.uniform(-10, 10), 2)
        a5 = round(random.uniform(-10, 10), 2)
        a = [a0,a1,a2,a3,a4,a5]
        a10.append(a)
        training_rob_loss = eval_train_adv_loss_ll(model, device,a)
        rob_list.append(training_rob_loss)
    strategy = a10[rob_list.index(max(rob_list))]
    return strategy
def worstof10_kl(model):
    k = 10
    rob_list = []
    a10 = []
    for i in range(k):
        a0 = random.uniform(-30, 30)
        a1 = random.uniform(-3, 3)
        a2 = random.uniform(-3, 3)
        a3 = random.uniform(0.8, 1.2)
        a4 = random.uniform(-10, 10)
        a5 = random.uniform(-10, 10)
        a = [a0,a1,a2,a3,a4,a5]
        a10.append(a)
        training_rob_loss = eval_train_adv_loss_kl(model, device,a)
        rob_list.append(training_rob_loss)
    strategy = a10[rob_list.index(max(rob_list))]
    return strategy
#------------------------------------------------------------------------------------------------------------
def w10retrain_epoch_ll():
    # init model, Net() can be also used here for training
    model = VGG().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # -----------------------------------------------------------------------
        train_adv_epoch_ll(args, model, device,optimizer, epoch)
        # evaluation on natural examples
        print('===========================W10==================================')
        eval_test_adv(model, device, test_loader)
        print('================================================================')

    torch.save(model.state_dict(),
                os.path.join(model_dir, 'vgg_W10_l_l_epoch_6_10_model-nn-epoch{}.pt'.format(epoch)))
def w10retrain_epoch_kl(model):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        # -----------------------------------------------------------------------
        train_adv_epoch_kl(args, model, device,optimizer, epoch)
        # evaluation on natural examples
        print('===========================W10==================================')
        eval_test_adv(model, device, test_loader)
        print('================================================================')
    torch.save(model.state_dict(),
                os.path.join(model_dir, 'vgg_W10_l_kl_epoch_6_10_model-nn-epoch{}.pt'.format(epoch)))
#------------------------------------------------------------------------------------------------------------
def eval_train_adv_loss_ll(model, device,strategy):
    model.eval()
    train_loss_rob = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output_adv = model(transform_MR6(data, strategy))
            train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
        # print('train loss (Robust): {:.2f}'.format(train_loss_rob))
    return train_loss_rob
def eval_train_adv_loss_kl(model, device,strategy):
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    train_loss_rob = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_adv = model(transform_MR6(data, strategy))
            train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
    # print('train loss (Robust): {}'.format(train_loss_rob))
    return train_loss_rob
def eval_test_adv(model, device, test_loader):
    model.eval()
    correct = 0
    correct_rob = 0
    correct_mts = 0
    a0 = random.uniform(-30, 30)
    a1 = random.uniform(-3, 3)
    a2 = random.uniform(-3, 3)
    a3 = random.uniform(0.8, 1.2)
    a4 = random.uniform(-10, 10)
    a5 = random.uniform(-10, 10)
    a = [a0, a1, a2, a3, a4, a5]
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_adv = model(transform_MR6(data, a))
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
            correct_mts += pred_adv.eq(pred.view_as(pred_adv)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy_rob = correct_rob / len(test_loader.dataset)
    test_accuracy_mts = correct_mts / len(test_loader.dataset)
    print('test_pgd Accuracy: {}'.format(test_accuracy))
    print('test_pgd Accuracy (Robust): {}'.format(test_accuracy_rob))
    print('test_pgd Accuracy (MTS): {}'.format(test_accuracy_mts))
    return test_accuracy,test_accuracy_rob,test_accuracy_mts

def eval_test_adv_snow(model, device, test_loader):
    model.eval()
    correct = 0
    correct_rob = 0
    # a0 = random.uniform(-30, 30)
    # a1 = random.uniform(-3, 3)
    # a2 = random.uniform(-3, 3)
    # a3 = random.uniform(0.9, 1.1)
    # a4 = random.uniform(-1, 1)
    # a5 = random.uniform(-1, 1)
    # a = [a0, a1, a2, a3, a4, a5]
    a = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output_adv = model(transform_snow(data, a))
            pred = output.max(1, keepdim=True)[1]
            pred_adv = output_adv.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    test_accuracy_rob = correct_rob / len(test_loader.dataset)
    print('test_pgd Accuracy: {}'.format(test_accuracy))
    print('test_pgd Accuracy (Robust): {}'.format(test_accuracy_rob))
    return test_accuracy,test_accuracy_rob
def eval_test_adv_worstk(model, device, test_loader, checkpoint_file = None):
    model.load_state_dict(torch.load(checkpoint_file))
    model = model.to(device)
    model.eval()
    # model.load_state_dict(torch.load('./GA/GTSRB/vgg_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/GTSRB/densenet43_W10_6_10_1.1_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/GTSRB/vgg_W10_l_kl_epoch_6_10_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./RL/GTSRB/vgg_SAC_gtsrb2_NO_l_kll_step400_val0.01_epoch100.pt'))
    # model.load_state_dict(torch.load('./RL/GTSRB/vgg_PPO_gtsrb2_l_kll_step400_val0.01_epoch100.pt'))
    # model.load_state_dict(torch.load('./RL/GTSRB/vgg_DDPG_gtsrb2_l_kll_step400_val0.01_epoch100.pt'))
    # model.load_state_dict(torch.load('./RL/GTSRB/vgg_TD3_gtsrb2_l_kll_step400_val0.01_epoch100.pt'))
    # checkpoint = torch.load(os.path.join(model_dir, 'vgg_GA_l_ll_10_5_retrain_opt-nn-checkpoint_epoch100.tar'))
    # model.load_state_dict(checkpoint['model_state_dict'])
    correct = 0
    test_accuracy_rob_list = []
    test_accuracy_mts_list = []
    a_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_accuracy = correct / len(test_loader.dataset)
        print('test_pgd Accuracy: {:.2f}'.format(test_accuracy*100))

        for i in range(400):
            random.seed(i)
            correct_rob = 0
            correct_mts = 0
            a0 = round(random.uniform(-30, 30), 2)
            a1 = round(random.uniform(-3, 3), 2)
            a2 = round(random.uniform(-3, 3), 2)
            a3 = round(random.uniform(0.8, 1.2), 2)
            a4 = round(random.uniform(-10, 10), 2)
            a5 = round(random.uniform(-10, 10), 2)
            a = [a0, a1, a2, a3, a4, a5]
            a_list.append(a)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                output_adv = model(transform_MR6(data, a))
                pred = output.max(1, keepdim=True)[1]
                pred_adv = output_adv.max(1, keepdim=True)[1]
                correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
                correct_mts += pred_adv.eq(pred.view_as(pred_adv)).sum().item()
            test_accuracy_rob = correct_rob / len(test_loader.dataset)
            test_accuracy_mts = correct_mts / len(test_loader.dataset)
            test_accuracy_rob_list.append(test_accuracy_rob)
            test_accuracy_mts_list.append(test_accuracy_mts)
        print('test_pgd Accuracy (Robust): {:.2f}'.format(min(test_accuracy_rob_list)*100))
        print('test_pgd Accuracy (MTS): {:.2f}'.format(min(test_accuracy_mts_list)*100))


def eval_test_adv_grid1(model, device, test_loader):
    model.eval()
    model.load_state_dict(torch.load('GA/SVHN/cnn_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_AUTOAUGMENT_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_AUGMIX_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/densenet43_W10_6_10_1.1_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_W10_l_6_10_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('RL/svhn/cnn_SAC_svhn2_ll_400_epoch100.pt'))
    correct = 0
    test_accuracy_rob_list = []
    a_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_accuracy = correct / len(test_loader.dataset)
        print('test_pgd Accuracy: {}'.format(test_accuracy))
        for a0 in range(-30,31,1):
            for a1 in [-3,-1.5,0,1.5,3]:
                for a2 in [-3, -1.5,0,1.5,3]:
        # for a0 in [-30, -15,0,15, 30]:
        #     for a1 in [-3,-1.5,0,1.5,3]:
        #         for a2 in [-3, -1.5,0,1.5, 3]:
        # for a3 in [0.9,0.95,1,1.05,1.1]:
        #     for a4 in [-1,-0.5,0,0.5,1]:
        #         for a5 in [-1,-0.5,0,0.5,1]:
                    correct_rob = 0
                    a = [a0, a1, a2, 1, 0, 0]
                    # a = [0, 0, 0, a3, a4, a5]
                    a_list.append(a)
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output_adv = model(transform_MR6(data, a))
                        pred_adv = output_adv.max(1, keepdim=True)[1]
                        correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
                    test_accuracy_rob = correct_rob / len(test_loader.dataset)
                    test_accuracy_rob_list.append(test_accuracy_rob)
                # print('strategy:{}'.format(a_list))
        print('test_pgd Accuracy (Robust): {}'.format(test_accuracy_rob_list))
        print('test_pgd Accuracy (Robust): {}'.format(min(test_accuracy_rob_list)))
def eval_test_adv_grid2(model, device, test_loader):
    model.eval()
    model.load_state_dict(torch.load('GA/SVHN/cnn_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_AUTOAUGMENT_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_AUGMIX_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/densenet43_W10_6_10_1.1_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('./GA/SVHN/cnn_W10_l_6_10_model-nn-epoch100.pt'))
    # model.load_state_dict(torch.load('RL/svhn/cnn_SAC_svhn2_ll_400_epoch100.pt'))
    correct = 0
    correct_rob = 0
    correct_mts = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_accuracy = correct / len(test_loader.dataset)
        print('test_pgd Accuracy: {}'.format(test_accuracy))

        for data, target in test_loader:
            correct_batch_rob_list = []
            data, target = data.to(device), target.to(device)
            for a0 in range(-30, 31, 1):
                for a1 in [-3, -1.5, 0, 1.5, 3]:
                    a = [a0, a1, 0, 1, 0, 0]
                    output_adv = model(transform_MR6(data, a))
                    pred_adv = output_adv.max(1, keepdim=True)[1]
                    correct_batch_rob = pred_adv.eq(target.view_as(pred_adv)).sum().item()
                    correct_batch_rob_list.append(correct_batch_rob)
            correct_rob += min(correct_batch_rob_list)
        test_accuracy_rob = correct_rob / len(test_loader.dataset)
        print('test_pgd Accuracy (Robust): {}'.format(test_accuracy_rob))
if __name__ == '__main__':
    start_time = time.time()
    model = VGG().to(device)
    # normaltrain(model)
    # GAretrain(model)
    # GAretrain_kl(model)
    # GAretrain_ll(model)
    # w10retrain_epoch_ll(model)
    w10retrain_epoch_kl(model)
    # randomtrain(model)
    # augmixtrain(model)
    # autoaugmenttrain(model)
    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print("函数的运行时间为: ", execution_time, "min")
    eval_test_adv_worstk(model, device, test_loader)
    # model = VGG().to(device)
    # eval_test_adv_grid2(model, device, test_loader)
    # model = VGG().to(device)
    # eval_test_adv_snow(model, device, test_loader)
