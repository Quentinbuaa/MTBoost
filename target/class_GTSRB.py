import argparse
import os, random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from SAC_epoch_ll import sac #find strategy every epoch
import time
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torchvision.models as models
from wideresnet import WideResNet
from torchvision.models import vgg11

'six'
parser = argparse.ArgumentParser()
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
parser.add_argument('--model-dir', default='./SavedCheckpoints',
                    help='directory of model for saving checkpoint')
parser.add_argument('--logs-dir', default='./Logs',
                    help='directory for saving logs')
parser.add_argument('--save-freq', '-s', default=5, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--repeat-time', '-r', default=5, type=int, metavar='N',
                    help='set the number of repeat times')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyDataset():
    def __init__(self, idx = 'cifar10'):
        self.idx = idx

    def get_loaders(self):   # need to be rewritten.
        return None


    def get_labels(self):   # need to be rewritten.
        return None


class SVHNDatset(MyDataset):
    def __init__(self):
        super().__init__('svhn')

    def get_loaders(self):
        # setup data loader
        self.trainset = datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor())
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.testset = datasets.SVHN('./data', split='test', download=True, transform=transforms.ToTensor())
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader
    def get_labels(self):
        return self.trainset.labels

class CIFAR10Dataset(MyDataset):
    def __init__(self):
        super().__init__('cifar10')
        # setup data loader
        mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 训练集的均值
        std = (0.2023, 0.1994, 0.2010)  # CIFAR-10 训练集的标准差
        # setup data loader
        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def get_loaders(self):
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform_train)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_test)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader

    def get_labels(self):
        return self.trainset.targets

class GTSRBDataset(MyDataset):
    def __init__(self):
        super().__init__('gtsrb')
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 将图像大小调整为32x32
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

    def get_loaders(self):

        self.trainset = datasets.GTSRB(root='./data', split='train', download=True, transform=self.transform)
        # 加载测试集
        self.testset = datasets.GTSRB(root='./data', split='test', download=True, transform=self.transform)
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader
    def get_labels(self):
        return self.trainset

class FashionMNISTDataset(MyDataset):
    def __init__(self):
        super().__init__('fashionmnist')
        # 定义适合单通道图像的变换，包括标准化
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 调整大小为32x32
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # 针对单通道图像的标准化
        ])

    def get_loaders(self):
        # 加载 FashionMNIST 数据集
        self.trainset = datasets.FashionMNIST('./data', train=True, download=True, transform=self.transform)
        self.train_loader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)  # 修改为适合的批量大小
        self.testset = datasets.FashionMNIST('./data', train=False, download=True, transform=self.transform)
        self.test_loader = DataLoader(self.testset, batch_size=args.batch_size, shuffle=False)
        return self.trainset, self.train_loader, self.testset, self.test_loader

    def get_labels(self):
        return self.trainset

class ThreeChannelledFashionMNISTDataset(FashionMNISTDataset):
    def __init__(self):
        super().__init__()
        # 定义适合单通道图像的变换，包括标准化
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # VGG11 输入大小为 224x224
            transforms.Grayscale(num_output_channels=3),  # 转换为 3 通道，以匹配 VGG11 输入
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # VGG11 预训练时的标准化参数
        ])



class SVHNCNN(nn.Module):
    def __init__(self):
        super(SVHNCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道改为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层的输入特征数
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 根据卷积层输出大小计算
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出大小: (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # 输出大小: (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # 输出大小: (batch_size, 128, 4, 4)
        x = x.view(-1, 128 * 4 * 4)  # Flatten: (batch_size, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)
        self.resnet18 = self.resnet18.to(device)

    def forward(self, x):
        return self.resnet18(x)

def CIFAR10Net():
    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.Linear(512,10)
    )
    return model


class VirtualTrainTest():
    def __init__(self, dataset, model, model_idx=1, retry=1):
        self.train_strategy = 'standard'
        self.dataset = dataset
        self.dataset_idx = dataset.idx
        self.model = model
        self.model_idx = model_idx
        self.retry = retry


    def reset_loaders(self):
        self.trainset, self.train_loader, self.testset, self.test_loader =  self.dataset.get_loaders()

    def set_save_path(self):
        self.logging_file = os.path.join(args.logs_dir, f'{self.dataset_idx}_model.log')
        self.checkpoint_file = os.path.join(args.model_dir, f'{self.dataset_idx}_model_{self.model_idx}_{self.train_strategy.upper()}_repeat_{self.retry}.pt')

    def adjust_learning_rate(self, optimizer, epoch):
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
    def run_training(self):
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        start_time = time.time()
        for epoch in range(1, args.retrain_epochs + 1):
            # adjust learning rate for SGD
            self.adjust_learning_rate(optimizer, epoch)
            self.train(optimizer, epoch)
            # reward_list = train_adv1(args, model, device, optimizer, epoch)
            # data_array = np.array([reward_list['episode'], reward_list['reward']])
            # np.save(os.path.join(args.model_dir,'episodereward{}.npy'.format(epoch)), data_array)
            # print('============================SAC=================================')
            # eval_test_adv(model, device, test_loader)
            # print('================================================================')
        end_time = time.time()
        self.execution_time = (end_time - start_time) / 60
        print(f"{self.dataset_idx}_model_{self.model_idx}_{self.train_strategy}_{self.retry} 函数的运行时间为: {self.execution_time} mins")
        self.save_weights()
        #self.eval_test_adv_worstk()

    def train(self, optimizer, epoch): #需要重写
        pass

    def eval_test_adv_worstk(self):
        self.model.eval()
        correct = 0
        test_accuracy_rob_list = []
        test_accuracy_mts_list = []
        a_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_accuracy = correct / len(self.test_loader.dataset)
            test_accuracy*=100
            print('test Accuracy (standard): {:.2f}'.format(test_accuracy))
            for i in range(400):
                print(i)
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
                for data, target in self.test_loader:
                    data, target = data.to(device), target.to(device)
                    output = self.model(data)
                    output_adv = self.model(self.transform_MR6(data, a))
                    pred = output.max(1, keepdim=True)[1]
                    pred_adv = output_adv.max(1, keepdim=True)[1]
                    correct_rob += pred_adv.eq(target.view_as(pred_adv)).sum().item()
                    correct_mts += pred_adv.eq(pred.view_as(pred_adv)).sum().item()
                test_accuracy_rob = correct_rob / len(self.test_loader.dataset)
                test_accuracy_mts = correct_mts / len(self.test_loader.dataset)
                test_accuracy_rob_list.append(test_accuracy_rob)
                test_accuracy_mts_list.append(test_accuracy_mts)
            test_accuracy_rob = min(test_accuracy_rob_list) * 100
            test_accuracy_mts = min(test_accuracy_mts_list) * 100
            print('test Accuracy (Robust): {:.2f}'.format(test_accuracy_rob))
            print('test Accuracy (MTS): {:.2f}'.format(test_accuracy_mts))

            self.logging(self.dataset_idx, self.train_strategy, self.model_idx, self.retry, time.time(), test_accuracy, test_accuracy_rob, test_accuracy_mts, self.execution_time)

    def save_weights(self):
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def logging(self, *args):
        args = list(map(str, args))
        info = '\t'.join(args)+'\n'
        with open(self.logging_file, 'a') as f:
            f.write(info)

    def transform_MR6(self, image, a):
        image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                          scale=a[3], shear=[a[4], a[5]])
        return image

class STD_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy='Standard'

    def train(self, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.item()))

class W_of_10_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'W10'
        self.k = 10

    def train(self, optimizer, epoch):
        '''Train with worst of 10 strategy'''
        self.model.eval()
        strategy = self.worstof10_ll()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            #output = self.model(data)
            output_adv = self.model(self.transform_MR6(data, strategy))
            #loss1 = F.cross_entropy(output, target)
            # loss2 = (1.0 / args.batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
            loss3 = F.cross_entropy(output_adv, target)
            loss = loss3# + loss1
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))

    def worstof10_ll(self):
        rob_list = []
        a10 = []

        for i in range(self.k):
            a0 = round(random.uniform(-30, 30), 2)
            a1 = round(random.uniform(-3, 3), 2)
            a2 = round(random.uniform(-3, 3), 2)
            a3 = round(random.uniform(0.8, 1.2), 2)
            a4 = round(random.uniform(-10, 10), 2)
            a5 = round(random.uniform(-10, 10), 2)
            a = [a0, a1, a2, a3, a4, a5]
            a10.append(a)
            training_rob_loss = self.eval_train_adv_loss_ll(a)
            rob_list.append(training_rob_loss)
        strategy = a10[rob_list.index(max(rob_list))]
        return strategy

    def eval_train_adv_loss_ll(self, strategy):
        train_loss_rob = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(device), target.to(device)
                output_adv = self.model(self.transform_MR6(data, strategy))
                train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
            # print('train loss (Robust): {:.2f}'.format(train_loss_rob))
        return train_loss_rob

class W10_KL_TrainTest(W_of_10_TrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'w10kl'
        self.k = 10

    def __soft_kl_loss(self, output_adv, output):
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        T = 2.0  # 温度参数
        output_soft = F.log_softmax(output_adv / T, dim=1)
        target_soft = F.softmax(output / T, dim=1).clamp(min=1e-7)
        kl_loss = criterion_kl(output_soft, target_soft) * (T * T)
        return kl_loss

    def __kl_loss(self, output_adv, output):
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        output_soft = F.log_softmax(output_adv, dim=1)
        target_soft = F.softmax(output, dim=1).clamp(min=1e-10)
        kl_loss = criterion_kl(output_soft, target_soft)
        return kl_loss

    def train(self, optimizer, epoch):
        self.model.eval()
        strategy = self.worstof10_ll()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.model(data)
            output_adv = self.model(self.transform_MR6(data, strategy))
            # loss = F.cross_entropy(output_adv, target)
            # loss = F.cross_entropy(output, target) + F.cross_entropy(output_adv, target)
            #kl_loss = self.__soft_kl_loss(output_adv, output)
            kl_loss = self.__kl_loss(output_adv, output)
            loss = F.cross_entropy(output, target) + kl_loss
            #       criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1)).clamp(min=1e-10)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))

class SAC_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'sac'
        self.test_size = 0.01
        self.step = 400

    def random_val_loader1(self, random_state):
        # random_state = random.randint(0,1000)
        labels = self.dataset.get_labels() # 这里需要增加一个函数了
        train_idx, val_idx = train_test_split(range(len(labels)), test_size=self.test_size, random_state=random_state)
        # print(val_idx)
        # 创建 SubsetRandomSampler 实例
        val_sampler = SubsetRandomSampler(val_idx)
        # 创建 DataLoader 来加载数据
        val_loader = DataLoader(self.trainset, batch_size=args.batch_size, sampler=val_sampler)
        return val_loader


    def train(self, optimizer, epoch):
        '''this is training with sac'''
        val_loader = self.random_val_loader1(epoch)
        criterion_kl = nn.KLDivLoss(size_average=False)
        strategy, reward_list = sac(self.model, val_loader, self.step)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # strategy = sac(model, data, target)
            self.model.train()
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = self.model(data)
            output_adv = self.model(self.transform_MR6(data, strategy))
            # loss = F.cross_entropy(output_adv, target)
            # loss = F.cross_entropy(output, target) + F.cross_entropy(output_adv, target)
            # loss = F.cross_entropy(output, target) + (1.0 / args.batch_size) * \
            #        criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
            # loss = F.cross_entropy(output, target) + F.cross_entropy(output_adv, target) +  \
            # criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
            loss = F.cross_entropy(output, target) + F.cross_entropy(output_adv, target) + \
                   criterion_kl(F.log_softmax(output_adv, dim=1),
                                F.softmax(output, dim=1).clamp(min=1e-10)) / args.batch_size
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        # return reward_list
        # return agent

def get_cifar10_model(model_idx):
    cifar10_model_dict={
        1:CIFAR10Net().to(device),
        2: WideResNet(40, 10, 2, 0.3).to(device)
    }
    args.lr=0.005
    return cifar10_model_dict[model_idx]

def vgg_for_gtsrb():
    model = vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 43)
    return model
def vgg_for_fashionmnist():
    model = vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)
    return model
def get_gtsrb_model(model_idx):
    gtsrb_mode_dict={
        1: WideResNet(40, 43, 2, 0.3).to(device),
        2: vgg_for_gtsrb().to(device)
    }
    gtsrb_lr_dict = {
        1: 0.01,
        2: 1e-2
    }
    args.lr= gtsrb_lr_dict[model_idx]
    return gtsrb_mode_dict[model_idx]

def get_fashionmnist_model(model_idx):
    fashionmnist_model_dict = {
        1: FashionMNISTCNN().to(device),
        2: vgg_for_fashionmnist().to(device)
    }
    fashionmnist_lr_dict = {
        1: 0.01,
        2: 0.005
    }
    args.lr=fashionmnist_lr_dict[model_idx]
    return fashionmnist_model_dict[model_idx]


def get_model(dataset_idx, model_idx):
    dataset_idx_get_model_dict = {
        'svhn':None,
        'cifar10': get_cifar10_model,
        'fashion_mnist': get_fashionmnist_model,
        'gtsrb':get_gtsrb_model
    }
    get_func = dataset_idx_get_model_dict[dataset_idx]
    model = get_func(model_idx)
    return model
def get_dataset(dataset_idx, model_idx = 1):
    first_dataset_idx_dict = {
        'svhn':SVHNDatset(),
        'cifar10': CIFAR10Dataset(),
        'fashion_mnist': FashionMNISTDataset(),
        'gtsrb':GTSRBDataset()
    }
    second_dataset_idx_dict = {
        'svhn': SVHNDatset(),
        'cifar10': CIFAR10Dataset(),
        'fashion_mnist': ThreeChannelledFashionMNISTDataset(),
        'gtsrb': GTSRBDataset()
    }
    if model_idx == 1:
        return first_dataset_idx_dict.get(dataset_idx)
    else:
        return second_dataset_idx_dict[dataset_idx]

def TrainStrategyFactory(strategy, dataset_idx, model_idx, retry):
    dataset = get_dataset(dataset_idx, model_idx)
    model = get_model(dataset_idx, model_idx)
    if strategy == 'STD':
        Operator = STD_TrainTest(dataset, model, model_idx, retry)
    if strategy == 'RDM':
        Operator = W_of_10_TrainTest(dataset, model, model_idx, retry)
        Operator.train_strategy='random'
        Operator.k = 1
    if strategy == 'W10':
        Operator = W_of_10_TrainTest(dataset, model, model_idx, retry)
    if strategy == 'W10KL':
        Operator = W10_KL_TrainTest(dataset, model, model_idx, retry)
    if strategy == 'SAC':
        Operator = SAC_TrainTest(dataset, model, model_idx, retry)
    Operator.dataset_idx= dataset_idx
    Operator.reset_loaders()
    Operator.set_save_path()
    return Operator
def exp(total_retry_num=5):
    dataset_idx = 'gtsrb' #['svhn', 'cifar10','gtsrb','fashion_mnist']
    for model_idx in [2]:  # [1,2]
        #for train_strategy in ['STD','RDM', 'W10', 'W10KL','SAC']: #['STD','RDM', 'W10', 'W10KL','SAC']
        for train_strategy in ['W10KL']:
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.run_training()
                operator.eval_test_adv_worstk()

if __name__ == '__main__':
    args.retrain_epochs = 100        #100
    total_retry_num=5             #5
    exp(total_retry_num)


