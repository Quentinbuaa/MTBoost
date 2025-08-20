import argparse
import torch
from models.SVHNModels import get_svhn_model
from models.CIFAR10Models import get_cifar10_model
from models.GTSRBModels import get_gtsrb_model
from models.FashionMNISTModels import get_fashionmnist_model
from Datasets import SVHNDatset, CIFAR10Dataset, GTSRBDataset, FashionMNISTDataset, ThreeChannelledFashionMNISTDataset
from TrainingStrategies import STD_TrainTest, W_of_10_TrainTest, W10_KL_TrainTest, RL_TrainTest, SENSEI_TrainTest

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


class Transformation:
    def __init__(self):
        self.parameters = {}
        self.parameters_range = {}

    def transform(self, image, parameters):
        image = TF.affine(image, angle=parameters[0], translate=[parameters[1], parameters[2]],
                          scale=parameters[3], shear=[parameters[4], parameters[5]])
        return image

class VirtualTrainTest():
    def __init__(self, dataset, model, model_idx=1, retry=1):
        self.train_strategy = 'standard'
        self.dataset = dataset
        self.dataset_idx = dataset.idx
        self.model = model
        self.model_idx = model_idx
        self.retry = retry

    def set_saving_parameters(self, logs_dir, model_saving_dir):
        self.logs_dir = logs_dir
        self.model_saving_dir = model_saving_dir

    def set_optimizer_parameters(self, lr, momentum, weight_decay=5e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def set_training_epochs(self, training_epoches, log_interval=100):
        self.training_epochs = training_epoches
        self.log_interval = log_interval

    def set_device(self, device):
        self.device = device

    def reset_loaders(self):
        self.trainset, self.train_loader, self.testset, self.test_loader =  self.dataset.get_loaders()

    def set_save_path(self, logs_dir, model_saving_dir):
        self.logging_file = os.path.join(logs_dir, f'{self.dataset_idx}_model.log')
        self.checkpoint_file = os.path.join(model_saving_dir, f'{self.dataset_idx}_model_{self.model_idx}_{self.train_strategy.upper()}_repeat_{self.retry}.pt')

    def adjust_learning_rate(self, optimizer, epoch):
        """decrease the learning rate"""
        lr = self.lr
        if epoch >= 55:
            lr = self.lr * 0.1
        if epoch >= 75:
            lr = self.lr * 0.01
        if epoch >= 90:
            lr = self.lr * 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def run_training(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        start_time = time.time()
        for epoch in range(1, self.training_epochs + 1):
            self.adjust_learning_rate(optimizer, epoch)
            self.train(optimizer, epoch)
        end_time = time.time()
        self.execution_time = (end_time - start_time) / 60
        print(f"{self.dataset_idx}_model_{self.model_idx}_{self.train_strategy}_{self.retry} 函数的运行时间为: {self.execution_time} mins")
        self.save_weights()

    def train(self, optimizer, epoch): #需要重写 Implementation Required
        pass

    def eval_test_adv_worstk(self):
        self.model.eval()
        correct = 0
        test_accuracy_rob_list = []
        test_accuracy_mts_list = []
        a_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            test_accuracy = correct / len(self.test_loader.dataset)
            test_accuracy*=100
            print('test Accuracy (standard): {:.2f}'.format(test_accuracy))
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
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
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

class RL_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'sac'
        self.test_size = 0.01
        self.alpha = 1.0
        self.beta = 1.0
        self.step = 400
        self.ablation_exp = False

    def set_ablation_mode(self):
        self.ablation_exp = True

    def set_test_size(self, test_size):
        self.test_size = test_size

    def set_alpha_beta(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
    def set_rl_strategy(self, strategy = 'sac'):
        self.train_strategy = strategy
        rl_dict ={
            'sac': sac,
            'td3': td3,
            'ddpg':ddpg,
            'ppo': ppo
        }
        self.rl_strategy = rl_dict[strategy]

    def random_val_loader1(self, random_state):
        # random_state = random.randint(0,1000)
        labels = self.dataset.get_labels() # 这里需要增加一个函数了
        train_idx, val_idx = train_test_split(range(len(labels)), test_size=self.test_size, random_state=random_state)
        # print(val_idx)
        # 创建 SubsetRandomSampler 实例
        val_sampler = SubsetRandomSampler(val_idx)
        # 创建 DataLoader 来加载数据
        val_loader = DataLoader(self.trainset, batch_size=self.dataset.batch_size, sampler=val_sampler)
        return val_loader

    def __get_loss(self,output, target,  output_adv, criterion_kl):
        loss_1 = F.cross_entropy(output, target)
        loss_2 = F.cross_entropy(output_adv, target)
        if not self.ablation_exp:  #not running ablation test
            loss_3 = criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
            return loss_1+self.alpha * loss_2+self.beta * loss_3
        else:                      # running ablation experiments.
            return loss_1+loss_2

    def train(self, optimizer, epoch):
        '''this is training with sac'''
        val_loader = self.random_val_loader1(epoch)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        strategy, reward_list = self.rl_strategy(self.model, val_loader, self.step) ## call sac, ddpg, ppo and td3
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # strategy = sac(model, data, target)
            self.model.train()
            optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            output_adv = self.model(self.transform_MR6(data, strategy))
            loss = self.__get_loss(output, target, output_adv, criterion_kl)
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {}\tLoss: {:.6f}'.format( epoch, loss.item()))

def get_model(dataset_idx, model_idx):
    dataset_idx_get_model_dict = {
        'svhn':get_svhn_model,
        'cifar10': get_cifar10_model,
        'fashion_mnist': get_fashionmnist_model,
        'gtsrb':get_gtsrb_model
    }
    get_func = dataset_idx_get_model_dict[dataset_idx]
    model, args.lr = get_func(model_idx)
    model = model.to(device)
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
        dataset = first_dataset_idx_dict.get(dataset_idx)
    else:
        dataset = second_dataset_idx_dict[dataset_idx]
    dataset.set_batch_size(args.batch_size)
    return dataset

def TrainStrategyFactory(strategy, dataset_idx, model_idx, retry):
    dataset = get_dataset(dataset_idx, model_idx)
    model = get_model(dataset_idx, model_idx)
    if strategy == 'SAC':
        Operator = RL_TrainTest(dataset, model, model_idx, retry)
        Operator.set_rl_strategy(strategy='sac')
    if strategy == 'TD3':
        Operator = RL_TrainTest(dataset, model, model_idx, retry)
        Operator.set_rl_strategy(strategy='td3')
    if strategy == 'DDPG':
        Operator = RL_TrainTest(dataset, model, model_idx, retry)
        Operator.set_rl_strategy(strategy='ddpg')
    if strategy == 'PPO':
        Operator = RL_TrainTest(dataset, model, model_idx, retry)
        Operator.set_rl_strategy(strategy='ppo')
    Operator.dataset_idx= dataset.idx
    Operator.reset_loaders()
    Operator.set_save_path(args.logs_dir, args.model_dir)
    Operator.set_optimizer_parameters(args.lr, args.momentum)
    Operator.set_training_epochs(args.retrain_epochs, args.log_interval)
    Operator.set_device(device)
    return Operator

def effective_exp(total_retry_num=5, evalute_immediately= True):
    for dataset_idx in ['svhn', 'cifar10','gtsrb','fashion_mnist']:
        for model_idx in [1, 2]:
            for train_strategy in ['SAC']:
                for retry in range(1, total_retry_num+1):
                    operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                    operator.run_training()
                    if evalute_immediately:
                        operator.eval_test_adv_worstk()




if __name__ == '__main__':
    args.retrain_epochs = 100        #should be set to 100
    total_retry_num=5             #should be set to 5
    evalute_immediately = True         #should be set to True
    effective_exp(total_retry_num, evalute_immediately)