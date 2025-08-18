import numpy as np
import os, random, time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.nn as nn
from RLStrategies.SAC_epoch_ll import sac #find strategy every epoch
from RLStrategies.TD3_epoch_ll import td3
from RLStrategies.PPO_epoch_ll import ppo
from RLStrategies.DDPG_epoch_ll import ddpg

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from deap import base, creator, tools

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

class STD_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy='Standard'

    def train(self, optimizer, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % self.log_interval == 0:
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
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output_adv = self.model(self.transform_MR6(data, strategy))
            loss3 = F.cross_entropy(output_adv, target)
            loss = loss3# + loss1
            loss.backward()
            optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))

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
                data, target = data.to(self.device), target.to(self.device)
                output_adv = self.model(self.transform_MR6(data, strategy))
                train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
            # print('train loss (Robust): {:.2f}'.format(train_loss_rob))
        return train_loss_rob

class W10_KL_TrainTest(W_of_10_TrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'w10kl'
        self.k = 10

    def train(self, optimizer, epoch):
        criterion_kl = nn.KLDivLoss(reduction = 'batchmean')
        self.model.eval()
        strategy = self.worstof10_ll()
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            output_adv = self.model(self.transform_MR6(data, strategy))
            loss = F.cross_entropy(output, target) + criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1).clamp(min=1e-10))
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format( epoch, loss.item()))

class SENSEI_TrainTest(VirtualTrainTest):
    def __init__(self, dataset, model, model_idx=1, retry=1):
        super().__init__(dataset, model, model_idx, retry)
        self.train_strategy = 'sensei'
        # 定义每个参数的有效范围（最小值、最大值）Define the Range for each Parameter
        self.PARAM_RANGES = {
            0: (-30, 30),  # a0
            1: (-3, 3),  # a1
            2: (-3, 3),  # a2
            3: (0.8, 1.2),  # a3
            4: (-10, 10),  # a4
            5: (-10, 10)  # a5
        }

    def set_single_dataloader(self):
        self.single_train_loader = DataLoader(self.trainset, batch_size=1, shuffle=True)

    def train(self, optimizer, epoch):
        kl = False
        # 用于存储对抗样本和对应标签的列表
        adv_data = []
        adv_targets = []
        # print('1', train_loader1.dataset)
        for batch_idx, (data, target) in enumerate(self.single_train_loader):
            data, target = data.to(self.device), target.to(self.device)
            if F.cross_entropy(self.model(data), target) < 0.01:
                adv_data.append(data.cpu())
            else:
                strategy = self.select_cross_mutate(self.model, data, target, kl)
                data_adv = self.transform_MR6(data, strategy)
                # 将生成的对抗样本和目标添加到列表
                adv_data.append(data_adv.cpu())  # 转移到 CPU，如果需要
            adv_targets.append(target.cpu())  # 转移到 CPU，如果需要
        # 将列表转换为张量

        adv_data_tensor = torch.cat(adv_data, dim=0)
        adv_targets_tensor = torch.cat(adv_targets, dim=0)
        # 创建一个新的 TensorDataset
        adv_dataset = TensorDataset(adv_data_tensor, adv_targets_tensor)
        # 用新的对抗样本数据集创建 DataLoader
        train_loader_adv = DataLoader(adv_dataset, batch_size=self.dataset.batch_size, shuffle=True)
        for batch_idx, (data, target) in enumerate(train_loader_adv):
            data, target = data.to(self.device), target.to(self.device)
            self.model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(self.model(data), target)
            loss.backward()
            optimizer.step()
            # print progress
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} \tLoss: {:.6f}'.format(
                    epoch, loss.item()))

    def select_cross_mutate(self, model, data, target, kl):
        POPULATION_SIZE = 10  # 增加种群规模
        GENERATIONS = 5  # 增加代数
        MUTATION_RATE = 0.5  # 变异概率
        CROSSOVER_RATE = 0.5  # 交叉概率

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        toolbox.register("individual", tools.initIterate, creator.Individual, self.generate_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", self.custom_mutate)
        toolbox.register("select", tools.selTournament, tournsize=3)  # 控制锦标赛规模

        population = toolbox.population(n=POPULATION_SIZE)

        # 遗传算法主循环
        for generation in range(GENERATIONS):
            # 评估个体适应度
            fitnesses = list(map(lambda ind: toolbox.evaluate(ind, model, data, target, kl), population))
            # 继续处理 fitnesses 和选择下代个体
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
                # 选择下代个体
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            # 交叉和变异产生新的个体
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CROSSOVER_RATE:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < MUTATION_RATE:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    # 将后代和当前种群合并，替换掉适应度差的个体
            population[:] = offspring
        best_individual = tools.selBest(population, k=1)[0]
        best_individual[:] = [round(value, 2) for value in best_individual]
        return best_individual

    def custom_mutate(self, individual, mutation_probability=0.5, mutation_range=0.5):
        for i in range(len(individual)):
            if random.random() < mutation_probability:
                min_val, max_val = self.PARAM_RANGES[i]
                mutation_value = np.random.uniform(-mutation_range, mutation_range)
                new_value = individual[i] + mutation_value
                individual[i] = round(np.clip(new_value, min_val, max_val), 2)
        return individual,

    def generate_individual(self):
        a0 = round(random.uniform(-30, 30), 2)
        a1 = round(random.uniform(-3, 3), 2)
        a2 = round(random.uniform(-3, 3), 2)
        a3 = round(random.uniform(0.8, 1.2), 2)
        a4 = round(random.uniform(-10, 10), 2)
        a5 = round(random.uniform(-10, 10), 2)
        a = [a0, a1, a2, a3, a4, a5]
        return a

    # 定义计算适应度的函数
    def evaluate(self, individual, model, data, target, kl):
        model.eval()
        with torch.no_grad():
            data, target = data.to(self.device), target.to(self.device)
            output_adv = model(self.transform_MR6(data, individual))
            if kl == True:
                criterion_kl = nn.KLDivLoss(reduction='sum')
                loss = criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(model(data), dim=1))
            else:
                loss = F.cross_entropy(output_adv, target, size_average=False).item()
        return (loss,)

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

