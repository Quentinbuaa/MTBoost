import torch.nn.functional as F
import torch
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn as nn
def transform1(image, a):
    image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                      scale=a[3], shear=[a[4], a[5]]
                     )
    return image
class BasicEnv():#use val to test_pgd max losss
    def __init__(self,data_loader,loss,m):
        self.state = None
        self.data_loader = data_loader
        self.device = torch.device("cuda")
        self.flag = loss
        self.m = m
    def reset(self,strategy):
        self.state = np.array([strategy[0],strategy[1],strategy[2],strategy[3],strategy[4],strategy[5],round(float(self.flag * self.m), 2)], dtype=np.float32)
        return self.state

    def step(self, action,model):
        self.a0 = round(30 * action[0], 2)
        self.a1 = round(3 * action[1], 2)
        self.a2 = round(3 * action[2], 2)
        self.a3 = round(0.2 * action[3] + 1, 2)
        self.a4 = round(10 * action[4], 2)
        self.a5 = round(10 * action[5], 2)
        if self.a0 < -30: self.a0 = -30
        if self.a0 > 30: self.a0 = 30
        if self.a1 < -3: self.a1 = -3
        if self.a1 > 3: self.a1 = 3
        if self.a2 < -3: self.a2 = -3
        if self.a2 > 3: self.a2 = 3
        if self.a3 < 0.8: self.a3 = 0.8
        if self.a3 > 1.2: self.a3 = 1.2
        if self.a4 < -10: self.a4 = -10
        if self.a4 > 10: self.a4 = 10
        if self.a5 < -10: self.a5 = -10
        if self.a5 > 10: self.a5 = 10
        self.strategy = [self.a0, self.a1, self.a2, self.a3, self.a4, self.a5]
        train_loss_rob = self.mysimulate(model)
        info = train_loss_rob
        return self.state, self.reward, self.done, info, self.strategy

    def mysimulate(self,model):
        model.eval()
        train_loss_rob = self.eval_train_adv_loss(model)
        # self.state = np.array([self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, round(float(train_loss_rob.item() - self.flag) * 1000, 2)], dtype=np.float32)
        self.state = np.array([self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, round(float(train_loss_rob.item()) * self.m, 2)], dtype=np.float32)

        # print(train_loss_rob,max(self.loss_list),self.loss_list)
        # self.reward = round(float(train_loss_rob.item() - self.flag), 2)
        self.reward = round((train_loss_rob.item() - self.flag)* self.m, 2)
        # print('reward', self.reward)
        self.done = True if train_loss_rob > self.flag else False
        # self.done = True if self.reward > 10 else False
        # print('reward', self.reward, 'state', self.state)
        return train_loss_rob
    #
    def eval_train_adv_loss(self,model):
        model.eval()
        train_loss_rob = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                # output = model(data)
                output_adv = model(transform1(data, self.strategy))
                train_loss_rob +=F.cross_entropy(output_adv, target, size_average=False)
                # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
                # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(target, dim=1))
                # target_onehot = F.one_hot(target, num_classes=output_adv.size(1)).float()
                # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1),  F.softmax(target_onehot, dim=1))
        train_loss_rob = train_loss_rob / len(self.data_loader.sampler)
        # print('train loss (Robust): {:.2f}'.format(train_loss_rob))
        return train_loss_rob
