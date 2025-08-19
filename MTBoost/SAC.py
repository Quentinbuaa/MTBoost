import copy
from MTBoost.ENV import BasicEnv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import torchvision.transforms.functional as TF
import pandas as pd
def build_net(layer_shape, hidden_activation, output_activation):
    '''Build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape)

        self.a_net = build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic, with_logprob):
        '''Network with Enforcing Action Bounds'''
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  # 总感觉这里clamp不利于学习
        # we learn log_std rather than std, so that exp(log_std) is always > 0
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()

        '''↓↓↓ Enforcing Action Bounds, see Page 16 of https://arxiv.org/pdf/1812.05905.pdf ↓↓↓'''
        a = torch.tanh(u)
        if with_logprob:
            # Get probability density of logp_pi_a from probability density of u:
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Critic, self).__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


# reward engineering for better training


def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return a * max_action


def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action


def evaluate_policy(env, agent, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test_pgd time
            a = agent.select_action(s, deterministic=True)
            s_next, r, dw, tr, info = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores / turns)


def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
class SAC_countinuous():
    state_dim = 7
    action_dim = 6
    # net_width = 64
    net_width = 256
    dvc = 'cuda'
    a_lr = 3e-4
    c_lr = 3e-4
    adaptive_alpha = False
    batch_size = 64
    gamma = 0.99
    alpha = 0.2
    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.actor = Actor(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, (self.net_width, self.net_width)).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(1e6), dvc=self.dvc)

        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-self.action_dim, dtype=float, requires_grad=True, device=self.dvc)
            # We learn log_alpha instead of alpha to ensure alpha>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.c_lr)

    def select_action(self, state, deterministic):
        # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)
            a, _ = self.actor(state, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def train(self, ):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)

        # ----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        with torch.no_grad():
            a_next, log_pi_a_next = self.actor(s_next, deterministic=False, with_logprob=True)
            target_Q1, target_Q2 = self.q_critic_target(s_next, a_next)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (~dw) * self.gamma * (
                        target_Q - self.alpha * log_pi_a_next)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, a)

        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        # Freeze critic so you don't waste computational effort computing gradients for them when update actor
        for params in self.q_critic.parameters(): params.requires_grad = False

        a, log_pi_a = self.actor(s, deterministic=False, with_logprob=True)
        current_Q1, current_Q2 = self.q_critic(s, a)
        Q = torch.min(current_Q1, current_Q2)

        a_loss = (self.alpha * log_pi_a - Q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters(): params.requires_grad = True

        # ----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure alpha>0
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # ----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, EnvName, timestep):
        torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep)))
        self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep)))


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        # 每次只放入一个时刻的数据
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.from_numpy(a).to(self.dvc)  # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size  # 存满了又重头开始存
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]

def transform1(image, a):
    image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                      scale=a[3], shear=[a[4], a[5]])
    return image
def eval_train_adv_loss(model, device, train_loader,strategy):
    model.eval()
    train_loss_rob = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # output = model(data)
            output_adv = model(transform1(data, strategy))
            # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
            # target_onehot = F.one_hot(target, num_classes=output_adv.size(1)).float()
            # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1),  F.softmax(target_onehot, dim=1))
            train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
    train_loss_rob = train_loss_rob / len(train_loader.sampler)
    # print("attack data set", len(train_loader.sampler))
    return train_loss_rob
def worstof10(model,train_loader,device):
    k = 10
    rob_list = []
    a10 = []
    for i in range(k):
        a0 = round(random.uniform(-30, 30),2)
        a1 = round(random.uniform(-3, 3),2)
        a2 = round(random.uniform(-3, 3),2)
        a3 = round(random.uniform(0.8, 1.2), 2)
        a4 = round(random.uniform(-10, 10), 2)
        a5 = round(random.uniform(-10, 10), 2)
        a = [a0, a1, a2, a3, a4, a5]
        a10.append(a)
        training_rob_loss = eval_train_adv_loss(model, device,train_loader,a10[i])
        rob_list.append(training_rob_loss)
    loss = max(rob_list)
    strategy = a10[rob_list.index(loss)]
    print('worst',loss,strategy)
    return strategy,loss
#every epoch
def sac(model,data_loader,step):
    data = pd.DataFrame(columns=['step', 'reward'])
    device = torch.device("cuda")
    strategy, loss = worstof10(model, data_loader, device)
    # m = round(5/loss,2)
    #m = 10
    # while (loss * m < 1):
    #     m *= 2
    # while(loss * m >50):
    #     m /=2
    m = 1000
    while (loss * m < 10):
        m *= 10
    while (loss * m > 100):
        m /= 10
    print('m', m)
    env = BasicEnv(data_loader, loss,m)
    loss_strategy = {}
    loss_strategy[loss] = strategy
    max_action = [-1.0,1.0]
    max_e_steps = step/10
    Max_train_steps = step
    # max_e_steps = 40
    # Max_train_steps = 400
    # max_e_steps = 20
    # Max_train_steps = 200
    update_every = 10
    agent = SAC_countinuous()
    total_steps = 0
    episode = 1
    while total_steps < Max_train_steps:
        if total_steps == 0:
            s_saved = env.reset(strategy)
        s = s_saved
        episode_step = 0
        episode_reward = 0
        done = False
        '''Interact & trian'''
        while not done:
            if total_steps < (5 * max_e_steps): #
                # act = env.action_space.sample()  # act∈[-max,max]
                act = np.array([round(np.random.uniform(-1, 1),2) for _ in range(6)])
                # print('explore',act)
            else:  #after agent learn change
                act = agent.select_action(s, deterministic=False)  # a∈[-1,1]
                act = np.round(act, 2)
                # print('replay',act)
            # act = np.array([random.uniform(-1, 1) for _ in range(6)])
            # print('explore',act)

            s_next, r, done, train_loss_rob, strategy = env.step(act,model)  # dw: dead&win; tr: truncated
            agent.replay_buffer.add(s, act, r, s_next, done)
            s = s_next
            total_steps += 1
            episode_step += 1
            episode_reward += r
            '''train if it's time'''
            # train 50 times every 50 steps rather than 1 training per step. Better!
            if (total_steps >= 2 * max_e_steps) and (total_steps % update_every == 0):
                # print('train agent')
                for j in range(update_every):
                    agent.train()
            # print('episode',episode,'episode step',episode_step)
            # new_data = {'step': [total_steps], 'reward': [r]}
            # data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
            if episode_step > max_e_steps:
                new_data = {'episode': [episode], 'reward': [episode_reward]}
                data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
                episode += 1
                break
            if done:
                new_data = {'episode': [episode], 'reward': [episode_reward]}
                data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
                episode += 1
                loss_strategy[train_loss_rob] = strategy
                # print('done', strategy)
    if len(loss_strategy) == 1:
        strategy = next(iter(loss_strategy.values()))
        # return strategy,data,loss*m,loss*m
        return strategy,data

    else:
        max_target = max(loss_strategy.keys())
        max_strategy = loss_strategy[max_target]
        print('max loss',max_target)
        print('max strategy', max_strategy)
        # return max_strategy,data,loss*m,max_target.cpu().item()*m
        return max_strategy,data

def worstof10_batch(model,data,target,device):
    k = 10
    rob_list = []
    a10 = []
    for i in range(k):
        a0 = round(random.uniform(-30, 30),2)
        a1 = round(random.uniform(-3, 3),2)
        a2 = round(random.uniform(-3, 3),2)
        a3 = round(random.uniform(0.8, 1.2), 2)
        a4 = round(random.uniform(-10, 10), 2)
        a5 = round(random.uniform(-10, 10), 2)
        a = [a0, a1, a2, a3, a4, a5]
        a10.append(a)
        training_rob_loss = eval_train_adv_batch_loss(model, data,target,a)
        rob_list.append(training_rob_loss)
    loss = max(rob_list)
    strategy = a10[rob_list.index(loss)]
    print('worst',loss,strategy)
    return strategy,loss
def eval_train_adv_batch_loss(model, data, target,strategy):
    model.eval()
    train_loss_rob = 0
    with torch.no_grad():
            output_adv = model(transform1(data, strategy))
            # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output, dim=1))
            # target_onehot = F.one_hot(target, num_classes=output_adv.size(1)).float()
            # train_loss_rob += criterion_kl(F.log_softmax(output_adv, dim=1),  F.softmax(target_onehot, dim=1))
            train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
    train_loss_rob = train_loss_rob / len(data)
    return train_loss_rob
def test_agent(model,data,target,agent):
    device = torch.device("cuda")
    strategy, loss = worstof10_batch(model,data,target,device)
    m = 1000
    while (loss * m < 10):
        m *= 10
    while (loss * m > 100):
        m /= 10
    s = np.array([strategy[0],strategy[1],strategy[2],strategy[3],strategy[4],strategy[5],round(float(loss * m), 2)], dtype=np.float32)
    rob_list= [loss]
    a10 = [strategy]
    for i in range(10):
        act = agent.select_action(s, deterministic=False)  # a∈[-1,1]
        a0 = round(30 * act[0], 2)
        a1 = round(3 * act[1], 2)
        a2 = round(3 * act[2], 2)
        a3 = round(0.2 * act[3] + 1, 2)
        a4 = round(10 * act[4], 2)
        a5 = round(10 * act[5], 2)
        if a0 < -30: a0 = -30
        if a0 > 30: a0 = 30
        if a1 < -3: a1 = -3
        if a1 > 3: a1 = 3
        if a2 < -3: a2 = -3
        if a2 > 3: a2 = 3
        if a3 < 0.8: a3 = 0.8
        if a3 > 1.2: a3 = 1.2
        if a4 < -10: a4 = -10
        if a4 > 10: a4 = 10
        if a5 < -10: a5 = -10
        if a5 > 10: a5 = 10
        act = [a0,a1,a2,a3,a4,a5]
        strategy = np.round(act, 2)
        training_rob_loss = eval_train_adv_batch_loss(model,data,target, strategy)
        a10.append(act)
        rob_list.append(training_rob_loss)
    loss = max(rob_list)
    strategy = a10[rob_list.index(loss)]
    return strategy



