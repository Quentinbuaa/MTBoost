import copy
from RLStrategies.env_epoch_ll import BasicEnv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
import torchvision.transforms.functional as TF
import pandas as pd
def transform_MR6(image, a):
    image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                      scale=a[3], shear=[a[4], a[5]])
    return image
import argparse
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

parser = argparse.ArgumentParser()
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
# optional parameters

parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--update_iteration', default=20, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
min_Val = torch.tensor(1e-7).float().to(device) # min value
state_dim = 7
action_dim = 6
max_action = 1.0

class Replay_buffer():
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


def eval_train_adv_loss(model, device, train_loader,strategy):
    model.eval()
    train_loss_rob = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output_adv = model(transform_MR6(data, strategy))
            train_loss_rob += F.cross_entropy(output_adv, target, size_average=False).item()
    train_loss_rob = train_loss_rob / len(train_loader.sampler)
    return train_loss_rob
def worstof10(model,train_loader,device):
    k = 10
    rob_list = []
    a10 = []
    for i in range(k):
        a0 = round(random.uniform(-30, 30),2)
        a1 = round(random.uniform(-3, 3),2)
        a2 = round(random.uniform(-3, 3),2)
        a3 = round(random.uniform(0.8, 1.2),2)
        a4 = round(random.uniform(-10, 10),2)
        a5 = round(random.uniform(-10, 10),2)
        a = [a0, a1, a2, a3, a4, a5]
        a10.append(a)
        training_rob_loss = eval_train_adv_loss(model, device,train_loader,a10[i])
        rob_list.append(training_rob_loss)
    loss = max(rob_list)
    strategy = a10[rob_list.index(loss)]
    print('worst',loss,strategy)
    return strategy,loss
#every epoch
def ddpg(model,data_loader,step):
    data = pd.DataFrame(columns=['step', 'reward'])
    device = torch.device("cuda")
    strategy, loss = worstof10(model, data_loader, device)
    m = 1000
    while(loss * m<10):
        m *=10
    while (loss * m > 100):
        m /= 10
    env = BasicEnv(data_loader, loss,m)
    loss_strategy = {}
    loss_strategy[loss] = strategy

    Max_train_steps = step
    max_e_steps = step / 10

    agent = DDPG(state_dim, action_dim, max_action)
    total_steps = 0
    episode = 1

    print_running_reward = 0
    print_running_episodes = 0

    while total_steps < Max_train_steps:
        if total_steps == 0:
            s_saved = env.reset(strategy)
        s = s_saved
        episode_step = 0
        episode_reward = 0
        done = False
        '''Interact & trian'''
        while not done:
            act = agent.select_action(s)
            act= (act + np.random.normal(0, args.exploration_noise, size=action_dim)).clip(
                -1, 1)
            # print(total_steps,act)
            s_next, r, done, train_loss_rob, strategy = env.step(act,model)  # dw: dead&win; tr: truncated
            #agent.replay_buffer.push((s, s_next, act, r, np.float(done))) # np.float is not supported after np.1.24. Thus, replace np.float with just float
            agent.replay_buffer.push((s, s_next, act, r, float(done)))

            if (total_steps >= 2 * max_e_steps):
                agent.update()
            s = s_next
            total_steps += 1
            episode_step += 1
            episode_reward += r
            # printing average reward
            if total_steps % 80 == 0 and print_running_episodes != 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(episode, total_steps,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if episode_step > max_e_steps:
                episode += 1
                print_running_episodes += 1
                print_running_reward += episode_reward
                break
            if done:
                # new_data = {'episode': [episode], 'reward': [episode_reward]}
                # data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
                episode += 1
                print_running_episodes += 1
                print_running_reward += episode_reward
                loss_strategy[train_loss_rob] = strategy

    if len(loss_strategy) == 1:
        strategy = next(iter(loss_strategy.values()))
        return strategy,data
    else:
        max_target = max(loss_strategy.keys())
        max_strategy = loss_strategy[max_target]
        print('max strategy', max_strategy)
        return max_strategy,data




