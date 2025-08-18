from RLStrategies.env_epoch_ll import BasicEnv
import random
import torchvision.transforms.functional as TF
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.6, type=float)    # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
args = parser.parse_args()

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=64):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

def transform_MR6(image, a):
    image = TF.affine(image, angle=a[0], translate=[a[1], a[2]],
                      scale=a[3], shear=[a[4], a[5]])
    return image

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
def td3(model,data_loader,step):
    data = pd.DataFrame(columns=['step', 'reward'])
    device = torch.device("cuda")
    strategy, loss = worstof10(model, data_loader, device)
    m = 1000
    while(loss * m<10):
        m *=10
    while (loss * m > 100):
        m /= 10
    print(loss * m)
    env = BasicEnv(data_loader, loss,m)
    loss_strategy = {}
    loss_strategy[loss] = strategy
    max_action = 1.0
    Max_train_steps = step
    max_e_steps = step/10
    update_every = 100

    state_dim = 7
    action_dim = 6

    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    policy_noise = args.policy_noise * max_action
    noise_clip = args.noise_clip * max_action
    policy_freq = args.policy_freq
    policy = TD3(state_dim, action_dim, max_action, args.discount, args.tau, policy_noise, noise_clip, policy_freq)
    replay_buffer = ReplayBuffer(state_dim, action_dim)

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
            if total_steps < (5 * max_e_steps): #
                act = np.array([np.random.uniform(-1, 1) for _ in range(6)])
                # print('explore',act)
            else:  #after agent learn change
                act = (
                        policy.select_action(np.array(s))
                        + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                ).clip(-max_action, max_action)
                # print('replay',act)
            # print('explore',act)

            # Perform action
            s_next, r, done, train_loss_rob, strategy = env.step(act, model)
            # Store data in replay buffer
            replay_buffer.add(s, act, s_next, r, done)
            s = s_next

            total_steps += 1
            episode_step += 1
            episode_reward += r
            '''train if it's time'''
            if (total_steps >= 2 * max_e_steps):
                policy.train(replay_buffer, args.batch_size)

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
                new_data = {'episode': [episode], 'reward': [episode_reward]}
                data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
                episode += 1
                print_running_episodes += 1
                print_running_reward += episode_reward
                break
            if done:
                new_data = {'episode': [episode], 'reward': [episode_reward]}
                data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
                episode += 1
                print_running_episodes += 1
                print_running_reward += episode_reward
                loss_strategy[train_loss_rob] = strategy
                # print('done', strategy)
    if len(loss_strategy) == 1:
        strategy = next(iter(loss_strategy.values()))
        return strategy,data
    else:
        max_target = max(loss_strategy.keys())
        max_strategy = loss_strategy[max_target]
        print(max_target,'max strategy', max_strategy)
        return max_strategy,data




