# sac_agent.py
import copy
import random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from transformations import TransformationParams
from MTBoost.basic_env import BasicEnv


# ---------- networks ---------- #
def _build_net(layer_shape, hidden_activation, output_activation):
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hidden_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hid_shape,
                 hidden_activation=nn.ReLU, output_activation=nn.ReLU):
        super().__init__()
        layers = [state_dim] + list(hid_shape)
        self.a_net = _build_net(layers, hidden_activation, output_activation)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, deterministic: bool, with_logprob: bool):
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out),
                              self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)

        logp_pi_a = None
        if with_logprob:
            # Tanh-squash correction (SAC paper Appendix)
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - \
                        (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        return a, logp_pi_a


class DoubleQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hid_shape):
        super().__init__()
        layers = [state_dim + action_dim] + list(hid_shape) + [1]
        self.Q1 = _build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = _build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.Q1(sa), self.Q2(sa)


# ---------- replay buffer ---------- #
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int, device: str):
        self.max_size = max_size
        self.device = device
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=device)
        self.a = torch.zeros((max_size, action_dim), dtype=torch.float, device=device)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=device)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=device)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=device)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.device)
        self.a[self.ptr] = torch.from_numpy(a).to(self.device)
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.device)
        self.dw[self.ptr] = dw
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, device=self.device, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


# ---------- SAC Agent ---------- #
class SACAgent:
    def __init__(self,
                 state_dim: int = 7,
                 action_dim: int = 6,
                 net_width: int = 256,
                 device: str = "cuda",
                 a_lr: float = 3e-4,
                 c_lr: float = 3e-4,
                 gamma: float = 0.99,
                 alpha: float = 0.2,
                 batch_size: int = 64,
                 adaptive_alpha: bool = False):
        self.device = torch.device(device)
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.adaptive_alpha = adaptive_alpha
        self.tau = 0.005

        self.actor = Actor(state_dim, action_dim, (net_width, net_width)).to(self.device)
        self.q = DoubleQCritic(state_dim, action_dim, (net_width, net_width)).to(self.device)
        self.q_tgt = copy.deepcopy(self.q)
        for p in self.q_tgt.parameters():
            p.requires_grad = False

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr=c_lr)
        self.rb = ReplayBuffer(state_dim, action_dim, int(1e6), self.device)

        if adaptive_alpha:
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float, device=self.device)
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float,
                                          requires_grad=True, device=self.device)
            self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=c_lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic=False) -> np.ndarray:
        s = torch.as_tensor(state[None, :], dtype=torch.float, device=self.device)
        a, _ = self.actor(s, deterministic, with_logprob=False)
        return a.cpu().numpy()[0]

    def update(self):
        s, a, r, s_next, dw = self.rb.sample(self.batch_size)

        with torch.no_grad():
            a_next, logp_next = self.actor(s_next, deterministic=False, with_logprob=True)
            tq1, tq2 = self.q_tgt(s_next, a_next)
            tQ = torch.min(tq1, tq2)
            tQ = r + (~dw) * self.gamma * (tQ - self.alpha * logp_next)

        q1, q2 = self.q(s, a)
        q_loss = F.mse_loss(q1, tQ) + F.mse_loss(q2, tQ)
        self.opt_q.zero_grad()
        q_loss.backward()
        self.opt_q.step()

        for p in self.q.parameters():
            p.requires_grad = False
        a_pi, logp = self.actor(s, deterministic=False, with_logprob=True)
        q1_pi, q2_pi = self.q(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        a_loss = (self.alpha * logp - q_pi).mean()
        self.opt_actor.zero_grad()
        a_loss.backward()
        self.opt_actor.step()
        for p in self.q.parameters():
            p.requires_grad = True

        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()
            self.alpha = self.log_alpha.exp()

        # Polyak averaging
        for p, pt in zip(self.q.parameters(), self.q_tgt.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)


# ---------- helpers ---------- #
def eval_train_adv_loss(model, device, loader, strategy: TransformationParams) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out_adv = model(strategy.apply(data))
            total += F.cross_entropy(out_adv, target, reduction="sum").item()
    total /= len(loader.sampler)
    return total


def worstof_k(model, loader, device: str = "cuda", k: int = 10) -> Tuple[List[float], float]:
    """Sample k random strategies and return the one with max loss."""
    best_loss = -1.0
    best_params = None
    for _ in range(k):
        strategy = TransformationParams.sample_random()
        loss = eval_train_adv_loss(model, torch.device(device), loader, strategy)
        if loss > best_loss:
            best_loss = loss
            best_params = strategy.to_list()
    return best_params, best_loss


# ---------- end-to-end SAC training routine ---------- #
def sac(model, data_loader, steps: int, device: str = "cuda"):
    import pandas as pd
    device_t = torch.device(device)

    # 1) Baseline and loss scaling m
    strategy0, loss0 = worstof_k(model, data_loader, device)
    m = 1000.0
    while loss0 * m < 10:
        m *= 10
    while loss0 * m > 100:
        m /= 10

    # 2) Env + Agent
    env = BasicEnv(data_loader, baseline_loss=loss0, m=m, device=device)
    agent = SACAgent(device=device)

    # 3) Loop
    max_e_steps = steps // 10
    Max_train_steps = steps
    update_every = 10

    total_steps = 0
    episode = 1
    df = pd.DataFrame(columns=["episode", "reward"])

    s = env.reset(strategy0)
    loss_strategy = {loss0: strategy0}

    while total_steps < Max_train_steps:
        episode_reward = 0.0
        episode_step = 0
        done = False
        while not done:
            if total_steps < (5 * max_e_steps):
                act = np.array([round(random.uniform(-1, 1), 2) for _ in range(6)], dtype=np.float32)
            else:
                act = agent.act(s, deterministic=False).astype(np.float32)
                act = np.round(act, 2)

            s_next, r, done, train_loss_rob, strategy = env.step(act, model)
            agent.rb.add(s, act, r, s_next, done)
            s = s_next
            total_steps += 1
            episode_step += 1
            episode_reward += r

            if (total_steps >= 2 * max_e_steps) and (total_steps % update_every == 0):
                for _ in range(update_every):
                    agent.update()

            if episode_step > max_e_steps or done:
                df = pd.concat([df, pd.DataFrame({"episode": [episode], "reward": [episode_reward]})],
                               ignore_index=True)
                episode += 1
                if done:
                    loss_strategy[train_loss_rob] = strategy
                if total_steps < Max_train_steps:
                    s = env.reset(strategy)
                break

    if len(loss_strategy) == 1:
        return list(loss_strategy.values())[0], df
    else:
        max_target = max(loss_strategy.keys())
        return loss_strategy[max_target], df


# ---------- test agent (optional) ---------- #
def test_agent(model, data, target, agent: SACAgent, device: str = "cuda"):
    device_t = torch.device(device)
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(data, target), batch_size=len(data))

    strategy0, loss0 = worstof_k(model, loader, device)
    m = 1000.0
    while loss0 * m < 10:
        m *= 10
    while loss0 * m > 100:
        m /= 10

    s = np.array(strategy0 + [round(float(loss0 * m), 2)], dtype=np.float32)
    candidates = [strategy0]
    losses = [loss0]

    for _ in range(10):
        act = agent.act(s, deterministic=False)
        strategy = TransformationParams.from_action(act).to_list()
        loss = eval_train_adv_loss(model, device_t, loader, TransformationParams(strategy))
        candidates.append(strategy)
        losses.append(loss)

    best_idx = int(np.argmax(losses))
    return candidates[best_idx]
