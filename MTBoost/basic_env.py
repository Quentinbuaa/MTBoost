# env_basic.py
import numpy as np
import torch
import torch.nn.functional as F
from transformations import TransformationParams


class BasicEnv:
    """
    Thin environment wrapper around a model + dataloader.
    State: [6 transform params, scaled_loss]
    Action: 6D in [-1,1] mapped via TransformationParams.from_action
    Reward: (current_loss - baseline_loss) * m
    Done: current_loss > baseline_loss
    """
    def __init__(self, data_loader, baseline_loss: float, m: float, device: str = "cuda"):
        self.data_loader = data_loader
        self.device = torch.device(device)
        self.flag = baseline_loss # baseline
        self.m = m
        self.strategy = None
        self.state = None
        self.reward = 0.0
        self.done = False


    def reset(self, strategy_list):
        self.strategy = TransformationParams(strategy_list)
        self.state = np.array(self.strategy.to_list() + [round(float(self.flag * self.m), 2)], dtype=np.float32)
        self.reward = 0.0
        self.done = False
        return self.state


    def step(self, action_6d, model):
        # Map action → transformation & simulate
        self.strategy = TransformationParams.from_action(action_6d)
        current_loss = self._simulate(model)
        info = current_loss # keep original signature
        return self.state, self.reward, self.done, info, self.strategy.to_list()


    # ----- internals ----- #
    def _simulate(self, model):
        model.eval()
        loss_val = self._eval_train_adv_loss(model)
        self.state = np.array(self.strategy.to_list() + [round(float(loss_val.item()) * self.m, 2)], dtype=np.float32)
        self.reward = round((loss_val.item() - self.flag) * self.m, 2)
        self.done = bool(loss_val > self.flag)
        return loss_val


    def _eval_train_adv_loss(self, model):
        total = 0.0
        with torch.no_grad():
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output_adv = model(self.strategy.apply(data))
                total += F.cross_entropy(output_adv, target, reduction="sum").item()
                total /= len(self.data_loader.sampler)
        return torch.tensor(total, device=self.device)