# mtboost.py
import os
import time
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split

# ---- Import your own datasets/models ----
from models.SVHNModels import get_svhn_model
from models.CIFAR10Models import get_cifar10_model
from models.GTSRBModels import get_gtsrb_model
from models.FashionMNISTModels import get_fashionmnist_model
from MTBoost.datasets import SVHNDataset, CIFAR10Dataset, GTSRBDataset, FashionMNISTDataset, ThreeChannelledFashionMNISTDataset

# ---- Ours ----
from MTBoost.transformations import TransformationParams
from MTBoost.sac_agent import sac


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--retrain-epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--model-dir', default='./SavedCheckpoints')
parser.add_argument('--logs-dir', default='./Logs')
parser.add_argument('--save-freq', '-s', default=5, type=int)
parser.add_argument('--repeat-time', '-r', default=5, type=int)
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'


# ---------- Factories ---------- #
def get_model(dataset_idx, model_idx):
    factory = {
        'svhn': get_svhn_model,
        'cifar10': get_cifar10_model,
        'fashion_mnist': get_fashionmnist_model,
        'gtsrb': get_gtsrb_model,
    }
    model, args.lr = factory[dataset_idx](model_idx)
    return model.to(DEVICE)


def get_dataset(dataset_idx, model_idx=1):
    primary = {
        'svhn': SVHNDataset(),
        'cifar10': CIFAR10Dataset(),
        'fashion_mnist': FashionMNISTDataset(),
        'gtsrb': GTSRBDataset(),
    }
    alt = {
        'svhn': SVHNDataset(),
        'cifar10': CIFAR10Dataset(),
        'fashion_mnist': ThreeChannelledFashionMNISTDataset(),
        'gtsrb': GTSRBDataset(),
    }
    dataset = primary[dataset_idx] if model_idx == 1 else alt[dataset_idx]
    dataset.set_batch_size(args.batch_size)
    return dataset


# ---------- Training Operators ---------- #
class RL_TrainTest:
    def __init__(self, dataset, model, model_idx=1, retry=1):
        self.dataset = dataset
        self.dataset_idx = dataset.idx
        self.model = model
        self.model_idx = model_idx
        self.retry = retry
        self.train_strategy = 'sac'
        self.test_size = 0.01
        self.alpha = 1.0
        self.beta = 1.0
        self.step = 400
        self.ablation_exp = False

    def reset_loaders(self):
        self.trainset, self.train_loader, self.testset, self.test_loader = self.dataset.get_loaders()

    def set_save_path(self, logs_dir, model_saving_dir):
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(model_saving_dir, exist_ok=True)
        self.logging_file = os.path.join(logs_dir, f'{self.dataset_idx}_model.log')
        self.checkpoint_file = os.path.join(
            model_saving_dir,
            f'{self.dataset_idx}_model_{self.model_idx}_{self.train_strategy.upper()}_repeat_{self.retry}.pt'
        )

    def run_training(self):
        import torch.optim as optim
        optimizer = optim.SGD(
            self.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4
        )
        for epoch in range(1, args.retrain_epochs + 1):
            self.train(optimizer, epoch)
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def train(self, optimizer, epoch):
        # 1) pick a small validation slice for SAC loop
        labels = self.dataset.get_labels()
        train_idx, val_idx = train_test_split(range(len(labels)), test_size=self.test_size, random_state=epoch)
        val_sampler = SubsetRandomSampler(val_idx)
        val_loader = DataLoader(self.trainset, batch_size=self.dataset.batch_size, sampler=val_sampler)
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        # 2) run SAC to get adversarial strategy
        strategy, _ = sac(self.model, val_loader, self.step, device=DEVICE)

        # 3) standard training step with this strategy
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            optimizer.zero_grad()
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = self.model(data)
            out_adv = self.model(TransformationParams(strategy).apply(data))
            loss1 = F.cross_entropy(out, target)
            loss2 = F.cross_entropy(out_adv, target)
            loss3 = criterion_kl(
                F.log_softmax(out_adv, dim=1),
                F.softmax(out, dim=1).clamp(min=1e-10)
            )
            loss = loss1 + self.alpha * loss2 + self.beta * loss3
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}] '
                      f'Loss: {loss.item():.6f}')


# ---------- Factory & Runner ---------- #
def TrainStrategyFactory(strategy, dataset_idx, model_idx, retry):
    dataset = get_dataset(dataset_idx, model_idx)
    model = get_model(dataset_idx, model_idx)
    operator = RL_TrainTest(dataset, model, model_idx, retry)
    operator.reset_loaders()
    operator.set_save_path(args.logs_dir, args.model_dir)
    return operator


def effective_exp(total_retry_num=5, evalute_immediately=True):
    for dataset_idx in ['svhn', 'cifar10', 'gtsrb', 'fashion_mnist']:
        for model_idx in [1, 2]:
            for retry in range(1, total_retry_num + 1):
                operator = TrainStrategyFactory('SAC', dataset_idx, model_idx, retry)
                operator.run_training()
                # Optionally add evaluation


if __name__ == '__main__':
    effective_exp(total_retry_num=args.repeat_time, evalute_immediately=True)
