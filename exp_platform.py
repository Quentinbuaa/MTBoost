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
    if strategy == 'AUTOAUG':  # Only works for ['svhn', 'cifar10'] datasets
        dataset.set_augment_mode('auto_augment')
        Operator = STD_TrainTest(dataset, model, model_idx, retry)
        Operator.train_strategy = 'autoaugment'
    if strategy == 'AUGMIX': # Only works for ['svhn', 'cifar10'] datasets
        dataset.set_augment_mode('augmix')
        Operator = STD_TrainTest(dataset, model, model_idx, retry)
        Operator.train_strategy = 'Augmix'
    if strategy == 'SENSEI': # Only works for ['svhn', 'cifar10'] datasets
        Operator = SENSEI_TrainTest(dataset, model, model_idx, retry)
        Operator.train_strategy = 'sensei'
    Operator.dataset_idx= dataset.idx
    Operator.reset_loaders()
    Operator.set_save_path(args.logs_dir, args.model_dir)
    Operator.set_optimizer_parameters(args.lr, args.momentum)
    Operator.set_training_epochs(args.retrain_epochs, args.log_interval)
    Operator.set_device(device)
    return Operator


# RQ1.1 Experiments
def effective_exp(total_retry_num=5, evalute_immediately= True):

    for dataset_idx in ['svhn', 'cifar10','gtsrb','fashion_mnist']:
        for model_idx in [1, 2]:
            for train_strategy in ['STD','RDM', 'W10', 'W10KL','SAC']:
                for retry in range(1, total_retry_num+1):
                    operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                    operator.run_training()
                    if evalute_immediately:
                        operator.eval_test_adv_worstk()

# RQ1.2 Experiments
def std_acc_effective_exp(total_retry_num=5, evalute_immediately= True):
    for dataset_idx in ['svhn', 'cifar10']:
        for model_idx in [1, 2]:
            for train_strategy in ['AUTOAUG','AUGMIX' ]:
                for retry in range(1, total_retry_num+1):
                    operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                    operator.run_training()
                    if evalute_immediately:
                        operator.eval_test_adv_worstk()

# RQ1.3 Experiments
def sensei_effective_exp(total_retry_num = 1, evalute_immediately= True):
    config_dict = {
        'svhn':1,
        'cifar10':2,
    }
    for dataset_idx in ['svhn', 'cifar10']:#['svhn', 'cifar10']:
        model_idx = config_dict[dataset_idx]
        for train_strategy in ['SENSEI']:
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.set_single_dataloader()
                operator.run_training()
                if evalute_immediately:
                    operator.eval_test_adv_worstk()

# RQ2.1 Experiments
def rl_horizontal_acc_effective_exp(total_retry_num=5, evalute_immediately= True):
    config_dict = {
        'svhn':1,
        'cifar10':2,
        'gtsrb':2,
        'fashion_mnist':1
    }
    for dataset_idx in config_dict:
        model_idx = config_dict[dataset_idx]
        for train_strategy in ['TD3','PPO','DDPG']:# SAC has been done in RQ1.1
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.run_training()
                if evalute_immediately:
                    operator.eval_test_adv_worstk()

#RQ2.2 Experiments
def rl_horizontal_ablation_exp(total_retry_num=5, evalute_immediately= True):
    config_dict = {
        'svhn':1,
        'cifar10':2,
        'gtsrb':2,
        'fashion_mnist':1
    }
    for dataset_idx in config_dict:
        model_idx = config_dict[dataset_idx]
        for train_strategy in ['SAC', 'TD3','PPO','DDPG']:
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.train_strategy+='_ablation'
                operator.set_ablation_mode()        # set the ablation experiment. The goal is to conduct test.
                operator.run_training()
                if evalute_immediately:
                    operator.eval_test_adv_worstk()

# RQ3.1 Experiments
def test_size_exp(total_retry_num=5, evalute_immediately= True):
    config_dict = {
        'svhn':1,
        'cifar10':2,
        'gtsrb':2,
        'fashion_mnist':1
    }
    train_strategy = 'SAC'
    for dataset_idx in config_dict:
        model_idx = config_dict[dataset_idx]
        for test_size in [5e-3, 1.5e-2]:
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.train_strategy+=f'_sub_{test_size}'
                operator.set_test_size(test_size)
                operator.run_training()
                if evalute_immediately:
                    operator.eval_test_adv_worstk()

# RQ3.2 Experiments
def alpha_beta_exp(total_retry_num=5, evalute_immediately= True):
    config_dict = {
        'svhn':1,
        'cifar10':2,
        'gtsrb':1,
        'fashion_mnist':1
    }
    train_strategy = 'SAC'
    for dataset_idx in ['gtsrb','fashion_mnist']:#config_dict:
        model_idx = config_dict[dataset_idx]
        for (alpha, beta) in [(2.0, 1.0), (4.0, 1.0), (1.0, 2.0)]:
            for retry in range(1, total_retry_num+1):
                operator = TrainStrategyFactory(train_strategy,dataset_idx, model_idx, retry)
                operator.train_strategy+=f'_alpha_{alpha}_beta_{beta}'
                operator.set_alpha_beta(alpha, beta)
                operator.run_training()
                if evalute_immediately:
                    operator.eval_test_adv_worstk()


if __name__ == '__main__':
    args.retrain_epochs = 1        #should be set to 100
    total_retry_num=1             #should be set to 5
    evalute_immediately =False         #should be set to True
    effective_exp(total_retry_num, evalute_immediately)
    std_acc_effective_exp(total_retry_num, evalute_immediately)
    sensei_effective_exp(total_retry_num, evalute_immediately)
    rl_horizontal_acc_effective_exp(total_retry_num, evalute_immediately)
    rl_horizontal_ablation_exp(total_retry_num, evalute_immediately)
    test_size_exp(total_retry_num, evalute_immediately)
    alpha_beta_exp(total_retry_num, evalute_immediately)
