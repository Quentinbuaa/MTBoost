import torch.nn as nn
from models.wideresnet import WideResNet
import torchvision.models as models

def CIFAR10Net():
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.Linear(512,10)
    )
    return model

def get_cifar10_model(model_idx):
    if model_idx == 1:
        lr = 0.005
        model = CIFAR10Net()
    if model_idx == 2:
        lr = 0.05
        model = WideResNet(40, 10, 2, 0.3)
    return model, lr
