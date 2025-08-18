import torch.nn as nn
from models.wideresnet import WideResNet
import torchvision.models as models

def vgg_for_gtsrb():
    model = models.vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 43)
    return model

def get_gtsrb_model(model_idx):
    gtsrb_mode_dict={
        1: WideResNet(40, 43, 2, 0.3),
        2: vgg_for_gtsrb()
    }
    lr=0.01
    return gtsrb_mode_dict[model_idx], lr
