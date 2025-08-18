import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)
        #self.resnet18 = self.resnet18.to(device)

    def forward(self, x):
        return self.resnet18(x)