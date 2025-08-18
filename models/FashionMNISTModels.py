import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def vgg_for_fashionmnist():
    model = models.vgg11(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)
    return model

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入通道改为1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算全连接层的输入特征数
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 根据卷积层输出大小计算
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出大小: (batch_size, 32, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # 输出大小: (batch_size, 64, 8, 8)
        x = self.pool(F.relu(self.conv3(x)))  # 输出大小: (batch_size, 128, 4, 4)
        x = x.view(-1, 128 * 4 * 4)  # Flatten: (batch_size, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_fashionmnist_model(model_idx):
    fashionmnist_model_dict = {
        1: FashionMNISTCNN(),
        2: vgg_for_fashionmnist()
    }
    fashionmnist_lr_dict = {
        1: 0.01,
        2: 0.005
    }
    lr=fashionmnist_lr_dict[model_idx]
    return fashionmnist_model_dict[model_idx], lr