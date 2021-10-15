import torch
import torch.nn as nn

class EUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cfg.conv1.filter, kernel_size=cfg.conv1.kernel_size, 
            stride=cfg.conv1.stride, padding=cfg.conv1.padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=cfg.maxpool1.kernel_size)
        self.conv2 = nn.Conv2d(cfg.conv1.filter, cfg.conv2.filter, kernel_size=cfg.conv2.kernel_size, 
            stride=cfg.conv2.stride, padding=cfg.conv2.padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=cfg.maxpool2.kernel_size)
        self.conv3 = nn.Conv2d(cfg.conv2.filter, cfg.conv3.filter, kernel_size=cfg.conv3.kernel_size,
            stride=cfg.conv3.stride, padding=cfg.conv3.padding)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=cfg.dropout1)
        self.fc1 = nn.Linear(cfg.fc1.in_channel, cfg.fc1.out_channel)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=cfg.dropout2)
        self.fc2 = nn.Linear(cfg.fc2.in_channel, cfg.fc2.out_channel)
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = self.dropout1(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout2(x)
        out = self.fc2(x)
        return out
