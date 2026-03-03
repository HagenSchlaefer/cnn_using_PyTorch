# MyCnn3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Convs
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)                     # input channels=1, output channels=16, kernel size=3, padding=1 (28x28 -> 28x28)   
        self.bn1 = nn.BatchNorm2d(16)                                   # BatchNorm for 16 channels

        self.convStride1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # input channels=16, output channels=32, kernel size=3, stride=2, padding=1 (28x28 -> 14x14)
        self.bn2 = nn.BatchNorm2d(32)                                   # BatchNorm for 32 channels

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)                    # input channels=32, output channels=64, kernel size=3, padding=1 (14x14 -> 14x14)  
        self.bn3 = nn.BatchNorm2d(64)                                   # BatchNorm for 64 channels

        self.convStride2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # input channels=64, output channels=128, kernel size=3, stride=2, padding=1 (14x14 -> 7x7)
        self.bn4 = nn.BatchNorm2d(128)                                  # BatchNorm for 128 channels

        # Fully Connected
        self.fc1 = nn.Linear(128 * 7 * 7, 512)                          # 6272 = 128*7*7
        self.fc2 = nn.Linear(512, 256)                                  # hidden layer
        self.fc3 = nn.Linear(256, 47)                                   # output layer for 47 classes

        # Dropout nur im FC-Teil
        self.dropout = nn.Dropout(p=0.3)                                # Dropout mit p=0.3 new p=0.5 old

    def forward(self, x):
        # Conv Layers mit BatchNorm + ReLU
        x = F.relu(self.bn1(self.conv1(x)))           # 1 -> 16, 28x28
        x = F.relu(self.bn2(self.convStride1(x)))     # 16 -> 32, 14x14
        x = F.relu(self.bn3(self.conv2(x)))           # 32 -> 64, 14x14
        x = F.relu(self.bn4(self.convStride2(x)))     # 64 -> 128, 7x7

        # Flatten
        x = torch.flatten(x, 1)                       # n, 128*7*7

        # Fully Connected mit Dropout
        x = F.relu(self.fc1(x))                       # n, 512
        x = self.dropout(x)
        x = F.relu(self.fc2(x))                       # n, 256
        x = self.dropout(x)

        # Output Layer
        x = self.fc3(x)                               # n, 47

        return x
