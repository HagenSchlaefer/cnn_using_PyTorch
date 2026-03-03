# MyCnn2.py
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)                     # input channels=1, output channels=16, kernel size=3, padding=1 (28x28 -> 28x28)
        self.convStride1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # input channels=16, output channels=32, kernel size=2, stride=2, padding=1 (28x28 -> 14x14) 
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)                    # input channels=32, output channels=64, kernel size=3, padding=1 (14x14 -> 14x14)
        self.convStride2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # input channels=64, output channels=128, kernel size=2, stride=2, padding=1 (7x7 -> 7x7)
        self.fc1 = nn.Linear(128 * 7 * 7, 512)                          # 6272 = 128*7*7  
        self.fc2 = nn.Linear(512, 256)                                  # hidden layer
        self.fc3 = nn.Linear(256, 10)                                   # output layer for 10 classes

    def forward(self, x):
        # -> n, 1, 28, 28
        conv1 = F.relu(self.conv1(x))                   # -> n, 16, 28, 28
        self.conv1_x = conv1.detach()

        convStride1 = self.convStride1(conv1)           # -> n, 32, 14, 14
        self.convStride1_x = convStride1.detach()

        conv2 = F.relu(self.conv2(convStride1))         # -> n, 64, 14, 14
        self.conv2_x = conv2.detach()

        convStride2 = self.convStride2(conv2)           # -> n, 128, 7, 7
        self.convStride2_x = convStride2.detach()

        view = convStride2.view(-1, 128 * 7 * 7)        # -> n, 6272
        self.view_x = view.detach()

        fc1 = F.relu(self.fc1(view))                    # -> n, 512
        self.fc1_x = fc1.detach()

        fc2 = F.relu(self.fc2(fc1))                     # -> n, 256
        self.fc2_x = fc2.detach()

        fc3 = self.fc3(fc2)                             # -> n, 10
        self.fc3_x = fc3.detach()

        return fc3