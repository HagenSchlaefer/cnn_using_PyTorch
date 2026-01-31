# MyCnn.py
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)      # input channels=1, output channels=2, kernel size=3, padding=1 (28x28 -> 28x28)
        self.pool1 = nn.MaxPool2d(2, 2)                 # 2x2 max pooling (28x28 -> 14x14) 
        self.conv2 = nn.Conv2d(2, 4, 3, padding=1)      # input channels=2, output channels=4, kernel size=3, padding=1 (14x14 -> 14x14)
        self.pool2 = nn.MaxPool2d(2, 2)                 # 2x2 max pooling (14x14 -> 7x7) 
        self.conv3 = nn.Conv2d(4, 8, 3, padding=1)      # input channels=4, output channels=8, kernel size=3, padding=1 (7x7 -> 7x7)
        self.fc1 = nn.Linear(8 * 7 * 7, 512)            # 392 = 8*7*7  
        self.fc2 = nn.Linear(512, 256)                  # hidden layer
        self.fc3 = nn.Linear(256, 10)                   # output layer for 10 classes

    def forward(self, x):
        # -> n, 1, 28, 28
        conv1 = F.relu(self.conv1(x))                   # -> n, 2, 28, 28
        self.conv1_x = conv1.detach()

        pool1 = self.pool1(conv1)                       # -> n, 2, 14, 14
        self.pool1_x = pool1.detach()

        conv2 = F.relu(self.conv2(pool1))               # -> n, 4, 14, 14
        self.conv2_x = conv2.detach()

        pool2 = self.pool2(conv2)                       # -> n, 4, 7, 7
        self.pool2_x = pool2.detach()

        conv3 = F.relu(self.conv3(pool2))               # -> n, 8, 7, 7
        self.conv3_x = conv3.detach()

        view = conv3.view(-1, 8 * 7 * 7)                # -> n, 392
        self.view_x = view.detach()

        fc1 = F.relu(self.fc1(view))                    # -> n, 512
        self.fc1_x = fc1.detach()

        fc2 = F.relu(self.fc2(fc1))                     # -> n, 256
        self.fc2_x = fc2.detach()

        fc3 = self.fc3(fc2)                             # -> n, 10
        self.fc3_x = fc3.detach()

        return fc3