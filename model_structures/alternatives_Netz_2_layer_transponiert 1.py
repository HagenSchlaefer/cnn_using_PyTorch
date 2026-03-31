import torch 
import torch.nn as nn
import torch.nn.functional as F

 
# Neuronale Netz anlegen
class MeinNetz(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)          # BatchNorm nach conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)          # BatchNorm nach conv2

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.dropout_fc = nn.Dropout(0.4)      # Dropout für Fully Connected Layer
        self.fc2 = nn.Linear(128, 47)
 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x