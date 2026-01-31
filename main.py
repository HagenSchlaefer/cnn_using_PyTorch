#main.py
import torch
import torch.nn as nn

from MyCnn import ConvNet
from cnn import train, plot_metrics

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 64
learning_rate = 0.001

model = ConvNet().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size)
plot_metrics(losses, accuracies, num_epochs)