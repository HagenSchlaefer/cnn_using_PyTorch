#main.py
import torch
import torch.nn as nn

from MyCnn import ConvNet
from myCnn2 import ConvNet as ConvNet2
from cnn import train, plot_metrics, run
#from data import prep_image, show_image
#import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 15
batch_size = 64
learning_rate = 0.001

#model = ConvNet().to(device)
model = ConvNet2().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
# losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size)
# plot_metrics(losses, accuracies, num_epochs)

# Test the model
model.load_state_dict(torch.load(f'cnn_epoch{num_epochs}.pth'))
model.eval()
for i in range(10):
    for j in range(3):
        img_path = f"../TestData/{i}.png"
        pred = run(model, device, img_path)
        print(f"Image: {i}  Predicted label: {pred}")
        #print(f"Testing image of digit {i}, run {j+1}")
        j += 1
    i += 1
