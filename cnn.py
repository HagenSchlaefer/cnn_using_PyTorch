import torch
import torch.nn as nn
#import torch.nn.functional as F
import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from data import get_mnist, load_mnist_cnn, batch_generator
from MyCnn import ConvNet

# get mnist training data
images_cnn, labels_cnn = load_mnist_cnn()
print(images_cnn.shape)  # (60000, 1, 28, 28)
print(labels_cnn.shape)  # (60000,) 


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 5
batch_size = 4
learning_rate = 0.001

model = ConvNet().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for batch_imgs, batch_lbls in batch_generator(images_cnn, labels_cnn, batch_size=32):
    print(batch_imgs.shape)  # (32, 1, 28, 28)
    print(batch_lbls.shape)  # (32,)
    print(batch_lbls)      # tensor of size (32,)
    break


# # Test von mir
# for epoch in range(num_epochs):
#     for img, l in zip(images, labels):
#         print(img.shape)  # (784, 1)
#         print(l.shape)    # (10, 1)
#         img_reshape = img.reshape(1, 1, 28, 28)  
#         print(img_reshape.shape) # (1, 1, 28, 28)   #(batch, channels, height, width)
#         print(img_reshape)  
#         img_tensor = torch.tensor(img_reshape, dtype=torch.float32).to(device)
#         imshow(img_tensor.cpu())
#         #img_tensor = img_tensor.unsqueeze(0)  # add batch dimension -> (1,
#         break
#     break

# n_total_steps = len(train_loader)
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
#         # input_layer: 3 input channels, 6 output channels, 5 kernel size
#         images = images.to(device)
#         labels = labels.to(device)

#         # Forward pass
#         outputs = model(images)
#         loss = loss_F(outputs, labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if (i+1) % 2000 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# print('Finished Training')
# PATH = './cnn.pth'
# torch.save(model.state_dict(), PATH)

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     n_class_correct = [0 for i in range(10)]
#     n_class_samples = [0 for i in range(10)]
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
        
#         for i in range(batch_size):
#             label = labels[i]
#             pred = predicted[i]
#             if (label == pred):
#                 n_class_correct[label] += 1
#             n_class_samples[label] += 1

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network: {acc} %')

#     for i in range(10):
#         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#         print(f'Accuracy of {classes[i]}: {acc} %')
