import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#import numpy as np

from data import load_mnist_cnn, batch_generator#, show_image
#from MyCnn import ConvNet

def train(model, device, loss_fn, optimizer, num_epochs, batch_size):
# Train the model
    
    # get mnist training data
    images_cnn, labels_cnn = load_mnist_cnn()
    # print(images_cnn.shape)  # (60000, 1, 28, 28)
    # print(labels_cnn.shape)  # (60000,)

    model.train()

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (batch_imgs, batch_lbls) in enumerate(batch_generator(images_cnn, labels_cnn, batch_size=batch_size)):

            images = batch_imgs.to(device)        # (B, 1, 28, 28)
            labels = batch_lbls.to(device)        # (B,)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / (i + 1)
        accuracy = 100 * correct / total if total > 0 else 0

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%')

    print('Finished Training')
    torch.save(model.state_dict(), f'cnn_epoch{epoch+1}.pth')
    return epoch_losses, epoch_accuracies

def plot_metrics(losses, accuracies, num_epochs):
# plot loss and accuracy curves
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    plt.figure()
    plt.plot(epochs, accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy")
    plt.show()

