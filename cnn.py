import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#import numpy as np

from data import batch_generator_augmented, load_mnist_cnn, prep_image, show_image#, show_image, batch_generator
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
        for i, (batch_imgs, batch_lbls) in enumerate(batch_generator_augmented(images_cnn, labels_cnn, batch_size=batch_size, augment=True)):
            
            images = batch_imgs.to(device)    # (batch_size, 1, 28, 28)
            labels = batch_lbls.to(device)    # (batch_size,)
                    
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

def run(model, device, image_path):
# predict label for a single image

    # Test with a single image
    image = prep_image(image_path)
    if image is None:
        print("Failed to preprocess image.")
        return None
    
    show_image(image[0])  # show the preprocessed image

    model.eval()
    with torch.no_grad():
        image = torch.from_numpy(image).float().to(device)  # (1, 1, 28, 28)
        output = model(image)
        pred = torch.argmax(output, dim=1)
        return pred.item()
    
    