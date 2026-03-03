#train.py
# this file is used to train the CNN model on the MNIST and EMNIST datasets, and to test the model on the test dataset and input images.
# It also includes code to visualize the activations of the layers for a given input image.

# steps to train the model:
# 1. Load the dataset (MNIST or EMNIST)
# 2. Define the model architecture (ConvNet)
# 3. Define the loss function and optimizer and hyperparameters
# 4. Train the model for a specified number of epochs, and save the model weights
# 5. Test the model on the test dataset and output the accuracy

#train and test the CNN model
import os

import torch
import torch.nn as nn

from model_structures.MNIST_structure import ConvNet as ConvNet1
# from model_structures.EMNIST_letters_structure import ConvNet as ConvNet2
from model_structures.EMNIST_balanced_structure import ConvNet as ConvNet3

from .cnn import test, train, plot_metrics, run
from .data import get_activation, load_emnist_mapping, save_activations, visualize_activations

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 30
batch_size = 256
learning_rate = 0.0001

# Dictionary to store activations
activations = {}

#model = ConvNet().to(device)
model = ConvNet3().to(device)
loss_F = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# # Train the model
# modelName like MINIST-CNN, EMNIST-letters-CNN, EMNIST-balanced-CNN
# losses, accuracies = train(model, device, loss_F, optimizer, num_epochs, batch_size, modelName=f'cnn_epoch{num_epochs}')
# plot_metrics(losses, accuracies, num_epochs)


# Load the trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "EMNIST-balanced-CNN.pth")

model.load_state_dict(torch.load(model_path))
model.eval()

# # Test the model with EMNIST test dataset
test(model, device)

# # Run the model

# #register hooks to capture activations
# model.conv1.register_forward_hook(get_activation(activations, "conv1"))
# model.bn1.register_forward_hook(get_activation(activations, "bn1"))
# model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
# model.bn2.register_forward_hook(get_activation(activations, "bn2"))
# model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))
# model.bn3.register_forward_hook(get_activation(activations, "bn3"))
# model.conv2.register_forward_hook(get_activation(activations, "conv2"))
# model.bn4.register_forward_hook(get_activation(activations, "bn4"))

# model.fc1.register_forward_hook(get_activation(activations, "fc1"))
# model.fc2.register_forward_hook(get_activation(activations, "fc2"))
# model.fc3.register_forward_hook(get_activation(activations, "fc3"))

# #forward pass a test image
# img_path = f"../TestData/L.png"
# pred, model = run(model, device, img_path)
# # output the predicted label
# # mapping of EMNIST labels
# #remember to change the mapping file if using the letters model
# mapping = load_emnist_mapping()
# print(f"Predicted label: {pred}")
# print(f"Predicted label: {mapping[pred]}")

# #visualize the activations of the layers for the test image
# visualize_activations(activations["conv1"], "conv1")
# visualize_activations(activations["convStride1"], "convStride1")
# visualize_activations(activations["conv2"], "conv2")
# visualize_activations(activations["convStride2"], "convStride2")

# #visualize_activations(mapping[activations["flatten"]], "flatten")
# visualize_activations(activations["fc1"], "fc1")
# visualize_activations(activations["fc2"], "fc2")
# visualize_activations(activations["fc3"], "fc3")

# #save the activations of the layers for visualization
# #labeld so that they can be easily identified in the visualization
# save_activations(activations["conv1"], "1_conv1")
# save_activations(activations["convStride1"], "2_convStride1")
# save_activations(activations["conv2"], "3_conv2")
# save_activations(activations["convStride2"], "4_convStride2")

# save_activations(activations["fc1"], "5_fc1")
# save_activations(activations["fc2"], "6_fc2")
# save_activations(activations["fc3"], "7_fc3")

