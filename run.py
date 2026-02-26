#run.py
#train.py
from xml.parsers.expat import model

import torch
import torch.nn as nn

from MyCnn import ConvNet
from myCnn3 import ConvNet as ConvNet3
from cnn import run
from data import get_activation, load_emnist_mapping, save_activations, visualize_activations

def run_EMINIST_balanced():
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

    # Load the trained model
    model.load_state_dict(torch.load("EMNIST-balanced-CNN.pth"))
    model.eval()

    # load the EMNIST mapping to convert predicted labels to characters
    mapping = load_emnist_mapping()

    #register hooks to capture activations
    model.conv1.register_forward_hook(get_activation(activations, "conv1"))
    model.bn1.register_forward_hook(get_activation(activations, "bn1"))
    model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
    model.bn2.register_forward_hook(get_activation(activations, "bn2"))
    model.conv2.register_forward_hook(get_activation(activations, "conv2"))
    model.bn3.register_forward_hook(get_activation(activations, "bn3"))
    model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))
    model.bn4.register_forward_hook(get_activation(activations, "bn4"))

    model.fc1.register_forward_hook(get_activation(activations, "fc1"))
    model.fc2.register_forward_hook(get_activation(activations, "fc2"))
    model.fc3.register_forward_hook(get_activation(activations, "fc3"))

    #forward pass a test image
    pred, model = run(model, device, r"input\input.png")

    # output the predicted label
    # mapping of EMNIST labels
    print(f"Predicted label: {pred}")
    print(f"Predicted label: {mapping[pred]}")

    #labeld so that they can be easily identified in the visualization
    save_activations(activations["conv1"], "1_conv1")
    save_activations(activations["convStride1"], "2_convStride1")
    save_activations(activations["conv2"], "3_conv2")
    save_activations(activations["convStride2"], "4_convStride2")

    save_activations(activations["fc1"], "5_fc1")
    save_activations(activations["fc2"], "6_fc2")
    save_activations(activations["fc3"], "7_fc3")

    return mapping[pred]
