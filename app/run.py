#run.py
#train.py
import os
from xml.parsers.expat import model

import torch
import torch.nn as nn

from model_structures.MNIST_structure import ConvNet as ConvNet1
# from model_structures.EMNIST_letters_structure import ConvNet as ConvNet2
from model_structures.EMNIST_balanced_structure import ConvNet as ConvNet3
from .cnn import run
from .data import get_activation, load_emnist_mapping, save_activations, visualize_activations

def run_EMINIST_balanced():
    #run the EMNIST balanced model on the input image and return the predicted label

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dictionary to store activations
    activations = {}

    # initialize the model
    model = ConvNet3().to(device)

    # Load the trained model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "EMNIST-balanced-CNN.pth")

    model.load_state_dict(torch.load(model_path))
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

    #forward pass a the input image
    pred, model = run(model, device, r"input\input.png")

    #save the activations of the layers for visualization
    #labeld so that they can be easily identified in the visualization
    save_activations(activations["conv1"], "1_conv1")
    save_activations(activations["convStride1"], "2_convStride1")
    save_activations(activations["conv2"], "3_conv2")
    save_activations(activations["convStride2"], "4_convStride2")

    save_activations(activations["fc1"], "5_fc1")
    save_activations(activations["fc2"], "6_fc2")
    save_activations(activations["fc3"], "7_fc3", show_x_labels=True, emnist_mapping_path= r"data/EMNIST/raw/emnist-balanced-mapping.txt")

    # mapping of EMNIST labels
    return mapping[pred]


def run_EMINIST_letters():
    #run the EMNIST letters model on the input image and return the predicted label

    # ENIST letters model is not implemented yet, so we will use the balanced model for demonstration!!!

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dictionary to store activations
    activations = {}

    # initialize the model
    model = ConvNet3().to(device)

    # Load the trained model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "EMNIST-balanced-CNN.pth")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # load the EMNIST mapping to convert predicted labels to characters
    # mapping = load_emnist_mapping(r"data\EMNIST\raw\emnist-letters-mapping.txt")
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

    #forward pass a the input image
    pred, model = run(model, device, r"input\input.png")

    #save the activations of the layers for visualization
    #labeld so that they can be easily identified in the visualization
    save_activations(activations["conv1"], "1_conv1", mapping)
    save_activations(activations["convStride1"], "2_convStride1")
    save_activations(activations["conv2"], "3_conv2")
    save_activations(activations["convStride2"], "4_convStride2")

    save_activations(activations["fc1"], "5_fc1")
    save_activations(activations["fc2"], "6_fc2")
    save_activations(activations["fc3"], "7_fc3", show_x_labels=True, emnist_mapping_path= r"data/EMNIST/raw/emnist-balanced-mapping.txt")

    # mapping of EMNIST labels
    return mapping[pred]


def run_MNIST():
    #run the MNIST model on the input image and return the predicted label

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dictionary to store activations
    activations = {}

    # initialize the model
    model = ConvNet1().to(device)

    # Load the trained model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "MNIST-CNN.pth")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    #register hooks to capture activations
    model.conv1.register_forward_hook(get_activation(activations, "conv1"))
    model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
    model.conv2.register_forward_hook(get_activation(activations, "conv2"))
    model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))

    model.fc1.register_forward_hook(get_activation(activations, "fc1"))
    model.fc2.register_forward_hook(get_activation(activations, "fc2"))
    model.fc3.register_forward_hook(get_activation(activations, "fc3"))

    #forward pass a the input image
    pred, model = run(model, device, r"input\input.png")

    #save the activations of the layers for visualization
    #labeld so that they can be easily identified in the visualization
    save_activations(activations["conv1"], "1_conv1")
    save_activations(activations["convStride1"], "2_convStride1")
    save_activations(activations["conv2"], "3_conv2")
    save_activations(activations["convStride2"], "4_convStride2")

    save_activations(activations["fc1"], "5_fc1")
    save_activations(activations["fc2"], "6_fc2")
    save_activations(activations["fc3"], "7_fc3", show_x_labels=True)

    # mapping of MNIST labels is just the digits 0-9, so we can directly return the predicted label
    return pred

run_EMINIST_balanced()