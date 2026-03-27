#run.py
import os
import torch
import importlib
import traceback

import importlib.util, sys
import os

from .cnn import run
from .data import get_activation, load_emnist_mapping

def run_EMINIST(dataset: str, model_name: str, model_structure: str):
    #run the EMNIST balanced model on the input image and return the predicted label

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dictionary to store activations
    activations = {}

    layer_names = []

    # initialize the model
    ModelClass = load_model_class_flex(model_structure)
    model = ModelClass().to(device)

    # Load the trained model
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", model_name)

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    try:
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

        elif hasattr(checkpoint, "state_dict"):
            state_dict = checkpoint.state_dict()

        else:
            raise ValueError("Unknown model format")

        model.load_state_dict(state_dict)

    except Exception as e:
        traceback.print_exc()
        raise ValueError(f"Unknown model format: {e}")

    model.eval()

    mapping_path = ""

    match dataset:
        case "digits":
            mapping_path = os.path.join(BASE_DIR, "data", "EMNIST", "raw", "emnist-digits-mapping.txt")
        case "letters":
            mapping_path = os.path.join(BASE_DIR, "data", "EMNIST", "raw", "emnist-letters-mapping.txt")
        case "balanced":
            mapping_path = os.path.join(BASE_DIR, "data", "EMNIST", "raw", "emnist-balanced-mapping.txt")
        case _: # default case
            raise ValueError(f"Unknown dataset: {dataset}")

    # load the EMNIST mapping to convert predicted labels to characters
    mapping = load_emnist_mapping(mapping_path)

    for name, layer in model.named_modules():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.MaxPool2d)):
            layer_names.append(name)
            layer.register_forward_hook(get_activation(activations, name))

    #forward pass a the input image
    pred = run(model, device, r"input\input.png")

    # mapping of EMNIST labels
    return mapping[pred], activations, layer_names 

def load_model_class(model_structure: str):
    try:
        module = importlib.import_module(f"model_structures.{model_structure}")
    except ModuleNotFoundError:
        raise ValueError(f"Model structure module not found: {model_structure}")

    try:
        return getattr(module, "ConvNet")
    except AttributeError:
        raise ValueError(f"'ConvNet' class not found in {model_structure}")

def load_model_class_flex(module_name: str):
    

    module_path = os.path.join("model_structures", module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    # Search for first nn.Module class
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, torch.nn.Module):
            return attr

    raise ValueError("No torch.nn.Module class found")
# def run_EMINIST_letters():
#     #run the EMNIST letters model on the input image and return the predicted label

#     # ENIST letters model is not implemented yet, so we will use the balanced model for demonstration!!!

#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Dictionary to store activations
#     activations = {}

#     # initialize the model
#     model = ConvNet3().to(device)

#     # Load the trained model
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     model_path = os.path.join(BASE_DIR, "models", "EMNIST-balanced-CNN.pth")

#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     # load the EMNIST mapping to convert predicted labels to characters
#     # mapping = load_emnist_mapping(r"data\EMNIST\raw\emnist-letters-mapping.txt")
#     mapping = load_emnist_mapping()

#     #register hooks to capture activations
#     model.conv1.register_forward_hook(get_activation(activations, "conv1"))
#     model.bn1.register_forward_hook(get_activation(activations, "bn1"))
#     model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
#     model.bn2.register_forward_hook(get_activation(activations, "bn2"))
#     model.conv2.register_forward_hook(get_activation(activations, "conv2"))
#     model.bn3.register_forward_hook(get_activation(activations, "bn3"))
#     model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))
#     model.bn4.register_forward_hook(get_activation(activations, "bn4"))

#     model.fc1.register_forward_hook(get_activation(activations, "fc1"))
#     model.fc2.register_forward_hook(get_activation(activations, "fc2"))
#     model.fc3.register_forward_hook(get_activation(activations, "fc3"))

#     #forward pass a the input image
#     pred = run(model, device, r"input\input.png")

#     #save the activations of the layers for visualization
#     #labeld so that they can be easily identified in the visualization
#     save_activations(activations["conv1"], "1_conv1", mapping)
#     save_activations(activations["convStride1"], "2_convStride1")
#     save_activations(activations["conv2"], "3_conv2")
#     save_activations(activations["convStride2"], "4_convStride2")

#     save_activations(activations["fc1"], "5_fc1")
#     save_activations(activations["fc2"], "6_fc2")
#     save_activations(activations["fc3"], "7_fc3", show_x_labels=True, emnist_mapping_path= r"data/EMNIST/raw/emnist-balanced-mapping.txt")

#     # mapping of EMNIST labels
#     return mapping[pred]


# def run_MNIST():
#     #run the MNIST model on the input image and return the predicted label

#     # Device configuration
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Dictionary to store activations
#     activations = {}

#     # initialize the model
#     model = ConvNet1().to(device)

#     # Load the trained model
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     model_path = os.path.join(BASE_DIR, "models", "MNIST-CNN.pth")

#     model.load_state_dict(torch.load(model_path))
#     model.eval()

#     #register hooks to capture activations
#     model.conv1.register_forward_hook(get_activation(activations, "conv1"))
#     model.convStride1.register_forward_hook(get_activation(activations, "convStride1"))
#     model.conv2.register_forward_hook(get_activation(activations, "conv2"))
#     model.convStride2.register_forward_hook(get_activation(activations, "convStride2"))

#     model.fc1.register_forward_hook(get_activation(activations, "fc1"))
#     model.fc2.register_forward_hook(get_activation(activations, "fc2"))
#     model.fc3.register_forward_hook(get_activation(activations, "fc3"))

#     #forward pass a the input image
#     pred = run(model, device, r"input\input.png")

#     #save the activations of the layers for visualization
#     #labeld so that they can be easily identified in the visualization
#     save_activations(activations["conv1"], "1_conv1")
#     save_activations(activations["convStride1"], "2_convStride1")
#     save_activations(activations["conv2"], "3_conv2")
#     save_activations(activations["convStride2"], "4_convStride2")

#     save_activations(activations["fc1"], "5_fc1")
#     save_activations(activations["fc2"], "6_fc2")
#     save_activations(activations["fc3"], "7_fc3", show_x_labels=True)

#     # mapping of MNIST labels is just the digits 0-9, so we can directly return the predicted label
#     return pred
