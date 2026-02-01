import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import cv2


def load_mnist_cnn():
    # load MNIST data from local npz file
    path = pathlib.Path(__file__).parent.absolute() / "data" / "mnist.npz"
    with np.load(path) as f:
        images, labels = f["x_train"], f["y_train"]

    # normalize to 0-1
    images = images.astype("float32") / 255.0

    # CNN-Format: (batch, channels, height, width)
    images = images.reshape(-1, 1, 28, 28)

    # labels to int64
    labels = labels.astype(np.int64)

    return images, labels

def batch_generator(images, labels, batch_size=64, shuffle=True):
# generate batches
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield images[batch_idx], labels[batch_idx]

def batch_generator_augmented(images, labels, batch_size=64, shuffle=True, augment=True):
# generate batches with optional data augmentation

     # NumPy → Torch Tensor
    images = torch.from_numpy(images).float()   # (N,1,28,28)
    labels = torch.from_numpy(labels).long()    # (N,)
    
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    # Augmentation definieren
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        batch_imgs = images[batch_idx]
        batch_lbls = labels[batch_idx]

        # Augmentation auf jedes Bild anwenden, falls gewünscht
        if augment:
            batch_imgs = torch.stack([transform(img) for img in batch_imgs])

        yield batch_imgs, batch_lbls


def show_image(img):
    # img size (1, 28, 28) oder (28, 28)
    if img.ndim == 3:
        img = img[0]  # delete channel dimension

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

def show_feature_maps(feature_maps, max_maps=16):
    # feature_maps: (1, channels, H, W)
    maps = feature_maps[0]  # delete batch dimension
    num_maps = min(maps.shape[0], max_maps)

    cols = 4
    rows = (num_maps + cols - 1) // cols

    plt.figure(figsize=(10, 10))
    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(maps[i], cmap="gray")
        plt.axis("off")
    plt.show()


def show_all_feature_maps(feature_maps):
#view all feature maps in a grid

    # feature_maps: (1, C, H, W)
    maps = feature_maps[0]          # delete batch dimension
    maps = maps.unsqueeze(1)        # (C, 1, H, W) for make_grid

    grid = torchvision.utils.make_grid(maps, nrow=8, padding=1)
    plt.imshow(grid.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

def prep_image(image_path):
    # load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{image_path} nicht gefunden")
    # normalize to 0-1
    img = img.astype("float32") / 255.0

    img = 1 - img  # invert colors

    # CNN-Format: (batch, channels, height, width)
    img = img.reshape(-1, 1, 28, 28)

    # uint8 -> float32 für Berechnungen
    img = img.astype(np.float32)
    
    return img

