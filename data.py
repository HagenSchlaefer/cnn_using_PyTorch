import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torchvision

def load_mnist_cnn():
    # MNIST laden
    path = pathlib.Path(__file__).parent.absolute() / "data" / "mnist.npz"
    with np.load(path) as f:
        images, labels = f["x_train"], f["y_train"]

    # Normalisieren (0–1)
    images = images.astype("float32") / 255.0

    # CNN-Format: (batch, channels, height, width)
    images = images.reshape(-1, 1, 28, 28)

    # Labels als Integer lassen (kein One-Hot nötig für CNN + Softmax)
    labels = labels.astype(np.int64)

    return images, labels

def batch_generator(images, labels, batch_size=64, shuffle=True):
    indices = np.arange(len(images))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(images), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield images[batch_idx], labels[batch_idx]

def show_image(img):
    # img erwartet: (1, 28, 28) oder (28, 28)
    if img.ndim == 3:
        img = img[0]  # Kanal entfernen

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()

def show_feature_maps(feature_maps, max_maps=16):
    # feature_maps: (1, channels, H, W)
    maps = feature_maps[0]  # Batch entfernen
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
    maps = feature_maps[0]          # Batch entfernen
    maps = maps.unsqueeze(1)        # (C, 1, H, W) für make_grid

    grid = torchvision.utils.make_grid(maps, nrow=8, padding=1)
    plt.imshow(grid.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

