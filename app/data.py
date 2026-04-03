import shutil

import numpy as np
import math
import pathlib
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
import cv2
import os

from torchvision.datasets import EMNIST
from torchvision import transforms

# EMNIST(
#     root="data2",
#     split="balanced",
#     train=True,
#     download=True
# )


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

def load_emnist_cnn(split="balanced",train=True):
# load EMNIST balanced split data from torchvision.datasets
    transform = transforms.Compose([
        transforms.ToTensor(),                                                  # -> [0,1], shape (1,28,28)
        transforms.Lambda(lambda x: torch.rot90(x, 3, [1,2]).contiguous()),     # rotate 3x90 degrees = -90 degrees
        transforms.Lambda(lambda x: torch.flip(x, [2]))                         # flip horizontal
    ])

    dataset = EMNIST(
        root="data",
        split=split,
        train=train,
        download=False,
        transform=transform
    )                           # dataset of PIL images and labels

    images = []
    labels = []

    for img, label in dataset:
        images.append(img.numpy())  # (1,28,28)
        labels.append(label)        # int (0-46)

    images = np.stack(images).astype("float32")   # (N,1,28,28)
    labels = np.array(labels, dtype=np.int64)

    return images, labels

def load_emnist_mapping(path: str = None):
    
    if path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, "data", "EMNIST", "raw", "emnist-balanced-mapping.txt")
    
    mapping = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            key, val = line.split()
            mapping[int(key)] = chr(int(val))
    
    if not mapping:
        raise ValueError(f"Mapping file is empty: {path}")

    return mapping

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

        # augmentation if enabled
        if augment:
            batch_imgs = torch.stack([transform(img) for img in batch_imgs])

        yield batch_imgs, batch_lbls


def show_image(img, block=True):
    # img size (1, 28, 28) oder (28, 28)
    if img.ndim == 3:
        img = img[0]  # delete channel dimension

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show(block=block)

def show_feature_maps(feature_maps):
#view all feature maps in a grid
    # feature_maps: (1, C, H, W)
    maps = feature_maps[0]          # delete batch dimension
    maps = maps.unsqueeze(1)        # (C, 1, H, W) for make_grid

    grid = torchvision.utils.make_grid(maps, nrow=8, padding=1) # make grid (C, 1, H, W) -> (1, H_grid, W_grid)
    grid = grid.permute(1, 2, 0).numpy()  # (H, W, 3)
    
    plt.imshow(grid)
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

def get_activation(activations, name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# def normalize_per_channel(x):
#     # normalise each channel to [0,1] separately
#     c, h, w = x.shape
#     x = x.view(c, -1)

#     min_vals = x.min(dim=1, keepdim=True)[0]
#     max_vals = x.max(dim=1, keepdim=True)[0]

#     x = (x - min_vals) / (max_vals - min_vals + 1e-6)
#     return x.view(c, h, w)

def normalize_per_channel(x):
    # normalise each channel to [0,1] separately
    c, h, w = x.shape
    x = x.view(c, -1)

    min_vals = x.min(dim=1, keepdim=True)[0]
    max_vals = x.max(dim=1, keepdim=True)[0]

    denom = max_vals - min_vals
    denom[denom == 0] = 1  # prevents division by 0

    x = (x - min_vals) / denom
    return x.view(c, h, w)

def visualize_activations(x, name="layer"):
# Visualize feature maps or activations of a layer
    x = x.cpu()

    # CONV FEATURE MAPS
    if x.dim() == 4:  # (B, C, H, W)
        maps = x[0]
        maps = normalize_per_channel(maps)
        maps = maps.unsqueeze(1) #(num_maps, 1, H, W)

        grid = torchvision.utils.make_grid(
            maps,
            nrow=8,
            padding=1,
            pad_value=0.0
        )

        grid = grid.permute(1, 2, 0).numpy()

        h, w, _ = grid.shape
        plt.figure(figsize=(w / 120, h / 120), dpi=120)
        plt.imshow(grid, interpolation="nearest")
        plt.title(name)
        plt.axis("off")
        plt.show()

    # FC / VIEW LAYERS
    elif x.dim() == 2:  # (B, N)
        vec = x[0].numpy()
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-6)

        plt.figure(figsize=(10, 2))
        plt.imshow(vec[np.newaxis, :], aspect="auto", cmap="viridis")
        plt.colorbar(label="activation")
        plt.title(name)
        plt.yticks([])
        plt.xlabel("Neuron index")
        plt.show()

    else:
        print(f"{name}: unsupported shape {x.shape}")

def clear_dir_safe(out_dir="outputs"):
    if os.path.exists(out_dir):
        for f in os.listdir(out_dir):
            fp = os.path.join(out_dir, f)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    shutil.rmtree(fp)
            except PermissionError:
                # Windows blocks files that are open in another program, ignore those
                pass

def save_activations_olt(
    x: torch.Tensor,
    name: str = "layer",
    out_dir: str = "outputs",
    normalize: bool = True,
    padding: int = 1
):
    os.makedirs(out_dir, exist_ok=True)
    assert x.dim() == 4, f"Erwarte (B, C, H, W), bekam {tuple(x.shape)}"

    with torch.no_grad():
        maps = x[0].detach().cpu()  # (C,H,W)
        if normalize:
            maps = normalize_per_channel(maps)
            pad_value = 0.0
        else:
            pad_value = maps.min().item()

        maps = maps.unsqueeze(1)  # (C,1,H,W)

        C = maps.shape[0]
        ncols = math.ceil(math.sqrt(C))

        grid = torchvision.utils.make_grid(
            maps, nrow=ncols, padding=padding, pad_value=pad_value
        )  # (3, H_total, W_total)

        # Als PNG speichern (wie bisher)
        grid = grid.permute(1, 2, 0).numpy()
        plt.imsave(os.path.join(out_dir, f"{name}.png"), grid)
        plt.close()

def save_activations_good(
    x: torch.Tensor,
    name: str = "layer",
    out_dir: str = "outputs",
    normalize: bool = True,
    padding: int = 1
):
    # save activations as a grid of images (for conv layers) or bar chart (for fc layers)
    os.makedirs(out_dir, exist_ok=True)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x muss torch.Tensor sein, bekam {type(x)}")

    x = x.detach().cpu()

    # ----------------------------------------
    # CONV FEATURE MAPS: (B, C, H, W)
    # ----------------------------------------
    if x.dim() == 4:
        maps = x[0]  # (C, H, W)

        if normalize:
            maps = normalize_per_channel(maps)
            pad_value = 0.0
        else:
            pad_value = maps.min().item()

        maps = maps.unsqueeze(1)  # (C, 1, H, W)

        num_maps = maps.shape[0]
        ncols = math.ceil(math.sqrt(num_maps))  # square grid

        grid = torchvision.utils.make_grid(
            maps, nrow=ncols, padding=padding, pad_value=pad_value
        )  # (3, H_total, W_total)

        grid = grid.permute(1, 2, 0).numpy()
        plt.imsave(os.path.join(out_dir, f"{name}.png"), grid, cmap="gray")
        plt.close()
        return

    # ----------------------------------------
    # FC / VIEW LAYERS: (B, N)  -> B=1, N=number of features Bar chart
    # ----------------------------------------
    if x.dim() == 2:
        vec = x[0].float().numpy()  # (N,)
        N = vec.shape[0]

        if normalize:
            vmin, vmax = vec.min(), vec.max()
            vec = (vec - vmin) / (max(vmax - vmin, 1e-6))  # [0,1]

        # dynamic figure size based on number of features (N) - more features -> wider figure
        width = max(8, min(20, N / 25))   
        height = 4
        fig, ax = plt.subplots(figsize=(width, height), dpi=120)

        x_idx = np.arange(N)
        ax.bar(x_idx, vec, color="#4c78a8", edgecolor="#2c3e50", linewidth=0.5)

        ax.set_title(f"{name} – FC-Ausgänge (N={N})", fontsize=10)
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0.0 if normalize else min(0.0, vec.min()), vec.max() * 1.05 if vec.size > 0 else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if N <= 40:
            ax.set_xticks(x_idx)
            ax.set_xticklabels([str(i) for i in x_idx], fontsize=8, rotation=90)
        else:
            step = max(1, N // 20)
            ticks = np.arange(0, N, step)
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(i) for i in ticks], fontsize=8, rotation=0)

        ax.set_xlabel("Feature-Index")
        ax.set_ylabel("Wert" + (" (normiert)" if normalize else ""))

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close(fig)
        return
    # ----------------------------------------
    print(f"{name}: unsupported shape {tuple(x.shape)}")

def prepare_featuremaps(x: torch.Tensor):
    x = x.detach().cpu()

    # CNN Layer (B,C,H,W)
    if x.dim() == 4:
        maps = x[0]  # (C,H,W)

        if maps.dim() != 3:
            raise ValueError(f"Expected (C,H,W), got {maps.shape}")

        maps = normalize_per_channel(maps)
        return maps.numpy(), "conv"

    # FC Layer (B,N)
    elif x.dim() == 2:
        vec = x[0].float()

        vmin, vmax = vec.min(), vec.max()
        denom = max((vmax - vmin).item(), 1e-5)
        vec = (vec - vmin) / denom

        return vec.numpy(), "fc"

    else:
        raise ValueError(f"Unsupported shape: {x.shape}")

def save_activations(
    x: torch.Tensor,
    name: str = "layer",
    out_dir: str = "outputs",
    normalize: bool = True,
    padding: int = 1,
    image_size: int = 280,
    show_x_labels: bool = False,
    emnist_mapping_path: str = None
):
    os.makedirs(out_dir, exist_ok=True)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a {type(x)}")

    x = x.detach().cpu()

    # ============================================================
    # CONV FEATURE MAPS (B,C,H,W)
    # ============================================================
    if x.dim() == 4:

        maps = x[0]  # (C,H,W)

        if normalize:
            maps = normalize_per_channel(maps)
            pad_value = 0.0
        else:
            pad_value = maps.min().item()

        maps = maps.unsqueeze(1)

        num_maps = maps.shape[0]
        ncols = math.ceil(math.sqrt(num_maps))

        grid = torchvision.utils.make_grid(
            maps,
            nrow=ncols,
            padding=padding,
            pad_value=pad_value
        )

        grid = grid.permute(1, 2, 0).numpy()

        fig = plt.figure(figsize=(image_size/50, image_size/50), dpi=100)
        plt.imshow(grid, cmap="gray")
        plt.axis("off")

        plt.savefig(
            os.path.join(out_dir, f"{name}.png"),
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()
        return


    # ============================================================
    # FC LAYER (B,N)
    # ============================================================
    if x.dim() == 2:

        vec = x[0].float().numpy()
        N = vec.shape[0]

        if normalize:
            vmin, vmax = vec.min(), vec.max()
            vec = (vec - vmin) / (max(vmax - vmin, 1e-6))

        fig, ax = plt.subplots(
            figsize=(image_size/50, image_size/50),
            dpi=100
        )

        x_idx = np.arange(N)

        ax.bar(
            x_idx,
            vec,
            width=0.5,
            color="#4c78a8",
            edgecolor="black",
            linewidth=0.5
        )

        #ax.set_title(name, fontsize=10)

        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0, max(vec.max()*1.1, 1e-3))

        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # -------------------------------------------------
    # X LABELS optional
    # -------------------------------------------------
    if show_x_labels:
        ax.set_xlabel("Class")
        ax.set_xticks(x_idx)
        if emnist_mapping_path != None:
            mapping = load_emnist_mapping(emnist_mapping_path)
            ax.set_xticklabels([mapping[i] for i in x_idx], fontsize=8)
        else:
            ax.set_xticklabels([str(i) for i in x_idx], fontsize=8)
    else:
        ax.set_xticks([])

    ax.set_ylabel("Activation" if not normalize else "Activation (norm)")

    plt.tight_layout()

    plt.savefig(
        os.path.join(out_dir, f"{name}.png"),
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.close()
    return

#XX
def show_activations_animated(
    x: torch.Tensor,
    name: str = "layer",
    normalize: bool = True,
    interval_ms: int = 50,
    padding: int = 1,
    cmap: str = "gray"
):
    """
    Zeigt die Aktivierungen als quadratisches Grid in einem Matplotlib-Fenster
    und lädt die Bildzeilen animiert von oben nach unten.

    Parameter:
      x          : Tensor (B, C, H, W)
      name       : Titel im Plot
      normalize  : True -> pro Kanal auf [0,1] normalisieren (vmin=0, vmax=1).
                   False -> Rohwerte verwenden; vmin/vmax global aus x[0] bestimmt.
      interval_ms: Pause zwischen Zeilen-Updates (in Millisekunden)
      padding    : Pixel-Rand zwischen den einzelnen Karten
      cmap       : Colormap für imshow (z.B. "gray", "magma", "viridis")
    """
    assert x.dim() == 4, f"Erwarte (B, C, H, W), bekam {tuple(x.shape)}"
    with torch.no_grad():
        maps = x[0].detach().cpu()  # (C, H, W)

        if normalize:
            maps = normalize_per_channel(maps)
            vmin, vmax = 0.0, 1.0
            pad_value = 0.0
        else:
            vmin = maps.min().item()
            vmax = maps.max().item()
            pad_value = vmin  # Hintergrund auf globales Minimum setzen

        # (C, 1, H, W) für torchvision.make_grid
        maps = maps.unsqueeze(1)

        C = maps.shape[0]
        # Quadratische Anordnung: 16 -> 4x4, 32 -> 6x6, etc.
        ncols = math.ceil(math.sqrt(C))  # Spalten = Zeilen = ceil(sqrt(C))

        # Grid bauen (make_grid erzeugt bei 1-Kanal i.d.R. 3 Kanäle; wir nehmen später Kanal 0)
        grid_t = torchvision.utils.make_grid(
            maps, nrow=ncols, padding=padding, pad_value=pad_value
        )  # (3, H_total, W_total)

        # Auf ein 2D-Bild (H_total, W_total) für "gray" reduzieren
        if grid_t.shape[0] == 1:
            grid = grid_t.squeeze(0).numpy()
        else:
            grid = grid_t[0].numpy()  # R-Kanal reicht; alle 3 sind identisch bei 1-Kanal-Input

    # Animation: Zeile für Zeile aktualisieren
    fig, ax = plt.subplots()
    ax.set_title(f"{name} – {C} Kanäle ({ncols}×{ncols})")
    ax.axis("off")
    ax.set_aspect("equal")  # quadratische Pixel

    Htot, Wtot = grid.shape
    current = np.full_like(grid, fill_value=vmin)  # Start mit Hintergrund (min)
    img = ax.imshow(current, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    for r in range(Htot):
        current[r, :] = grid[r, :]     # Zeile r übernehmen
        img.set_data(current)
        plt.pause(interval_ms / 1000.0)  # GUI-Redraw + gewünschte Verzögerung

    plt.show()

def show_activations_animated_sync(
    x,
    name="layer",
    normalize=True,
    interval_ms=500,
    padding=1,
    cmap="gray"
):
    assert x.dim() == 4, "x must be (B, C, H, W)"

    # --- Prepare maps ---
    maps = x[0].detach().cpu()  # (C,H,W)

    if normalize:
        maps = normalize_per_channel(maps)
        vmin, vmax = 0.0, 1.0
        pad_value = 0.0
    else:
        vmin = maps.min().item()
        vmax = maps.max().item()
        pad_value = vmin

    C, H, W = maps.shape

    # empty container for animation
    current_maps = torch.full_like(maps, fill_value=pad_value)  # (C,H,W)

    # grid width
    ncols = math.ceil(math.sqrt(C))

    # --- Setup matplotlib ---
    fig, ax = plt.subplots()
    ax.set_title(f"{name} – {C} Featuremaps ({ncols}×{ncols})")
    ax.axis("off")

    # initial grid
    grid = torchvision.utils.make_grid(
        current_maps.unsqueeze(1), nrow=ncols, padding=padding, pad_value=pad_value
    )[0].numpy()

    img = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal")

    # --- Animation ---
    for row in range(H):
        # update ALL featuremaps in this row
        current_maps[:, row, :] = maps[:, row, :]

        # rebuild the grid
        grid = torchvision.utils.make_grid(
            current_maps.unsqueeze(1), nrow=ncols, padding=padding, pad_value=pad_value
        )[0].numpy()

        img.set_data(grid)
        plt.pause(interval_ms / 1000.0)

    plt.show()

def save_feature_maps(feature_map, layer_name, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    layer_dir = os.path.join(out_dir, layer_name)
    os.makedirs(layer_dir, exist_ok=True)

    fmap = feature_map[0]  # Batch 0 → (C, H, W)

    for i in range(fmap.shape[0]):
        img = fmap[i].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        plt.imsave(
            os.path.join(layer_dir, f"channel_{i:03d}.png"),
            img,
            cmap="gray"
        )

