# import math
# import numpy as np
# import torch
# import torchvision
# import matplotlib.pyplot as plt


# import numpy as np
# import matplotlib.pyplot as plt

# # featuremap = np.random.rand(26,26)

# # fig, ax = plt.subplots()

# # current = np.ones_like(featuremap)

# # img = ax.imshow(current, cmap="gray", vmin=0, vmax=1)
# # ax.axis("off")

# # for row in range(featuremap.shape[0]):

# #     current[row, :] = featuremap[row, :]
# #     img.set_data(current)

# #     plt.pause(0.05)

# # plt.show()



# import os
# import math
# import numpy as np
# import torch
# import torchvision
# import matplotlib.pyplot as plt

# def normalize_per_channel(x: torch.Tensor):
#     # x: (C, H, W)
#     c, h, w = x.shape
#     x_flat = x.view(c, -1)
#     min_vals = x_flat.min(dim=1, keepdim=True)[0]
#     max_vals = x_flat.max(dim=1, keepdim=True)[0]
#     x_norm = (x_flat - min_vals) / (max_vals - min_vals + 1e-6)
#     return x_norm.view(c, h, w)

# def save_activations_old(
#     x: torch.Tensor,
#     name: str = "layer",
#     out_dir: str = "outputs",
#     normalize: bool = True,
#     padding: int = 1
# ):
#     # save activations as a grid of images (for conv layers) or bar chart (for fc layers)
#     os.makedirs(out_dir, exist_ok=True)

#     if not isinstance(x, torch.Tensor):
#         raise TypeError(f"x muss torch.Tensor sein, bekam {type(x)}")

#     x = x.detach().cpu()

#     # ----------------------------------------
#     # CONV FEATURE MAPS: (B, C, H, W)
#     # ----------------------------------------
#     if x.dim() == 4:
#         maps = x[0]  # (C, H, W)

#         if normalize:
#             maps = normalize_per_channel(maps)
#             pad_value = 0.0
#         else:
#             pad_value = maps.min().item()

#         maps = maps.unsqueeze(1)  # (C, 1, H, W)

#         num_maps = maps.shape[0]
#         ncols = math.ceil(math.sqrt(num_maps))  # square grid

#         grid = torchvision.utils.make_grid(
#             maps, nrow=ncols, padding=padding, pad_value=pad_value
#         )  # (3, H_total, W_total)

#         grid = grid.permute(1, 2, 0).numpy()
#         plt.imsave(os.path.join(out_dir, f"{name}.png"), grid, cmap="gray")
#         plt.close()
#         return

#     # ----------------------------------------
#     # FC / VIEW LAYERS: (B, N)  -> B=1, N=number of features Bar chart
#     # ----------------------------------------
#     if x.dim() == 2:
#         vec = x[0].float().numpy()  # (N,)
#         N = vec.shape[0]

#         if normalize:
#             vmin, vmax = vec.min(), vec.max()
#             vec = (vec - vmin) / (max(vmax - vmin, 1e-6))  # [0,1]

#         # dynamic figure size based on number of features (N) - more features -> wider figure
#         width = max(8, min(20, N / 25))   
#         height = 4
#         fig, ax = plt.subplots(figsize=(width, height), dpi=120)

#         x_idx = np.arange(N)
#         ax.bar(x_idx, vec, color="#4c78a8", edgecolor="#2c3e50", linewidth=0.5)

#         ax.set_title(f"{name} – FC-Ausgänge (N={N})", fontsize=10)
#         ax.set_xlim(-0.5, N - 0.5)
#         ax.set_ylim(0.0 if normalize else min(0.0, vec.min()), vec.max() * 1.05 if vec.size > 0 else 1.0)
#         ax.grid(axis="y", linestyle="--", alpha=0.3)

#         if N <= 40:
#             ax.set_xticks(x_idx)
#             ax.set_xticklabels([str(i) for i in x_idx], fontsize=8, rotation=90)
#         else:
#             step = max(1, N // 20)
#             ticks = np.arange(0, N, step)
#             ax.set_xticks(ticks)
#             ax.set_xticklabels([str(i) for i in ticks], fontsize=8, rotation=0)

#         ax.set_xlabel("Feature-Index")
#         ax.set_ylabel("Wert" + (" (normiert)" if normalize else ""))

#         plt.tight_layout()
#         fig.savefig(os.path.join(out_dir, f"{name}.png"))
#         plt.close(fig)
#         return
#     # ----------------------------------------
#     print(f"{name}: unsupported shape {tuple(x.shape)}")



# def normalize_per_channel(x):
#     c, h, w = x.shape
#     x = x.view(c, -1)
#     min_vals = x.min(dim=1, keepdim=True)[0]
#     max_vals = x.max(dim=1, keepdim=True)[0]
#     x = (x - min_vals) / (max_vals - min_vals + 1e-6)
#     return x.view(c, h, w)

# def show_activations_animated_sync(
#     x,
#     name="layer",
#     normalize=True,
#     interval_ms=50,
#     padding=1,
#     cmap="gray"
# ):
#     assert x.dim() == 4, "x must be (B, C, H, W)"

#     # --- Prepare maps ---
#     maps = x[0].detach().cpu()  # (C,H,W)

#     if normalize:
#         maps = normalize_per_channel(maps)
#         vmin, vmax = 0.0, 1.0
#         pad_value = 0.0
#     else:
#         vmin = maps.min().item()
#         vmax = maps.max().item()
#         pad_value = vmin

#     C, H, W = maps.shape

#     # empty container for animation
#     current_maps = torch.full_like(maps, fill_value=pad_value)  # (C,H,W)

#     # grid width
#     ncols = math.ceil(math.sqrt(C))

#     # --- Setup matplotlib ---
#     fig, ax = plt.subplots()
#     ax.set_title(f"{name} – {C} Featuremaps ({ncols}×{ncols})")
#     ax.axis("off")

#     # initial grid
#     grid = torchvision.utils.make_grid(
#         current_maps.unsqueeze(1), nrow=ncols, padding=padding, pad_value=pad_value
#     )[0].numpy()

#     img = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.set_aspect("equal")

#     # --- Animation ---
#     for row in range(H):
#         # update ALL featuremaps in this row
#         current_maps[:, row, :] = maps[:, row, :]

#         # rebuild the grid
#         grid = torchvision.utils.make_grid(
#             current_maps.unsqueeze(1), nrow=ncols, padding=padding, pad_value=pad_value
#         )[0].numpy()

#         img.set_data(grid)
#         plt.pause(interval_ms / 1000.0)

#     plt.show()



# import os
# import math
# import numpy as np
# import torch
# import torchvision
# import matplotlib.pyplot as plt

# def normalize_per_channel(x: torch.Tensor):
#     # normalise each channel to [0,1] separately

#     c, h, w = x.shape
#     x = x.view(c, -1)
#     min_vals = x.min(dim=1, keepdim=True)[0]
#     max_vals = x.max(dim=1, keepdim=True)[0]
#     x = (x - min_vals) / (max_vals - min_vals + 1e-6)
#     return x.view(c, h, w)

# def show_activations_animated(
#     x: torch.Tensor,
#     name: str = "layer",
#     normalize: bool = True,
#     interval_ms: int = 50,
#     padding: int = 1,
#     cmap: str = "gray"
# ):
#     """
#     Zeigt die Aktivierungen als quadratisches Grid in einem Matplotlib-Fenster
#     und lädt die Bildzeilen animiert von oben nach unten.

#     Parameter:
#       x          : Tensor (B, C, H, W)
#       name       : Titel im Plot
#       normalize  : True -> pro Kanal auf [0,1] normalisieren (vmin=0, vmax=1).
#                    False -> Rohwerte verwenden; vmin/vmax global aus x[0] bestimmt.
#       interval_ms: Pause zwischen Zeilen-Updates (in Millisekunden)
#       padding    : Pixel-Rand zwischen den einzelnen Karten
#       cmap       : Colormap für imshow (z.B. "gray", "magma", "viridis")
#     """
#     assert x.dim() == 4, f"Erwarte (B, C, H, W), bekam {tuple(x.shape)}"
#     with torch.no_grad():
#         maps = x[0].detach().cpu()  # (C, H, W)

#         if normalize:
#             maps = normalize_per_channel(maps)
#             vmin, vmax = 0.0, 1.0
#             pad_value = 0.0
#         else:
#             vmin = maps.min().item()
#             vmax = maps.max().item()
#             pad_value = vmin  # Hintergrund auf globales Minimum setzen

#         # (C, 1, H, W) für torchvision.make_grid
#         maps = maps.unsqueeze(1)

#         C = maps.shape[0]
#         # Quadratische Anordnung: 16 -> 4x4, 32 -> 6x6, etc.
#         ncols = math.ceil(math.sqrt(C))  # Spalten = Zeilen = ceil(sqrt(C))

#         # Grid bauen (make_grid erzeugt bei 1-Kanal i.d.R. 3 Kanäle; wir nehmen später Kanal 0)
#         grid_t = torchvision.utils.make_grid(
#             maps, nrow=ncols, padding=padding, pad_value=pad_value
#         )  # (3, H_total, W_total)

#         # Auf ein 2D-Bild (H_total, W_total) für "gray" reduzieren
#         if grid_t.shape[0] == 1:
#             grid = grid_t.squeeze(0).numpy()
#         else:
#             grid = grid_t[0].numpy()  # R-Kanal reicht; alle 3 sind identisch bei 1-Kanal-Input

#     # Animation: Zeile für Zeile aktualisieren
#     fig, ax = plt.subplots()
#     ax.set_title(f"{name} – {C} Kanäle ({ncols}×{ncols})")
#     ax.axis("off")
#     ax.set_aspect("equal")  # quadratische Pixel

#     Htot, Wtot = grid.shape
#     current = np.full_like(grid, fill_value=vmin)  # Start mit Hintergrund (min)
#     img = ax.imshow(current, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

#     for r in range(Htot):
#         current[r, :] = grid[r, :]     # Zeile r übernehmen
#         img.set_data(current)
#         plt.pause(interval_ms / 1000.0)  # GUI-Redraw + gewünschte Verzögerung

#     plt.show()

# # (Optional) Wenn du die bisherige Speicherfunktion behalten willst, kannst du sie lassen.
# # Hier leicht bereinigt, damit sie dieselbe Quadrat-Logik nutzt, falls du weiterhin PNGs speichern möchtest.
# def save_activations(
#     x: torch.Tensor,
#     name: str = "layer",
#     out_dir: str = "outputs",
#     normalize: bool = True,
#     padding: int = 1
# ):
#     os.makedirs(out_dir, exist_ok=True)
#     assert x.dim() == 4, f"Erwarte (B, C, H, W), bekam {tuple(x.shape)}"

#     with torch.no_grad():
#         maps = x[0].detach().cpu()  # (C,H,W)
#         if normalize:
#             maps = normalize_per_channel(maps)
#             pad_value = 0.0
#         else:
#             pad_value = maps.min().item()

#         maps = maps.unsqueeze(1)  # (C,1,H,W)

#         C = maps.shape[0]
#         ncols = math.ceil(math.sqrt(C))

#         grid = torchvision.utils.make_grid(
#             maps, nrow=ncols, padding=padding, pad_value=pad_value
#         )  # (3, H_total, W_total)

#         # Als PNG speichern (wie bisher)
#         grid = grid.permute(1, 2, 0).numpy()
#         plt.imsave(os.path.join(out_dir, f"{name}.png"), grid)
#         plt.close()

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel
)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("CNN GUI")
        self.resize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)

        # Hauptlayout
        main_layout = QVBoxLayout()

        # Titel
        title = QLabel("CNN Visualizer")
        title.setStyleSheet("font-size: 20px")

        main_layout.addWidget(title)

        # Buttonreihe
        button_layout = QHBoxLayout()

        btn1 = QPushButton("Load Image")
        btn2 = QPushButton("Run Network")
        btn3 = QPushButton("Open Outputs")

        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)
        button_layout.addWidget(btn3)

        main_layout.addLayout(button_layout)

        # Platzhalter für Content
        content = QLabel("Hier kommt später dein Bild / Canvas rein")
        content.setStyleSheet("background: #ddd")
        content.setMinimumHeight(200)

        main_layout.addWidget(content)

        central.setLayout(main_layout)

        # StyleSheet für dunkles Design
        self.setStyleSheet("""
        QMainWindow {
            background-color: #1e1f2b;
        }   

        QLabel {
            color: white;
            font-size: 16px;
        }

        QRadioButton {
            color: white;
            font-size: 14px;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: #3a3c4f;
            border-radius: 3px;
        }

        QSlider::handle:horizontal {
            background: #4c78a8;
            width: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }

        QMenuBar {
            background: #2a2c3a;
            color: white;
        }

        QMenuBar::item:selected {
            background: #4c78a8;
        }

        QMenu {
            background: #2a2c3a;
            color: white;
        }

        QMenu::item:selected {
            background: #4c78a8;
        }

        QPushButton {
            background-color: #4c78a8;
            color: white;
            border-radius: 6px;
            padding: 6px;
        }

        QPushButton:hover {
            background-color: #5d8ec4;
        }
        """)


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()