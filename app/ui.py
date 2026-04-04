#ui.py

from cProfile import label
import json
from pdb import pm

from PIL import Image
import io
import os
import shutil
import sys
import traceback

from PySide6.QtCore import Qt, QPoint, QSize, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QAction, QImage
from PySide6.QtWidgets import (
    QApplication, QFrame, QMainWindow, QWidget, QLabel, QFileDialog, QColorDialog, QToolBar, QVBoxLayout, QHBoxLayout, QMessageBox, QRadioButton, QSlider, QPushButton, QSizePolicy, QGroupBox, QComboBox, QInputDialog
)
from matplotlib import axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

from torch import layout

from app.data import clear_dir_safe, load_emnist_mapping, prepare_featuremaps
from app.train import BASE_DIR

from .run import run_EMINIST, load_model_class_flex
from .cnn import test_input_image

class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

# PaintArea: Widget to draw on, using mouse events to create a simple painting application.
class PaintArea(QWidget):
    def __init__(self, width=280, height=280, parent=None):
        super().__init__(parent)

        self.start_width = width
        self.start_height = height

        self.setMinimumSize(width, height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # drawing area as a pixmap
        self.canvas = QPixmap(width, height)
        self.canvas.fill(Qt.white)


        # drawing parameters
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(0, 0, 0)   # black
        self.pen_width = 3

        # StyleSheet for the drawing area
        self.setStyleSheet("""
        background: white;
        border-radius: 10px;
        border: 2px solid #3a3c4f;
        """)

    # set the pen color
    def set_pen_color(self, color: QColor):
        self.pen_color = color

    # not needed
    def set_pen_width(self, width: int):
        self.pen_width = max(1, width)

    # clear the canvas
    def clear(self):
        self.canvas.fill(Qt.GlobalColor.white)
        self.update()

    # save the canvas to a file
    def save_image(self, path: str):
        if not self.canvas.save(path):
            raise RuntimeError("Picture could not be saved.")

    # paint event to draw the canvas
    def paintEvent(self, event):
        # shows the pixmap on the widget
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.canvas)

    # mouse events to draw on the canvas
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.last_point = event.position().toPoint()

    # mouse events to draw on the canvas
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            current_point = event.position().toPoint()
            painter = QPainter(self.canvas)
            pen = QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine,
                       Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()  # draw the updated canvas on the widget

    # mouse events to draw on the canvas
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drawing:
            self.drawing = False
    
    # size hint to keep the canvas square
    def sizeHint(self):
        return QSize(self.start_width, self.start_height)

class DisplayWindow(QWidget):
    def __init__(self):
        super().__init__()

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       Display Window Setup
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       define widgets 
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        
        # Prediction Label
        self.prediction_label = QLabel("Prediction: None")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        # StyleSheet for the prediction label with a different color to make it stand out
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #7aa2f7;")

        # Layer Label
        self.layer_label = QLabel("Layer of the CNN:")
        self.layer_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        # Output Matplotlib Canvas
        self.output_canvas = MplCanvas()
        self.output_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_canvas.setMinimumSize(280, 280)

        # Processed Matplotlib Canvas
        self.processed_canvas = MplCanvas()
        self.processed_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.processed_canvas.setMinimumSize(280, 280)

        # Beispielplot
        self.processed_canvas.ax.plot([0,1,2], [0,1,0])

        # Processed Label
        # self.processed_label = QLabel("No Image")
        # self.processed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.processed_label.setMinimumSize(280, 280)
        # self.processed_label.setScaledContents(True)

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       define Layouts and Containers
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        # main layout and container for the central widget
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # model group box and layout
        output_box = QGroupBox("Output")   
        output_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;                    
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                  
        }
        """)
        output_layout = QVBoxLayout()

        # model group box and layout
        processed_box = QGroupBox("Processed Image")   
        processed_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;                    
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                  
        }
        """)
        processed_layout = QVBoxLayout()

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       add widgets to the layouts
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        # add Widgets to the output layout
        output_layout.addWidget(self.prediction_label)
        output_layout.addWidget(self.layer_label)
        output_layout.addWidget(self.output_canvas, alignment=Qt.AlignCenter)
        output_box.setLayout(output_layout)

        # add processed label to the processed layout
        processed_layout.addWidget(self.processed_canvas, alignment=Qt.AlignCenter)
        processed_box.setLayout(processed_layout)
        
        # add the model output box and processed box to the main layout
        main_layout.addWidget(output_box, 1)
        main_layout.addWidget(processed_box, 1)

        # style the main layout with spacing and margins
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       StyleSheet for the main window and widgets
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        
        self.setWindowTitle("Network Outputs")
        self.resize(400, 400)

        self.setStyleSheet("""
        QWidget {
            background-color: #1e1f2b;
            color: white;
        }
        QLabel {
            color: white;
            font-size: 16px;
        }
        """)
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #                                                       define methods for events and actions
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    # XX
    # # resize event to keep the output display square
    # def resizeEvent(self, event):
    #     # calculate the available space for the output display, considering the space taken by the prediction and layer labels
    #     available_height = self.height() - self.prediction_label.height() - self.layer_label.height() - 40
    #     available_width = self.width() - 40
    #     side = min(available_height, available_width)
    #     self.outputs_label.setFixedSize(side, side)
    #     super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       Main Window Setup
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        # variables 
        self.last_saved_path = None
        
        self.activations = None
        self.layer_names = None

        self.animation_enabled = True
        self.is_animating = False

        self.animation_delay = 500  # delay in milliseconds between animation frames

        #XX
        self.pixmap_cache = {}  # key = filename, value = QPixmap
        
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       define widgets 
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        # Display Window for CNN Outputs
        self.display_window = DisplayWindow()

        # Radio Buttons for dataset selection
        self.rb_digits = QRadioButton("digits")
        self.rb_digits.setChecked(True)  # default selection
        self.rb_letters = QRadioButton("letters")
        self.rb_balanced = QRadioButton("balanced")

        # Buttons and Dropdown for model selection 
        self.model_button = QPushButton("select new model")
        self.model_button.clicked.connect(self.select_model)
        self.model_dropdown = QComboBox()

        # Buttons and Dropdown for model structure selection 
        self.model_structure_button = QPushButton("select new model structure")
        self.model_structure_button.clicked.connect(self.select_model_structure)
        self.model_structure_dropdown = QComboBox()

        # Buttons and Dropdown for recipe selection 
        self.save_recipe_button = QPushButton("save recipe")
        self.save_recipe_button.clicked.connect(self.save_recipe)
        self.recipe_dropdown = QComboBox()
        # Buttons for recipe application
        self.apply_recipe_button = QPushButton("apply recipe")
        self.apply_recipe_button.clicked.connect(lambda _: self.apply_recipe(self.recipe_dropdown.currentText()))

        # Buttons to delete recipe
        self.delete_recipe_button = QPushButton("delete recipe")
        self.delete_recipe_button.clicked.connect(self.delete_recipe)

        # PaintArea and Layout
        self.paint_area = PaintArea(280, 280)
        self.paint_area.set_pen_width(28)  # default pen width
        
        # Run Button
        self.run_button = QPushButton("run")
        self.run_button.clicked.connect(self.run)

        # Clear Button
        self.clear_button = QPushButton("clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.clicked.connect(self.paint_area.clear)

        # Slider for Output diashow
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)  # default range, will be updated based on number of output layers
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        # slider event to update the output display when slider value changes
        #XX
        #self.display_output()  # display the initial output
        self.slider.valueChanged.connect(self.on_slider_change) # update output canvas when slider value changes

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       define menu bar
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        #MenuBar
        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File") 
        helpMenu = menubar.addMenu("Help")
        viewMenu = menubar.addMenu("View")

        exitAction = fileMenu.addAction("Exit")
        exitAction.triggered.connect(self.close)
        clear_action = fileMenu.addAction("Clear")
        clear_action.triggered.connect(self.paint_area.clear)
        run_action = fileMenu.addAction("Run")
        run_action.triggered.connect(self.run)
        testInput_action = fileMenu.addAction("Test Image")
        testInput_action.triggered.connect(self.testImage)
       
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(lambda: QMessageBox.information(self, "Info", "CNN using PyTorch to classify MNIST data."))

        openDisplayAction = viewMenu.addAction("Open Display Window")
        openDisplayAction.triggered.connect(self.open_display_window)

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       define Layouts and Containers
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        # main layout and container for the central widget
        main_layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # dataset group box and layout
        dataset_box = QGroupBox("Dataset Selection")   
        dataset_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;                    
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                  
        }
        """)
        dataset_layout = QHBoxLayout()

        # model and model_structure group box and layout
        model_box = QGroupBox("Model and Model structure Selection")   
        model_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;                    
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                  
        }
        """)
        main_model_layout = QHBoxLayout()
        model_layout = QVBoxLayout()
        model_structure_layout = QVBoxLayout()

        # recipe group box and layout
        recipe_box = QGroupBox("recipes")   
        recipe_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;                    
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                  
        }
        """)
        main_recipe_layout = QHBoxLayout()
        recipe_layout = QVBoxLayout()
        apply_recipe_layout = QVBoxLayout()

        # draw box qframe and layout
        draw_box = QFrame()
        draw_box.setStyleSheet("""
        QFrame {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;  
        }
        """)
        draw_layout = QHBoxLayout()
        button_layout = QVBoxLayout()

        # slider box group box and layout
        slider_box = QGroupBox("Layer Selection")
        slider_box.setStyleSheet("""
        QGroupBox {
            border: 2px solid #3a3c4f;
            border-radius: 10px;
            background-color: #252736;
            margin-top: 10px;
        }

        QGroupBox::title {
            color: white;
            font-size: 16px;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;                        
        }
        """)
        slider_layout = QVBoxLayout()

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       add widgets to the layouts
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        # add radio buttons to the dataset selection layout
        dataset_layout.addStretch()
        dataset_layout.addWidget(self.rb_digits)
        dataset_layout.addStretch()
        dataset_layout.addWidget(self.rb_letters)
        dataset_layout.addStretch()
        dataset_layout.addWidget(self.rb_balanced)
        dataset_layout.addStretch()
        # set the dataset selection layout to the dataset selection group box
        dataset_box.setLayout(dataset_layout)

        # add button and dropdown to the model layout
        model_layout.addStretch()
        model_layout.addWidget(self.model_button)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addStretch()
        # add button and dropdown to the model structure layout
        model_structure_layout.addStretch()
        model_structure_layout.addWidget(self.model_structure_button)
        model_structure_layout.addWidget(self.model_structure_dropdown)
        model_structure_layout.addStretch()
        # add the model layout and model structure layout to the main model layout
        main_model_layout.addLayout(model_layout)
        main_model_layout.addLayout(model_structure_layout)
        # set the main model layout to the model box
        model_box.setLayout(main_model_layout) 
        
        # add button and dropdown to the recipe layout
        recipe_layout.addStretch()
        recipe_layout.addWidget(self.save_recipe_button)
        recipe_layout.addWidget(self.recipe_dropdown)
        recipe_layout.addStretch()
        # add button to the apply recipe layout
        apply_recipe_layout.addStretch()
        apply_recipe_layout.addWidget(self.apply_recipe_button)
        apply_recipe_layout.addWidget(self.delete_recipe_button)
        apply_recipe_layout.addStretch()
        # add the recipe layout and apply recipe layout to the main recipe layout
        main_recipe_layout.addLayout(recipe_layout)
        main_recipe_layout.addLayout(apply_recipe_layout)
        # set the main model layout to the model box
        recipe_box.setLayout(main_recipe_layout) 

        # add buttons to the button layout
        button_layout.addStretch()
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        # add the paint area and buttons to the draw layout
        draw_layout.addWidget(self.paint_area, alignment=Qt.AlignCenter)
        draw_layout.addLayout(button_layout)
        # set the draw layout to the draw box
        draw_box.setLayout(draw_layout)

        # add slider to the slider layout
        slider_layout.addWidget(self.slider)
        slider_box.setLayout(slider_layout)

        # add the model selection box, draw box and slider to the main layout
        main_layout.addWidget(dataset_box)
        main_layout.addWidget(model_box)
        main_layout.addWidget(recipe_box)
        main_layout.addWidget(draw_box)
        main_layout.addWidget(slider_box)

        # style the main layout with spacing and margins

        dataset_layout.setContentsMargins(10, 5, 10, 10)
        dataset_layout.setSpacing(15)

        dataset_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        draw_layout.setSpacing(10)

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       open display window
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        self.open_display_window()  # open the display window at startup

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       StyleSheet for the main window and widgets
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        self.setWindowTitle("CNN-GUI")
        self.resize(600, 400)

        # StyleSheet for dark theme
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
        
        QPushButton#clearButton {
            background-color: #555;
        }

        QPushButton#clearButton:hover {
            background-color: #777;
        }
        """)

        #-----------------------------------------------------------------------------------------------------------------------------------------------
        #                                                       init functions
        #-----------------------------------------------------------------------------------------------------------------------------------------------

        self.scan_models()
        self.scan_structures()
        self.scan_recipes()
        self.load_last_recipe()
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #                                                       define methods for events and actions
    #-----------------------------------------------------------------------------------------------------------------------------------------------

    # MainWindow-Methods

    # save the current canvas as a 28x28 image for CNN input
    def save_image(self):
        # make sure the "input" directory exists
        os.makedirs("input", exist_ok=True)

        # save to a fixed path for simplicity
        path = os.path.join("input", "input.png")

        # original-pixmap (280x280)
        original = self.paint_area.canvas

        # convert to QImage for scaling
        img = original.toImage()

        # scale to 28x28 (CNN input size)
        small = img.scaled(
            28, 28,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # save to a fixed path for simplicity
        try:
            small.save(path)
            self.last_saved_path = path
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving image: {e}")
            print("Error saving image:", e)

    # XX
    def load_pixmap_from_file(self, path):

        if path in self.pixmap_cache:
            return self.pixmap_cache[path]

        pm = QPixmap(path)

        # label = self.display_window.outputs_label
        # side = min(label.width(), label.height())  # fit to the smaller dimension of the label

        pm = pm.scaled(
            2800, 
            2800,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.pixmap_cache[path] = pm

        return pm

    def on_slider_change(self):
        if self.is_animating:
            return  # ignore slider changes during animation
        
        index = self.slider.value() - 1
        self.show_layer(index)

    #XX
    # def display_output(self, initial=False):
    #     if initial:
    #         self.display_window.layer_label.setText("Layer of the CNN:")
    #         return

    #     index = self.slider.value() - 1
    #     self.show_layer(index)
    
    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #                                                       define methods for displaying the CNN outputs and animation
    #-----------------------------------------------------------------------------------------------------------------------------------------------

    def show_layer(self, layer_index):
        layer_name = self.layer_names[layer_index]
        x = self.activations[layer_name]

        self.display_window.layer_label.setText(f"Layer: {layer_name}")

        data, layer_type = prepare_featuremaps(x)
        #XX
        print("Prepared:", data.shape)

        if layer_type == "conv":
            self.show_featuremaps(data, animation_mode=False)
        else:
            is_last = (layer_index == len(self.layer_names) - 1)
            self.show_fc(data, show_x_labels=is_last, dataset=self.get_selected_dataset())

    def show_featuremaps(self, maps, animation_mode=True):

        fig = self.display_window.output_canvas.fig
        fig.clear()

        C = maps.shape[0]
        cols = int(np.ceil(np.sqrt(C)))
        rows = int(np.ceil(C / cols))

        axes = fig.subplots(rows, cols)

        # if there is only one feature map, axes is not an array, so we need to make it an array for consistency
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        axes = axes.flatten()

        self.images = []

        for i in range(len(axes)):
            axes[i].axis("off")

            if i < C:

                fm = maps[i]

                # sometimes the feature map has an extra dimension, we need to squeeze it to get the correct shape for imshow
                if fm.ndim != 2:
                    print("ERROR Featuremap shape:", fm.shape)
                    fm = fm.squeeze()
                # start with a blank image and then fill it in incrementally for the animation effect
                img = axes[i].imshow(
                    np.zeros_like(fm),
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
                self.images.append((img, fm))

        if not animation_mode:
            for img, fm in self.images:
                img.set_data(fm)

        self.display_window.output_canvas.draw()

    def show_fc(self, vec, show_x_labels=False, dataset=None):

        fig = self.display_window.output_canvas.fig
        fig.clear()

        ax = fig.add_subplot(111)

        x_idx = np.arange(len(vec))
        ax.bar(x_idx, vec)

        ax.set_title("Fully Connected Layer")

        if show_x_labels:
            ax.set_xlabel("Class")
            ax.set_xticks(x_idx)

            if dataset is not None:

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
    
                ax.set_xticklabels([mapping[i] for i in x_idx], fontsize=8)
            else:
                ax.set_xticklabels([str(i) for i in x_idx], fontsize=8)
        else:
            ax.set_xticks([])

        self.display_window.output_canvas.draw()
    
    # start the animation
    def start_animation(self):
        self.is_animating = True
        self.set_controls_enabled(False)

        self.current_layer = 0
        self.slider.setValue(1)

        self.animate_layer()

    # animation for the current layer, updates the display incrementally to create an animation effect
    def animate_layer(self):
        # at the end of the animation, enable controls and return
        if self.current_layer >= len(self.layer_names):
            self.is_animating = False
            self.set_controls_enabled(True)
            self.display_window.prediction_label.setText(f"Prediction: {self.current_prediction}")
            return

        # update slider to current layer
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_layer + 1)
        self.slider.blockSignals(False)

        layer_name = self.layer_names[self.current_layer]
        self.display_window.layer_label.setText(f"Layer: {layer_name}")

        # get the activations for the current layer and display them
        layer_name = self.layer_names[self.current_layer]
        x = self.activations[layer_name]

        data, layer_type = prepare_featuremaps(x)

        if layer_type == "conv":
            self.show_featuremaps(data)
            self.current_pixel = 0

            self.timer = QTimer()
            self.timer.timeout.connect(self.update_animation)
            self.timer.start(5)
        else:
            # FC → no Animation
            is_last = (self.current_layer == len(self.layer_names) - 1)
            self.show_fc(data, show_x_labels=is_last, dataset=self.get_selected_dataset())

            QTimer.singleShot(self.animation_delay, self.next_layer)
            

 
    # XX
    def update_animation_alt(self):

        for img, full in self.images:
            h, w = full.shape

            y = self.current_pixel // w
            x = self.current_pixel % w

            if y < h and x < w:
                temp = np.array(img.get_array())
                temp[y, x] = full[y, x]
                img.set_data(temp)

        self.current_pixel += 1
        self.display_window.output_canvas.draw_idle()

        max_pixels = max(fm.size for _, fm in self.images)

        if self.current_pixel >= max_pixels:
            self.timer.stop()
            self.next_layer()
    
    # animate the current layer, updates the display incrementally to create an animation effect
    def update_animation(self):

        finished = True

        for img, full in self.images:
            h, w = full.shape
            max_pixels = h * w

            if self.current_pixel < max_pixels:
                pixel = min(self.current_pixel, max_pixels - 1)
                y = pixel // w
                x = pixel % w

                #if y < h and x < w:
                temp = np.array(img.get_array())
                temp[y, x] = full[y, x]
                img.set_data(temp)

                finished = False  # important: if at least one image is not finished, we are not finished with the animation

        self.display_window.output_canvas.draw_idle()

        self.current_pixel += 1

        if finished:

            for img, full in self.images:
                img.set_data(full)

            self.display_window.output_canvas.draw()

            self.timer.stop()
            self.next_layer()

    # animate next layer
    def next_layer(self):
        self.current_layer += 1
        self.animate_layer()

    # enable or disable controls during animation to prevent user interaction that could interfere with the animation
    def set_controls_enabled(self, enabled):
        self.slider.setEnabled(enabled)
        self.run_button.setEnabled(enabled)

#-----------------------------------------------------------------------------------------------------------------------------------------------
#                                                       run the model and get the activations for the display and animation
#-----------------------------------------------------------------------------------------------------------------------------------------------

    def run(self):
        dataset = self.get_selected_dataset()
        model = self.model_dropdown.currentText()
        model_structure = self.model_structure_dropdown.currentText()

        # validate model and model structure input
        if not self.validate_inputs():
            return
        
        if not self.validate_structure(model_structure):
            return
        
        self.save_last_recipe(self.recipe_dropdown.currentText())

        #save the current canvas as an image for CNN input
        self.save_image()

        #XX
        # original-pixmap (280x280)
        #original = self.paint_area.canvas
        #data = np.asarray(original)
        data = np.zeros((26,26))  # Beispiel
        self.img = self.display_window.processed_canvas.ax.imshow(data, cmap="gray")

        prediction = None

        if self.last_saved_path == None:
            QMessageBox.warning(self, "Warning", "No input to run.")
            return
        
        print("Running model...")
        prediction, activations, layer_names = run_EMINIST(dataset, model, model_structure)
        self.activations = activations
        self.layer_names = layer_names
        self.slider.setRange(1, len(self.layer_names))
        self.slider.setValue(1)
        print("Prediction:", prediction)

        if self.animation_enabled:
            self.start_animation()
        else:
            self.show_layer(0)

        self.current_prediction = prediction

        

    #XX
    def testImage(self):
        self.save_image()
        test_input_image("input/input.png")
    
    def open_display_window(self):
        self.display_window.show()
        self.display_window.raise_()   # bring to front
        self.display_window.activateWindow()

    #XX
    def update_plot(self):
        self.img.set_data(self.current_frame)
        self.canvas.draw()

    #-----------------------------------------------------------------------------------------------------------------------------------------------
    #                                                       methods for model, model structure and recipe management
    #-----------------------------------------------------------------------------------------------------------------------------------------------

    def select_model(self): 
        file_path, _ = QFileDialog.getOpenFileName(self, "select new model", "", "PyTorch models (*.pth *.pt)")
        if file_path:
            file_name = os.path.basename(file_path)

            target_path = os.path.join("models", file_name)
            shutil.copy(file_path, target_path)
            print(f"Model copied to {target_path}")

            self.model_dropdown.addItem(file_name)

    def scan_models(self):
        self.model_dropdown.clear()

        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        for file in os.listdir(model_dir):
            if file.endswith((".pth", ".pt")):
                self.model_dropdown.addItem(file)

    def select_model_structure(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "select new model", "", "PyTorch models (*.py)")
        if file_path:
            file_name = os.path.basename(file_path)

            target_path = os.path.join("model_structures", file_name)
            shutil.copy(file_path, target_path)
            print(f"Model Structure copied to {target_path}")

            module_name = file_name.replace(".py", "")
            self.model_structure_dropdown.addItem(module_name)

    def scan_structures(self):
        self.model_structure_dropdown.clear()

        structure_dir = "model_structures"

        for file in os.listdir(structure_dir):
            if file.endswith(".py") and file != "__init__.py":
                name = file.replace(".py", "")
                self.model_structure_dropdown.addItem(name)
    
    def get_selected_dataset(self):
        if self.rb_digits.isChecked():
            return "digits"
        elif self.rb_letters.isChecked():
            return "letters"
        return "balanced"

    def load_recipes(self):
        if not os.path.exists("recipes.json"):
            return {}

        with open("recipes.json", "r") as f:
            return json.load(f)

    def scan_recipes(self):
        self.recipe_dropdown.clear()
        data = self.load_recipes()

        for name in data.keys():
            self.recipe_dropdown.addItem(name)

    def save_recipe(self):
        name, ok = QInputDialog.getText(self, "save recipe", "name:")
        if not ok or not name:
            return

        recipe = {
            "dataset": self.get_selected_dataset(),
            "model": self.model_dropdown.currentText(),
            "structure": self.model_structure_dropdown.currentText()
        }

        data = self.load_recipes()
        data[name] = recipe

        with open("recipes.json", "w") as f:
            json.dump(data, f, indent=4)

        self.recipe_dropdown.addItem(name)

    def apply_recipe(self, name):
        print(name)
        data = self.load_recipes()
        recipe = data.get(name)

        if not recipe:
            return

        # set dataset
        if recipe["dataset"] == "digits":
            self.rb_digits.setChecked(True)
        elif recipe["dataset"] == "letters":
            self.rb_letters.setChecked(True)
        else:
            self.rb_balanced.setChecked(True)

        # set dropdowns
        self.model_dropdown.setCurrentText(recipe["model"])
        self.model_structure_dropdown.setCurrentText(recipe["structure"])

    def delete_recipe(self):
        name = self.recipe_dropdown.currentText()
        if not name:
            return

        data = self.load_recipes()

        if name in data:
            reply = QMessageBox.question(
                self,
                "Confirm",
                f"delete recipe '{name}':",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

            del data[name]

            with open("recipes.json", "w") as f:
                json.dump(data, f, indent=4)

            self.recipe_dropdown.removeItem(self.recipe_dropdown.currentIndex())

    def save_last_recipe(self, name):
        with open("last_recipe.txt", "w") as f:
            f.write(name)

    def load_last_recipe(self):
        if not os.path.exists("last_recipe.txt"):
            return

        with open("last_recipe.txt", "r") as f:
            name = f.read().strip()

        self.apply_recipe(name)

    def validate_inputs(self):
        if self.model_dropdown.currentText() == "":
            QMessageBox.warning(self, "Error", "No model selected")
            return False

        if self.model_structure_dropdown.currentText() == "":
            QMessageBox.warning(self, "Error", "No structure selected")
            return False

        return True
    
    
    def validate_structure(self, structure_name):
        try:
            load_model_class_flex(structure_name)
            return True
        except Exception as e:
            print("Error loading module:")
            traceback.print_exc()  # shows complete error message
            QMessageBox.warning(self, "Error", f"Error loading model structure: {e}")
        return False
