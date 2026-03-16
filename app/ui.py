#ui.py

from cProfile import label
from pdb import pm

from PIL import Image
import io
import os
import sys

from PySide6.QtCore import Qt, QPoint, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QAction, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QColorDialog, QToolBar, QVBoxLayout, QHBoxLayout, QMessageBox, QRadioButton, QSlider, QPushButton, QSizePolicy
)

from app.data import clear_dir_safe

from .run import run_EMINIST_balanced, run_EMINIST_letters, run_MNIST
from .cnn import test_input_image


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

        self.setWindowTitle("Network Outputs")
        self.resize(400, 400)

        container = QWidget()
        layout = QVBoxLayout(container)

        # Prediction Label
        self.prediction_label = QLabel("Prediction: None")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        # StyleSheet for the prediction label with a different color to make it stand out
        self.prediction_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #7aa2f7;")

        # Layer Label
        self.layer_label = QLabel("Layer of the CNN:")
        self.layer_label.setAlignment(Qt.AlignCenter)
        self.layer_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        # Outputs Label
        self.outputs_label = QLabel("Prediction: None")
        self.outputs_label.setAlignment(Qt.AlignCenter)
        self.outputs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.outputs_label.setScaledContents(True)

        layout.addStretch(1)
        layout.addWidget(self.prediction_label)
        layout.addWidget(self.layer_label)
        layout.addWidget(self.outputs_label, stretch=1, alignment=Qt.AlignCenter)
        layout.addStretch(1)

        self.setLayout(layout)

        self.setStyleSheet("""
        QWidget {
            background-color: #1e1f2b;
            color: white;
        }
        QLabel {
            color: white;
        }
        """)
    
    # resize event to keep the output display square
    def resizeEvent(self, event):
        # calculate the available space for the output display, considering the space taken by the prediction and layer labels
        available_height = self.height() - self.prediction_label.height() - self.layer_label.height() - 40
        available_width = self.width() - 40
        side = min(available_height, available_width)
        self.outputs_label.setFixedSize(side, side)
        super().resizeEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN-GUI")

        # display window for CNN outputs
        self.display_window = DisplayWindow()

        # variables to keep track of the last saved input image and the selected model and a cache for loaded pixmaps to avoid reloading from disk
        self.last_saved_path = None
        self.selected_model = "MINIST-CNN"  # default model
        self.pixmap_cache = {}  # key = filename, value = QPixmap
        
        # Container
        container = QWidget()
        inner_container_1 = QWidget()

        self.setCentralWidget(container)
        self.open_display_window()  # open the display window at startup

        # Layouts
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)

        inner_layout_1 = QHBoxLayout(inner_container_1)
        inner_layout_1.setAlignment(Qt.AlignCenter)

        model_label = QLabel("Select Model:")
        model_label.setStyleSheet("font-size: 14px; font-weight: bold;")

        # Radio Buttons for model selection
        radio1 = QRadioButton("MINIST-CNN") # maybe later: "EMNIST-digits-CNN"
        radio1.setChecked(True)  # default selection
        radio2 = QRadioButton("EMNIST-letters-CNN")
        radio3 = QRadioButton("EMNIST-balanced-CNN")
        for r in (radio1, radio2, radio3):
            r.toggled.connect(self.radio_changed)

        # PaintArea and Layout
        self.paint_area = PaintArea(280, 280)
        self.paint_area.set_pen_width(28)  # default pen width
        
        # Slider for Output diashow
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)  # default range, will be updated based on number of output images
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        # StyleSheet for the slider with a custom design
        # self.slider.setStyleSheet("""
        # QSlider::groove:horizontal {
        #     background: #3a3c4f;
        #     height: 6px;
        #     border-radius: 3px;
        # }

        # QSlider::handle:horizontal {
        #     background: #7aa2f7;
        #     width: 16px;
        #     margin: -6px 0;
        #     border-radius: 8px;
        # }
        # """)

        # slider event to update the output display when slider value changes
        self.display_output(initial=True)  # display the initial output
        self.slider.valueChanged.connect(lambda _: self.display_output(initial=False)) # update output display when slider value changes


        #MenuBar
        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File") 
        helpMenu = menubar.addMenu("Help")

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

        # add widgets to the layouts
        inner_layout_1.addStretch()
        inner_layout_1.addWidget(radio1)
        inner_layout_1.addWidget(radio2)
        inner_layout_1.addWidget(radio3)
        inner_layout_1.addStretch()

        layout.addWidget(model_label)
        layout.addWidget(inner_container_1)
        layout.addWidget(self.paint_area, alignment=Qt.AlignCenter)
        layout.addWidget(self.slider)
         
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
        """)

    # MainWindow-Methods
    def radio_changed(self):
        r = self.sender()
        if r.isChecked():
            self.selected_model = r.text()
            print("Selected model:", self.selected_model)

    def choose_color(self):
        color = QColorDialog.getColor(initial=self.paint_area.pen_color, parent=self, title="select color")
        if color.isValid():
            self.paint_area.set_pen_color(color)

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

    def load_pixmap_from_file(self, path):

        if path in self.pixmap_cache:
            return self.pixmap_cache[path]

        pm = QPixmap(path)

        # label = self.display_window.outputs_label
        # side = min(label.width(), label.height())  # fit to the smaller dimension of the label

        pm = pm.scaled(
            280, 
            280,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.pixmap_cache[path] = pm

        return pm

    def display_output(self, initial=False):
        os.makedirs("outputs", exist_ok=True)

        if initial:
            # display a default image or message when there are no outputs yet
            self.display_window.outputs_label.setText("No outputs to display yet.")
            self.display_window.layer_label.setText("Layer of the CNN:")
            return
        # find all output images and update slider range
        png_files = [f for f in os.listdir("outputs") if f.endswith(".png")]
        num_outputs = len(png_files)
        self.slider.setRange(1, max(1, num_outputs))

        for name in png_files:
            identifier = name.split("_")[0]
            if identifier == str(self.slider.value()):
                output_path = os.path.join("outputs", name)
                pm = self.load_pixmap_from_file(output_path)

                self.display_window.outputs_label.setPixmap(pm)
                self.display_window.outputs_label.show()

                self.display_window.layer_label.setText(f"Layer of the CNN: {name.split('_')[1].split('.')[0]}")  # update layer label
                break    

    def run(self):
        #save the current canvas as an image for CNN input
        self.save_image()
        
        # clear old outputs and pixmap cache
        self.display_window.outputs_label.clear()
        self.display_window.outputs_label.setPixmap(QPixmap())
        self.pixmap_cache.clear()
        QApplication.processEvents()  # Ressourcen freigeben

        # delete old outputs and create new directory
        clear_dir_safe("outputs")

        prediction = None

        if self.last_saved_path == None:
            QMessageBox.warning(self, "Warning", "No input to run.")
            return
        
        if self.selected_model == "MINIST-CNN":
            print("Running MINIST-CNN...")
            prediction = run_MNIST()
            print("Prediction:", prediction)

        elif self.selected_model == "EMNIST-letters-CNN":
            print("Running EMNIST-letters-CNN...")
            prediction = run_EMINIST_letters()
            print("Prediction:", prediction)

        elif self.selected_model == "EMNIST-balanced-CNN":
            print("Running EMNIST-balanced-CNN...")
            prediction = run_EMINIST_balanced()
            print("Prediction:", prediction)

        self.display_window.prediction_label.setText(f"Prediction: {prediction}")

        self.slider.setValue(1)
        self.display_output(initial=False)

    def testImage(self):
        self.save_image()
        test_input_image("input/input.png")
    
    def open_display_window(self):
        self.display_window.show()
        self.display_window.raise_()   # bring to front
        self.display_window.activateWindow()

