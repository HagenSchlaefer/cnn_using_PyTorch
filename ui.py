
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor, QAction, QImage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QFileDialog, QColorDialog, QToolBar, QVBoxLayout, QHBoxLayout, QMessageBox, QRadioButton, QSlider
)

from run import run_EMINIST_balanced

import sys
import os

# PaintArea: Widget to draw on, using mouse events to create a simple painting application.
class PaintArea(QWidget):
    def __init__(self, width=280, height=280, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StaticContents)
        self.setFixedSize(width, height)

        # drawing area as a pixmap
        self.canvas = QPixmap(width, height)
        self.canvas.fill(Qt.white)


        # drawing parameters
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(0, 0, 0)   # black
        self.pen_width = 3

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
            raise RuntimeError("Bild konnte nicht gespeichert werden.")

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CNN-GUI")

        self.last_saved_path = None
        self.selected_model = None
        
        # Container
        container = QWidget()
        inner_container_1 = QWidget()
        inner_container_2 = QWidget()
        inner_container_3 = QWidget()
        self.setCentralWidget(container)

        # Layouts
        layout = QVBoxLayout(container)
        
        inner_layout_1 = QHBoxLayout(inner_container_1)
        inner_layout_2 = QHBoxLayout(inner_container_2)
        inner_layout_3 = QHBoxLayout(inner_container_3)

        # Radio Buttons for model selection
        radio1 = QRadioButton("MINIST-CNN") # maybe later: "EMNIST-digits-CNN"
        radio1.setChecked(True)  # default selection
        radio2 = QRadioButton("EMNIST-letters-CNN")
        radio3 = QRadioButton("EMNIST-balanced-CNN")
        for r in (radio1, radio2, radio3):
            r.toggled.connect(self.radio_changed)

        # PaintArea and Layout
        self.paint_area = PaintArea(280, 280)

        # Outputs Label
        self.outputs_label = QLabel()
        self.outputs_label.setFixedSize(280, 280)
        self.outputs_label.setAlignment(Qt.AlignCenter)

        # Slider for Output diashow
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)  # default range, will be updated based on number of output images
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.display_output()  # display the initial output
        self.slider.valueChanged.connect(self.display_output)  # update output display when slider value changes
        

        #MenuBar
        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File") 
        helpMenu = menubar.addMenu("Help")

        exitAction = fileMenu.addAction("Exit")
        exitAction.triggered.connect(self.close)
        run_action = fileMenu.addAction("Run")
        run_action.triggered.connect(self.run)
        clear_action = fileMenu.addAction("Clear")
        clear_action.triggered.connect(self.paint_area.clear)
        
        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(lambda: QMessageBox.information(self, "Info", "CNN using PyTorch to classify MNIST data."))

        # Prediction Label
        self.prediction_label = QLabel("Prediction: None")
        self.prediction_label.setAlignment(Qt.AlignCenter)

        # Toolbar
        toolbar = QToolBar("Toolbar")
        self.addToolBar(toolbar)

        # Toolbar-Action: select color
        color_action = QAction("Color", self)
        color_action.triggered.connect(self.choose_color)
        toolbar.addAction(color_action)

        #not needed
        # Toolbar-Action: select pen width
        # self.width_spin = QSpinBox()
        # self.width_spin.setRange(1, 50)
        # self.width_spin.setValue(3)
        # self.width_spin.valueChanged.connect(self.paint_area.set_pen_width)
        # toolbar.addWidget(self.width_spin)
        self.paint_area.set_pen_width(32)  # default pen width

        # Toolbar-Action: Clear canvas
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self.paint_area.clear)
        toolbar.addAction(clear_action)

        # Toolbar-Action: Save canvas as image
        run_action = QAction("Run", self)
        run_action.triggered.connect(self.run)
        toolbar.addAction(run_action)

        # add widgets to the layouts
        inner_layout_1.addWidget(radio1)
        inner_layout_1.addWidget(radio2)
        inner_layout_1.addWidget(radio3)

        inner_layout_2.addWidget(self.prediction_label)

        inner_layout_3.addWidget(self.paint_area)
        inner_layout_3.addWidget(self.outputs_label)

        layout.addWidget(inner_container_1)
        layout.addWidget(inner_container_2)
        layout.addWidget(inner_container_3)
        layout.addWidget(self.slider)
        
        self.setFixedSize(600, 400)

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
            QMessageBox.critical(self, "Error", f"Fehler beim Speichern: {e}")
            print("Fehler beim Speichern:", e)

    def display_output(self):
        os.makedirs("outputs", exist_ok=True)

        #number of output images in the "outputs" directory to set slider range
        num_outputs = len(os.listdir("outputs"))
        self.slider.setRange(1, num_outputs)  #set range based on number of output images

        for name in os.listdir("outputs"):
            if name.endswith(".png"):
                output_path = os.path.join("outputs", name)
                identifier = name.split("_")[0]  # get the name without extension
                if identifier == str(self.slider.value()):  # show the image corresponding to the slider value
                    pm = QPixmap(output_path)
                    self.outputs_label.setPixmap(pm)
                    break


    def run(self):
        #save the current canvas as an image for CNN input
        self.save_image()

        prediction = None

        if self.last_saved_path == None:
            QMessageBox.warning(self, "Warning", "No input to run.")
            return
        
        if self.selected_model == "MINIST-CNN":
            print("Running MINIST-CNN...")
            #run_MINIST()
        elif self.selected_model == "EMNIST-letters-CNN":
            print("Running EMNIST-letters-CNN...")
            #run_EMINIST_letters()
        elif self.selected_model == "EMNIST-balanced-CNN":
            print("Running EMNIST-balanced-CNN...")
            prediction = run_EMINIST_balanced()
            print("Prediction:", prediction)

        self.prediction_label.setText(f"Prediction: {prediction}")
        self.display_output()



def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
