from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QSlider, QProgressBar, QComboBox, QListWidget, QRadioButton, QCheckBox
from PySide6.QtCore import Qt
from torch import layout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hello World Application")

        container = QWidget()
        inner_container = QWidget()

        self.setCentralWidget(container)

        layout = QVBoxLayout(container)
        inner_layout = QHBoxLayout(inner_container)

        # add a label to the main window
        label = QLabel("One")
        label.setAlignment(Qt.AlignCenter)

        button = QPushButton("Click Me")

        line_edit = QLineEdit()
        text_edit = QTextEdit()

        combobox = QComboBox()
        combobox.addItems(["Option 1", "Option 2", "Option 3"]) 

        listwidget = QListWidget()
        listwidget.addItems(["Item 1", "Item 2", "Item 3"])


        checkbox1 = QCheckBox("Checkbox 1")
        checkbox2 = QCheckBox("Checkbox 2")
        checkbox3 = QCheckBox("Checkbox 3")

        inner_layout.addWidget(checkbox1)
        inner_layout.addWidget(checkbox2)
        inner_layout.addWidget(checkbox3)

        radio1 = QRadioButton("Radio 1")
        radio2 = QRadioButton("Radio 2")
        radio3 = QRadioButton("Radio 3")

        inner_layout.addWidget(radio1)
        inner_layout.addWidget(radio2)
        inner_layout.addWidget(radio3)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)

        layout.addWidget(label)
        layout.addWidget(button)
        layout.addWidget(line_edit)
        layout.addWidget(text_edit)
        layout.addWidget(combobox)
        layout.addWidget(listwidget)
        layout.addWidget(inner_container)
        layout.addWidget(slider)

     
        
  
        
app = QApplication()

window = MainWindow()
window.show()

app.exec()