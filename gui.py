from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QTextEdit, QSlider, QProgressBar, QComboBox, QListWidget, QRadioButton, QCheckBox, QMessageBox
from PySide6.QtCore import Qt
from torch import layout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hello World Application")

        menubar = self.menuBar()

        fileMenu = menubar.addMenu("File")
        editMenu = menubar.addMenu("Edit")  
        helpMenu = menubar.addMenu("Help")

        supMenu = fileMenu.addMenu("Submenu")
        exitAction = supMenu.addAction("Exit")
        exitAction.triggered.connect(self.close)

        aboutAction = helpMenu.addAction("About")
        aboutAction.triggered.connect(lambda: print("This is a simple PySide6 application."))

        container = QWidget()
        inner_container = QWidget()

        self.setCentralWidget(container)

        self.count = 1
        self.windows = []

        layout = QVBoxLayout(container)
        inner_layout = QHBoxLayout(inner_container)

        buttonSW = QPushButton("Open Secondary Window")
        buttonSW.clicked.connect(self.open_secondary_window)
       

        button = QPushButton("Show Message")
        button.clicked.connect(lambda: QMessageBox.information(self, "Info", "Hello World"))

        buttonY = QPushButton("Show Choices")
        buttonY.clicked.connect(self.ask_choices)

        # add a label to the main window
        label = QLabel("One")
        label.setAlignment(Qt.AlignCenter)

        button1 = QPushButton("Click Me")
        button1.clicked.connect(self.do_something)

        button2 = QPushButton("Click Me")
        button2.clicked.connect(lambda: print("Button 2 clicked!"))

        listwidget = QListWidget()
        listwidget.addItems(["Item 1", "Item 2", "Item 3"])

        listwidget.itemClicked.connect(lambda item: print("List item clicked:", item.text()))   
        listwidget.itemDoubleClicked.connect(lambda item: print("List item double-clicked:", item.text()))  

        radio1 = QRadioButton("Radio 1")
        radio2 = QRadioButton("Radio 2")
        radio3 = QRadioButton("Radio 3")

        for r in (radio1, radio2, radio3):
            r.toggled.connect(self.radio_changed)

        inner_layout.addWidget(radio1)
        inner_layout.addWidget(radio2)
        inner_layout.addWidget(radio3)

        layout.addWidget(button)
        layout.addWidget(buttonY)
        layout.addWidget(buttonSW)

        layout.addWidget(label)
        layout.addWidget(button1)
        layout.addWidget(button2)   
        layout.addWidget(listwidget)
        layout.addWidget(inner_container)

    def do_something(self):
        print("Button clicked!")

    def radio_changed(self):
        r = self.sender()
        if r.isChecked():
            print("Radio button was checked:", r.text())
     
    def ask_choices(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Choice")
        msg_box.setText("Choose an option:")

        msg_box.addButton("Python", QMessageBox.AcceptRole)
        msg_box.addButton("C++", QMessageBox.AcceptRole)
        msg_box.addButton("JavaScript", QMessageBox.AcceptRole)

        msg_box.exec()

        if msg_box.clickedButton().text() == "Python":
            print("User chose Python!")
        elif msg_box.clickedButton().text() == "C++":
            print("User chose C++!")
        elif msg_box.clickedButton().text() == "JavaScript":
            print("User chose JavaScript!")
        
    def ask_yes_no(self):
        reply = QMessageBox.question(self, "Question", "Do you like PySide6?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            print("User likes PySide6!")
        else:
            print("User does not like PySide6.")
        
    def open_secondary_window(self):
        w = SecondaryWindow(self.count)
        self.count += 1
        self.windows.append(w)  # keep a reference 
        w.show()

class SecondaryWindow(QMainWindow):
    def __init__(self, n):
        super().__init__()

        self.setWindowTitle(f"Window Number {n}")

        label = QLabel(f"Number {n}")
        label.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(label)

app = QApplication()

window = MainWindow()
window.show()

app.exec()