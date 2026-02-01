# main.py
import sys
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, Qt
from PySide6.QtGui import QPixmap


class MainWindow:
    def __init__(self):
        loader = QUiLoader()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(base_dir, "mainwindow.ui")

        ui_file = QFile(ui_path)
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError("Could not open UI file")

        self.window = loader.load(ui_file)
        ui_file.close()

        if self.window is None:
            raise RuntimeError("UI loading failed")

        # Referenzen auf Widgets
        self.output_label = self.window.findChild(type(self.window.output), "output")

        # Ursprüngliches Pixmap merken
        self.original_pixmap = self.output_label.pixmap()

        # resizeEvent überschreiben
        self.window.resizeEvent = self.on_resize

        self.window.setMinimumSize(910, 389)
        self.output_label.setAlignment(Qt.AlignCenter)


        self.window.show()

    def on_resize(self, event):
        """Skaliert das Output-Bild proportional beim Resize"""
        if self.original_pixmap:
            scaled = self.original_pixmap.scaled(
                self.output_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.output_label.setPixmap(scaled)

        # Wichtig: Original-Event weiterreichen
        event.accept()



app = QApplication(sys.argv)
mw = MainWindow()
sys.exit(app.exec())
