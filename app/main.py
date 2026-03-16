# main.py
import sys
from PySide6.QtWidgets import QApplication
from .ui import MainWindow

def main():
    app = QApplication.instance()

    if app is None:
        app = QApplication(sys.argv)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()