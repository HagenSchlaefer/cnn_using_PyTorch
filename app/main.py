# main.py
import sys
from PySide6.QtWidgets import QApplication
from .ui import MainWindow

def main():
    # if there is already a QApplication instance (e.g. if we are running in an interactive environment), use it, otherwise create a new one
    app = QApplication.instance()
    created_here = False
    if app is None:
        app = QApplication(sys.argv)
        created_here = True

    win = MainWindow()
    win.show()

    # if we created the QApplication, we need to start the event loop, otherwise we assume it's already running
    if created_here:
        sys.exit(app.exec())

if __name__ == "__main__":
    main()