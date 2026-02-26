# main.py
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader

app = QApplication()

loader = QUiLoader()

file = QFile("mainwindow.ui")
file.open(QFile.ReadOnly)

window = loader.load(file)
file.close()

window.pushButton.clicked.connect(lambda: QMessageBox.information(window, "Message", window.label.text()))

window.show()
app.exec()
