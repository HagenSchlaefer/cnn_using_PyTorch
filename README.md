# cnn_using_PyTorch
cnn using PyTorch to classify MNIST data

---------------------------------------------------------------------------------------------------------
repository Klonen
git clone https://github.com/HagenSchlaefer/cnn_using_PyTorch.git
---------------------------------------------------------------------------------------------------------

!!Python 3.12 instaliern!!
---------------------------------------------------------------------------------------------------------
Installation von pyenv‑win zum verwalten von Python
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" | Invoke-Expression

Python 3.12 installieren
pyenv install 3.12.2

Python‑Version global setzen
pyenv global 3.12.2

Python‑Version prüfen
python --version
---------------------------------------------------------------------------------------------------------
New Poetry als Virtuelle Umgebeung

poetry instaliern
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

eventuell poetry Pfad angben ("C:\Users\<NAME>\AppData\Roaming\Python\Scripts")
Umgebungsvariablen → PATH

Virtuelle Poetry umgebung instalieren
poetry install

Abhängigkeit hinzufügen
poetry add numpy

start poetry
poetry env activate

Sobald du das Plugin installiern
poetry self add poetry-plugin-shell

Poetry Umgebung starten
poetry shell

((cnn-using-pytorch-py3.12) PS C:\...)
---------------------------------------------------------------------------------------------------------