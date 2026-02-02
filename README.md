# cnn_using_PyTorch
cnn using PyTorch to classify MNIST data

## repository Klonen
```powershell
git clone https://github.com/HagenSchlaefer/cnn_using_PyTorch.git
```

### Python 3.12 instaliern!!
Installation von pyenv‑win zum verwalten von Python
```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" | Invoke-Expression
```

Python 3.12 installieren und danach Python‑Version global setzen
```powershell
pyenv install 3.12.2
pyenv global 3.12.2
python --version
```

### New Poetry als Virtuelle Umgebeung
poetry instaliern
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```
eventuell poetry Pfad unter Umgebungsvariablen → PATH angeben ("C:\Users\<NAME>\AppData\Roaming\Python\Scripts")

Virtuelle Poetry umgebung instalieren
```powershell
poetry install
```

Poetry shell Plugin installiern
```powershell
poetry self add poetry-plugin-shell
```

Poetry Umgebung starten
```powershell
poetry shell
```
----------------------------------------------------------------------------------------
### nicht wichtig für die instalation

Abhängigkeit hinzufügen (nur wenn neue Abhängigkeit verwändet werden)
```powershell
poetry add numpy
```

start poetry
```powershell
poetry env activate
```
