# How to Select Python Interpreter

## In VS Code / Cursor

### Method 1: Command Palette (Recommended)
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: `Python: Select Interpreter`
3. Choose from the list of available interpreters
4. If your Python isn't listed, click "Enter interpreter path..." and browse to your Python executable

### Method 2: Status Bar
1. Look at the bottom-right corner of VS Code/Cursor
2. Click on the Python version shown (e.g., "Python 3.x.x")
3. Select your desired interpreter from the list

### Method 3: Settings
1. Press `Ctrl+,` to open Settings
2. Search for: `python.defaultInterpreterPath`
3. Enter the full path to your Python executable, e.g.:
   - `C:\Python313\python.exe`
   - `C:\Users\YourName\AppData\Local\Programs\Python\Python313\python.exe`
   - `C:\Program Files\Python313\python.exe`

## Finding Your Python Installation

### If Python is Installed:
Run this in PowerShell to find Python:
```powershell
Get-Command python* | Select-Object Name, Source
```

Or check common locations:
- `C:\Python*\python.exe`
- `C:\Program Files\Python*\python.exe`
- `C:\Users\YourName\AppData\Local\Programs\Python\Python*\python.exe`
- `C:\Users\YourName\anaconda3\python.exe` (if using Anaconda)
- `C:\Users\YourName\miniconda3\python.exe` (if using Miniconda)

### If Python is NOT Installed:
1. Download from: https://www.python.org/downloads/
2. **IMPORTANT**: Check "Add Python to PATH" during installation
3. Restart VS Code/Cursor after installation
4. Then use Method 1 or 2 above to select it

## Verify Your Selection

After selecting the interpreter, open a terminal in VS Code/Cursor (`Ctrl+`` or Terminal → New Terminal) and run:
```bash
python --version
```

This should show the Python version you selected.

## For This Project

Once you've selected the interpreter:
1. Install dependencies: `pip install -r requirements.txt`
2. Run the project: `python run_grammar_scoring.py`

