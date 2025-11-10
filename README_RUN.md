# How to Run the Grammar Scoring Engine

## Option 1: Run the Python Script (Recommended)

1. Make sure Python 3.8+ is installed with required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:
   ```bash
   python run_grammar_scoring.py
   ```
   
   Or on Windows:
   ```bash
   py run_grammar_scoring.py
   ```
   
   Or double-click `run_project.bat`

## Option 2: Run the Jupyter Notebook

1. Install Jupyter if not already installed:
   ```bash
   pip install jupyter
   ```

2. Open and run the notebook:
   ```bash
   jupyter notebook grammar_scoring_engine.ipynb
   ```

3. Run all cells (Cell → Run All)

## Output

The script will:
- Extract features from all audio files (this may take 10-30 minutes)
- Train the models
- Display RMSE and Pearson Correlation on training data
- Generate `submission.csv` with test predictions

## Note

If Python is not found, please:
1. Install Python from https://www.python.org/downloads/
2. Make sure to check "Add Python to PATH" during installation
3. Install required packages: `pip install -r requirements.txt`

