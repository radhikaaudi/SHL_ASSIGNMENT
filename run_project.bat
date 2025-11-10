@echo off
echo ========================================
echo Grammar Scoring Engine - Runner
echo ========================================
echo.

REM Try different Python commands
where python >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python via 'python' command
    python run_grammar_scoring.py
    goto :end
)

where python3 >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python via 'python3' command
    python3 run_grammar_scoring.py
    goto :end
)

where py >nul 2>&1
if %errorlevel% == 0 (
    echo Found Python via 'py' launcher
    py run_grammar_scoring.py
    goto :end
)

echo.
echo ERROR: Python not found in PATH
echo.
echo Please install Python 3.8+ and add it to your PATH, or run:
echo   python run_grammar_scoring.py
echo.
echo Alternatively, open the Jupyter notebook:
echo   jupyter notebook grammar_scoring_engine.ipynb
echo.

:end
pause

