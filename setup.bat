@echo off

title Console

if not exist ".venv" (
    echo Setting up virtual environment...
    python -m venv .venv
)

call .venv/Scripts/activate
cls

echo Loading dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -U

echo Installing Playwright browsers...
python -m playwright install chromium

echo Dependencies loaded!

echo Starting program...
cls
python main.py

pause > nul
deactivate
