#!/bin/bash

if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
clear

echo "Loading dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt -U

echo "Installing Playwright browsers..."
python3 -m playwright install chromium

echo "Dependencies loaded!"

echo "Starting program..."
clear
python3 main.py

read -p "Press any key to continue..." -n1 -s
deactivate