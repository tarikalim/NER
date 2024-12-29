#!/bin/bash

set -e

echo "[INFO] Creating Python virtual environment..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

$PYTHON -m venv venv

echo "[INFO] Virtual environment created."

echo "[INFO] Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "[INFO] Virtual environment activated."

echo "[INFO] Installing requirements from requirements.txt..."
if pip install --upgrade pip && pip install -r requirements.txt; then
    echo "[INFO] Requirements installed successfully."
else
    echo "[ERROR] Failed to install requirements. Exiting..."
    exit 1
fi

echo "[INFO] Starting training process with train.py..."
if python ner/train.py; then
    echo "[INFO] Training completed successfully."
else
    echo "[ERROR] Training failed. Exiting..."
    exit 1
fi

echo "[INFO] Running pipeline.py to process data..."
if python ner/pipeline.py; then
    echo "[INFO] Pipeline execution completed successfully."
else
    echo "[ERROR] Pipeline execution failed. Exiting..."
    exit 1
fi

echo "[INFO] All tasks completed successfully! Output is ready."