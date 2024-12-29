#!/bin/bash

set -e

echo "Please enter your CUDA version (e.g., 121, 118, cpu):"
read cuda_version

if [[ "$cuda_version" == "cpu" ]]; then
    torch_command="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
else
    torch_command="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu$cuda_version"
fi

echo "[INFO] Creating Python virtual environment..."
python3 -m venv venv

echo "[INFO] Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "[INFO] Installing PyTorch and dependencies..."
eval "$torch_command"
pip install -r requirements.txt

echo "[INFO] Starting training process..."
if python ner/train.py; then
    echo "[INFO] Training completed successfully!"
else
    echo "[ERROR] Training failed. Exiting..."
    exit 1
fi

echo "[INFO] Running pipeline..."
if python ner/pipeline.py; then
    echo "[INFO] Pipeline execution completed successfully!"
else
    echo "[ERROR] Pipeline execution failed. Exiting..."
    exit 1
fi

echo "[INFO] All tasks completed successfully!"
