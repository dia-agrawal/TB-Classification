#!/bin/bash

echo "Installing dependencies for inference_evaluation.py..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip first."
    exit 1
fi

# Install dependencies
echo "Installing Python packages..."
pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
    echo "You can now run: python inference_evaluation.py --help"
else
    echo "Error: Failed to install some dependencies."
    echo "Please check the error messages above and try again."
    exit 1
fi 