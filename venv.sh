#!/bin/bash
# Create and activate a Python 3.10 virtual environment, then install dependencies

# Set Python executable (assumes python3.10 is in PATH)
PYTHON_EXEC=python3.10

# Check if Python 3.10 is installed
if ! command -v $PYTHON_EXEC &> /dev/null; then
    echo "Python 3.10 is not installed or not in PATH. Please install Python 3.10 and try again."
    exit 1
fi

# Create the virtual environment
$PYTHON_EXEC -m venv tf-GPU
if [ $? -ne 0 ]; then
    echo "Failed to create the virtual environment."
    exit 1
fi

# Activate the virtual environment
source tf-GPU/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install required dependencies
python -m pip install tensorflow==2.10.0 tensorflow-io[tensorflow] numpy==1.25.2 scikit-learn==1.3.0 pandas==2.0.3 matplotlib==3.7.2 scikit-image==0.21.0 opencv-python ipykernel pillow==10.0.1

echo "Virtual environment setup complete."