#!/bin/bash

echo "=============================================="
echo "ML Project 2025 - conda env setup"
echo "=============================================="

# Environment name
ENV_NAME="ml-project-2025"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

# Create conda environment
echo ""
echo "Creating conda environment: ${ENV_NAME}"
echo "Python version: 3.11"
conda create -n ${ENV_NAME} python=3.11 -y

if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment"
    exit 1
fi

echo ""
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install requirements"
    exit 1
fi

# Install Jupyter kernel
echo ""
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=${ENV_NAME} --display-name "Python (${ENV_NAME})"

echo ""
echo "=============================================="
echo "Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""

