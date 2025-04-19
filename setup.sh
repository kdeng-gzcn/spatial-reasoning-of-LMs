#!/bin/bash

# Create a Conda environment
echo "Creating Conda environment..."
conda create -n spatial_reasoning_env python=3.10 -y

# Activate the Conda environment
echo "Activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatial_reasoning_env

# Install required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Conda environment setup complete."