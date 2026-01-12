#!/bin/bash

# Script to run GPU tests for Datarax
# Usage: bash run_gpu_tests.sh

set -e

# Check if GPU is available
echo "Checking for GPU availability..."
python check_gpu.py

# Ensure the environment is properly set up
echo "Setting up the environment for GPU tests..."
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 # Limit memory usage to avoid OOM errors

# Run the GPU tests
echo "Running GPU tests..."
python run_gpu_tests.py

echo "GPU test run complete!"
