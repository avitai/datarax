#!/bin/bash

# Script to set up a Python 3.11 virtual environment with JAX GPU (CUDA 12) support
# Usage: bash scripts/setup_jax_gpu_env.sh
# This script is idempotent and can be run multiple times safely.
#
# NOTE: This is a legacy script. For most use cases, prefer using:
#   ./setup.sh && source activate.sh
# which provides better CUDA configuration and cross-platform support.

set -e

echo "⚠️  NOTE: This is a legacy setup script."
echo "   For recommended setup, use: ./setup.sh && source activate.sh"
echo ""

VENV_DIR=".venv"
PYTHON_BIN="python3.11"

# 1. Ensure Python 3.11 is installed
if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "Python 3.11 is not installed. Installing..."
    sudo apt update && sudo apt install -y python3.11 python3.11-venv
fi

# 2. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    $PYTHON_BIN -m venv $VENV_DIR
fi

# 3. Activate the virtual environment
source $VENV_DIR/bin/activate

# 4. Upgrade pip and install uv if not present
python -m pip install --upgrade pip
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 5. Install project dependencies (edit as needed for your project)
uv pip install -e ".[dev,test,tfds]"

# 6. Uninstall any existing jax/jaxlib to avoid conflicts
pip uninstall -y jax jaxlib jax-cuda12-pjrt jax-cuda12-plugin || true

# 7. Install JAX with CUDA 12 support
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 8. Test GPU availability
python -c "import jax; print('JAX devices:', jax.devices()); print('GPU devices:', jax.devices('gpu'))"

printf "\nSetup complete! Activate your environment with:\n"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or for the recommended activation (if activate.sh exists):"
echo "  source activate.sh"
echo ""
echo "Verify GPU usage with:"
echo "  python -c 'import jax; print(jax.devices())'"
