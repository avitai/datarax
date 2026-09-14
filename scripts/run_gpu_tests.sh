#!/bin/bash

# Script to run GPU tests for Datarax
# Usage: bash scripts/run_gpu_tests.sh (from project root)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo -e "${BLUE}🎮 Datarax GPU Test Runner${NC}"
echo "================================"

# Load the environment setup.sh generated (backend file, .env and .env.local)
if [ -f "activate.sh" ] && [ -d ".venv" ]; then
    echo -e "${GREEN}✅ Activating the project environment${NC}"
    source ./activate.sh
else
    echo -e "${YELLOW}⚠️  No environment found - run ./setup.sh --backend cuda12 first${NC}"
fi

# Test runs stay on the CPU unless they ask for an accelerator explicitly
export DATARAX_TEST_JAX_PLATFORMS="cuda"
export XLA_CLIENT_MEM_FRACTION="${XLA_CLIENT_MEM_FRACTION:-0.75}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# Check if GPU is available
echo ""
echo -e "${BLUE}Checking for GPU availability...${NC}"
if ! uv run python scripts/check_gpu.py; then
    echo ""
    echo -e "${RED}❌ GPU check failed. Cannot run GPU tests.${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure NVIDIA drivers are installed: nvidia-smi"
    echo "  2. Set up the CUDA 12 backend: ./setup.sh --backend cuda12"
    echo "  3. Inspect the JAX backend: uv run python scripts/verify_datarax_gpu.py"
    exit 1
fi

# Run the test suite on the GPU
echo ""
echo -e "${BLUE}Running GPU tests...${NC}"
uv run pytest -v tests/

echo ""
echo -e "${GREEN}✅ GPU test run complete!${NC}"
