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

# Source .env if it exists (contains CUDA library paths)
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Loading environment configuration from .env${NC}"
    source .env
else
    echo -e "${YELLOW}⚠️  No .env file found - using default CUDA configuration${NC}"
    echo "   Run ./setup.sh first for proper CUDA setup"
fi

# Set GPU-specific environment variables
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
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
    echo "  2. Ensure CUDA is configured: ./setup.sh --force"
    echo "  3. Check LD_LIBRARY_PATH includes CUDA libraries"
    exit 1
fi

# Run GPU tests using pytest with --device=gpu
echo ""
echo -e "${BLUE}Running GPU tests...${NC}"
uv run pytest --device=gpu -v tests/

echo ""
echo -e "${GREEN}✅ GPU test run complete!${NC}"
