#!/bin/bash
# Datarax Environment Activation Script
# Created by setup script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🚀 Activating Datarax Development Environment${NC}"
echo "============================================="

# Deactivate any existing virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment already active: $VIRTUAL_ENV${NC}"
    echo -e "${CYAN}🔄 Deactivating current environment...${NC}"

    # Properly call deactivate function if it exists
    # This runs in the current shell context, not a subshell
    if declare -f deactivate >/dev/null; then
        deactivate 2>/dev/null || true
        echo -e "${GREEN}✅ Previous environment deactivated${NC}"
    else
        echo -e "${YELLOW}⚠️  No deactivate function found, will override environment${NC}"
    fi
fi

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Run './setup.sh' to create the environment first."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    source .env
    echo -e "${GREEN}✅ Environment configuration loaded${NC}"

    # Show configuration based on JAX_PLATFORMS
    if [[ "$JAX_PLATFORMS" == *"cuda"* ]]; then
        echo -e "${CYAN}   🎮 GPU Mode: CUDA enabled${NC}"
        echo -e "${CYAN}   📍 CUDA_HOME: ${CUDA_HOME:-not set}${NC}"
    else
        echo -e "${CYAN}   💻 CPU Mode: CPU-only configuration${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found - using minimal setup${NC}"
    export JAX_PLATFORMS="cpu"
fi

# Display system information
echo ""
echo -e "${BLUE}🔍 Environment Status:${NC}"
echo -e "${CYAN}   Python: $(python --version)${NC}"
echo -e "${CYAN}   Working Directory: $(pwd)${NC}"
echo -e "${CYAN}   Virtual Environment: $VIRTUAL_ENV${NC}"

# Check JAX installation and display configuration
echo ""
echo -e "${BLUE}🧪 JAX Configuration:${NC}"

# JAX configuration check with error handling
python << 'PYTHON_EOF'
try:
    import jax
    import jax.numpy as jnp

    print(f"   JAX version: {jax.__version__}")
    print(f"   Default backend: {jax.default_backend()}")

    devices = jax.devices()
    print(f"   Available devices: {len(devices)} total")

    # Check for GPU devices
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    cpu_devices = [d for d in devices if d.platform == 'cpu']

    if gpu_devices:
        print(f"   🎉 GPU devices: {len(gpu_devices)} ({[str(d) for d in gpu_devices]})")
        print("   ✅ CUDA acceleration ready!")

        # Quick GPU test
        try:
            x = jnp.array([1., 2., 3.])
            y = jnp.sum(x**2)
            print(f"   🧮 GPU test successful: {float(y)}")
        except Exception as e:
            print(f"   ⚠️  GPU test warning: {e}")
    else:
        print(f"   💻 CPU devices: {len(cpu_devices)} ({[str(d) for d in cpu_devices]})")
        print("   📱 Running in CPU-only mode")

    # Quick functionality test
    try:
        x = jnp.linspace(0, 1, 100)
        y = jnp.sin(2 * jnp.pi * x)
        print(f"   ✅ JAX functionality verified")
    except Exception as e:
        print(f"   ❌ JAX functionality test failed: {e}")

except ImportError as e:
    print(f"   ❌ JAX not installed properly: {e}")
    print("   Run './setup.sh' to reinstall dependencies")
except Exception as e:
    print(f"   ⚠️  JAX configuration issue: {e}")
PYTHON_EOF

# Display usage information
echo ""
echo -e "${BLUE}🚀 Ready for Development!${NC}"
echo "========================="
echo ""
echo -e "${GREEN}📝 Common Commands:${NC}"
echo -e "${CYAN}   uv run pytest tests/ -v                     ${NC}# Run all tests"
echo -e "${CYAN}   uv run pytest tests/core/ -v                ${NC}# Run core tests"
echo -e "${CYAN}   uv run pytest tests/sources/ -v             ${NC}# Run data source tests"
echo -e "${CYAN}   uv run python your_script.py                ${NC}# Run your code"
echo -e "${CYAN}   uv run jupyter lab                          ${NC}# Start Jupyter lab"
echo ""
echo -e "${GREEN}🔧 Development Tools:${NC}"
echo -e "${CYAN}   uv add package_name                  ${NC}# Add new dependency"
echo -e "${CYAN}   uv run pre-commit run --all-files    ${NC}# Run code quality checks"
echo -e "${CYAN}   uv run pytest --cov=datarax tests/  ${NC}# Run tests with coverage"
echo ""
echo -e "${GREEN}📊 Benchmarking:${NC}"
echo -e "${CYAN}   uv run python -m datarax.benchmarking  ${NC}# Run benchmarks"
echo -e "${CYAN}   ./scripts/gpu_test_manager.py        ${NC}# GPU testing utilities"
echo ""
echo -e "${YELLOW}💡 To deactivate: ${NC}deactivate"
