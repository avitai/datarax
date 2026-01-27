#!/bin/bash
# Datarax Development Environment Setup
# Script for creating, building, and activating venv for both CPU and GPU development

set -e  # Exit on any error

# Default values
DEEP_CLEAN=false
CPU_ONLY=false
ENABLE_METAL=false
HELP=false
VERBOSE=false
FORCE_REINSTALL=false

# Detect operating system and architecture
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)
IS_MACOS=false
IS_APPLE_SILICON=false

if [ "$OS_TYPE" = "Darwin" ]; then
    IS_MACOS=true
    if [ "$ARCH_TYPE" = "arm64" ]; then
        IS_APPLE_SILICON=true
    fi
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deep-clean)
            DEEP_CLEAN=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --metal)
            ENABLE_METAL=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    cat << 'EOF'
🚀 Datarax Development Environment Setup
=======================================

Creates, builds, and prepares the virtual environment for Datarax development
with automatic GPU/CPU detection and optimal configuration.

Supports Linux (with NVIDIA CUDA) and macOS (Intel and Apple Silicon).

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    --deep-clean     Perform complete cleaning (JAX cache, pip cache, etc.)
    --cpu-only       Force CPU-only setup (skip GPU/Metal detection)
    --metal          Enable Metal acceleration on Apple Silicon Macs
    --force          Force reinstallation even if environment exists
    --verbose, -v    Show detailed output during setup
    --help, -h       Show this help message

EXAMPLES:
    ./setup.sh                    # Standard setup with auto GPU detection
    ./setup.sh --deep-clean       # Clean setup with cache clearing
    ./setup.sh --cpu-only         # Force CPU-only development setup
    ./setup.sh --metal            # macOS: Enable Metal acceleration (Apple Silicon)
    ./setup.sh --force --verbose  # Verbose forced reinstallation

PLATFORM SUPPORT:
    Linux:
        - Auto-detects NVIDIA CUDA GPUs
        - Falls back to CPU if no GPU found

    macOS (Intel):
        - CPU-only mode (no GPU acceleration available)

    macOS (Apple Silicon M1/M2/M3+):
        - CPU mode by default
        - Use --metal flag to enable Metal acceleration

ACTIVATION:
    After setup, activate the environment with:
    source ./activate.sh

FILES CREATED:
    .venv/           Virtual environment directory
    .env             Environment variables and configuration
    activate.sh      Activation script
    uv.lock          Dependency lock file

REQUIREMENTS:
    - uv package manager (installed automatically if missing)
    - Python 3.11+ (handled by uv)
    - Linux: NVIDIA drivers (for GPU support)
    - macOS: Xcode Command Line Tools

EOF
    exit 0
fi

# Utility functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_step() {
    echo -e "${PURPLE}🔧 $1${NC}"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}   → $1${NC}"
    fi
}

# Function to check and install uv if needed
ensure_uv_installed() {
    if ! command -v uv &> /dev/null; then
        log_warning "uv not found. Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            log_error "Failed to install uv. Please install manually:"
            echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        log_success "uv installed successfully"
    else
        verbose_log "uv already installed: $(uv --version)"
    fi
}

# Function to detect Metal support on macOS
detect_metal_support() {
    if [ "$IS_MACOS" != true ]; then
        return 1
    fi

    if [ "$CPU_ONLY" = true ]; then
        log_info "CPU-only mode requested, skipping Metal detection"
        return 1
    fi

    if [ "$ENABLE_METAL" != true ]; then
        log_info "Metal not enabled (use --metal flag on Apple Silicon)"
        return 1
    fi

    if [ "$IS_APPLE_SILICON" = true ]; then
        log_success "Apple Silicon detected ($ARCH_TYPE) - Metal acceleration available"
        return 0
    else
        log_warning "Metal requested but not on Apple Silicon (Intel Mac detected)"
        log_info "Metal acceleration is only available on M1/M2/M3+ chips"
        return 1
    fi
}

# Function to detect CUDA availability (Linux only)
detect_cuda_support() {
    if [ "$CPU_ONLY" = true ]; then
        log_info "CPU-only mode requested, skipping GPU detection"
        return 1
    fi

    # CUDA is not available on macOS
    if [ "$IS_MACOS" = true ]; then
        verbose_log "macOS detected - CUDA not available (use --metal for Apple Silicon)"
        return 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        local gpu_info
        if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) && [ -n "$gpu_info" ]; then
            log_success "NVIDIA GPU detected: $gpu_info"

            # Check CUDA installation
            if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ]; then
                verbose_log "CUDA installation found"
                return 0
            else
                log_warning "GPU detected but CUDA not found in standard locations"
                log_info "Will attempt GPU setup anyway"
                return 0
            fi
        fi
    fi

    log_info "No NVIDIA GPU detected - setting up CPU-only environment"
    return 1
}

# Function to perform cleaning
perform_cleaning() {
    log_step "Cleaning existing environment..."

    # Remove virtual environment
    if [ -d ".venv" ]; then
        verbose_log "Removing virtual environment (.venv)"
        rm -rf .venv
    fi

    # Remove lock files if force reinstall
    if [ "$FORCE_REINSTALL" = true ] && [ -f "uv.lock" ]; then
        verbose_log "Removing lock file (uv.lock)"
        rm -f uv.lock
    fi

    # Clean uv cache to avoid package conflicts
    verbose_log "Cleaning uv cache"
    uv cache clean 2>/dev/null || true

    # Remove existing environment files
    if [ -f ".env" ]; then
        verbose_log "Removing existing .env file"
        rm -f .env
    fi

    # Remove old activation scripts
    for script in activate_datarax.sh activate_env.sh activate_datarax_venv.sh setup_dev.sh; do
        if [ -f "$script" ]; then
            verbose_log "Removing old script: $script"
            rm -f "$script"
        fi
    done

    # Clean Python cache files
    verbose_log "Cleaning Python cache files"
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true

    # Deep cleaning if requested
    if [ "$DEEP_CLEAN" = true ]; then
        log_step "Performing deep cleaning..."

        # Clean JAX compilation cache
        if [ -d "$HOME/.cache/jax" ]; then
            verbose_log "Removing JAX compilation cache"
            rm -rf "$HOME/.cache/jax"
        fi

        # Clean pip cache
        verbose_log "Cleaning pip cache"
        python -m pip cache purge 2>/dev/null || pip cache purge 2>/dev/null || true

        # Clean pytest cache
        if [ -d ".pytest_cache" ]; then
            verbose_log "Removing pytest cache"
            rm -rf .pytest_cache
        fi

        # Clean coverage files
        for file in .coverage .coverage.*; do
            if [ -f "$file" ]; then
                verbose_log "Removing coverage file: $file"
                rm -f "$file"
            fi
        done
        if [ -d "htmlcov" ]; then
            verbose_log "Removing HTML coverage directory"
            rm -rf htmlcov
        fi

        # Clean benchmark results
        if [ -d "benchmark_results" ]; then
            verbose_log "Cleaning benchmark results"
            find benchmark_results -name "*.json" -delete 2>/dev/null || true
        fi

        # Clean temporary files
        verbose_log "Cleaning temporary files"
        find . -name "tmp*" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name ".tmp*" -type f -delete 2>/dev/null || true

        # Clean temp directory
        if [ -d "temp" ]; then
            verbose_log "Cleaning temp directory"
            rm -rf temp/*
        fi
    fi

    log_success "Environment cleaned"
}

# Function to create .env file
create_env_file() {
    local has_cuda=$1
    local has_metal=$2

    log_step "Creating environment configuration..."

    # Create cache directories
    mkdir -p .cache/jax .cache/xla 2>/dev/null || true

    if [ "$has_cuda" = true ]; then
        # Check if .env.example template exists and use it
        if [ -f ".env.example" ]; then
            # Copy template - it uses $(pwd) which will be evaluated when sourced
            cp .env.example .env
            verbose_log "Created GPU-enabled .env configuration from template"
        else
            # Fallback to embedded template if .env.example is missing
            # Detect Python version dynamically
            PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")

            cat > .env << EOF
# Datarax Environment Configuration - GPU Enabled (Linux/CUDA)
# Auto-generated by setup script

# CUDA Library Configuration - Use local venv CUDA installation
# Point to locally installed CUDA libraries in venv
export LD_LIBRARY_PATH="\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cusolver/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cusparse/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cufft/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/curand/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH}"

# JAX Configuration for CUDA
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_latency_hiding_scheduler=true"

# CUDA Performance Settings
export CUDA_MODULE_LOADING="LAZY"
export CUDA_CACHE_DISABLE="1"

# JAX Compilation Cache
export JAX_COMPILATION_CACHE_DIR="\$(pwd)/.cache/jax"
export XLA_CACHE_DIR="\$(pwd)/.cache/xla"

# JAX CUDA Plugin Configuration
export JAX_CUDA_PLUGIN_VERIFY="false"

# Reduce CUDA warnings
export TF_CPP_MIN_LOG_LEVEL="1"

# Performance settings
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="\${PYTHONPATH:+\${PYTHONPATH}:}\$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="true"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
            verbose_log "Created GPU-enabled .env configuration (CUDA)"
        fi
    elif [ "$has_metal" = true ]; then
        cat > .env << 'EOF'
# Datarax Environment Configuration - Metal Enabled (macOS/Apple Silicon)
# Auto-generated by setup script

# JAX Configuration for Metal (Apple Silicon)
# Metal backend is automatically used when jax-metal is installed
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Metal-specific optimizations
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_METAL_ENABLED="true"
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"
EOF
        verbose_log "Created Metal-enabled .env configuration (macOS)"
    elif [ "$IS_MACOS" = true ]; then
        cat > .env << 'EOF'
# Datarax Environment Configuration - CPU Only (macOS)
# Auto-generated by setup script

# JAX Configuration for CPU
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"
export PYTEST_METAL_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"
EOF
        verbose_log "Created CPU-only .env configuration (macOS)"
    else
        cat > .env << 'EOF'
# Datarax Environment Configuration - CPU Only (Linux)
# Auto-generated by setup script

# JAX Configuration for CPU
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"
EOF
        verbose_log "Created CPU-only .env configuration (Linux)"
    fi

    log_success "Environment configuration created"
}

# Function to create unified activation script with enhanced process detection
create_activation_script() {
    log_step "Creating activation script with enhanced process detection..."

    cat > activate.sh << 'EOF'
#!/bin/bash
# Datarax Environment Activation Script
# Created by setup script - includes enhanced process detection

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🚀 Activating Datarax Development Environment${NC}"
echo "============================================="

# Check if already activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment already active: $VIRTUAL_ENV${NC}"
    echo "Deactivating current environment..."

    # Improved process detection and user feedback
    # Check for processes using the virtual environment
    VENV_PROCESSES=$(pgrep -f "$VIRTUAL_ENV" | xargs -I {} ps -p {} -o pid,etime,args --no-headers 2>/dev/null || true)

    if [[ -n "$VENV_PROCESSES" ]]; then
        echo -e "${YELLOW}🔍 Checking for processes using the virtual environment...${NC}"

        # Count processes
        PROCESS_COUNT=$(echo "$VENV_PROCESSES" | wc -l)
        if [[ $PROCESS_COUNT -gt 0 ]]; then
            echo -e "${YELLOW}⏳ Found $PROCESS_COUNT process(es) using the virtual environment:${NC}"
            echo ""

            # Show process details in a readable format
            echo "$VENV_PROCESSES" | while IFS= read -r line; do
                if [[ -n "$line" ]]; then
                    PID=$(echo "$line" | awk '{print $1}')
                    ETIME=$(echo "$line" | awk '{print $2}')
                    CMD=$(echo "$line" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/[[:space:]]*$//')
                    echo -e "${CYAN}   • PID $PID (running for $ETIME): ${NC}$CMD"
                fi
            done

            echo ""
            echo -e "${YELLOW}💡 Options:${NC}"
            echo -e "${CYAN}   1. Wait for processes to complete naturally${NC}"
            echo -e "${CYAN}   2. Press Ctrl+C to cancel activation${NC}"
            echo -e "${CYAN}   3. In another terminal, stop processes manually:${NC}"
            echo -e "${CYAN}      pkill -f pytest  # Stop test processes${NC}"
            echo -e "${CYAN}      pkill -f jupyter # Stop Jupyter processes${NC}"
            echo ""
            echo -e "${YELLOW}⏳ Waiting for processes to complete (this may take a while)...${NC}"
            echo -e "${CYAN}   You can press Ctrl+C to cancel and handle processes manually${NC}"
        fi
    fi

    # Attempt deactivation with timeout and progress indication
    echo -e "${YELLOW}🔄 Attempting environment deactivation...${NC}"

    # Function to show waiting animation
    show_waiting() {
        local delay=1
        local spinstr="|/-\\"
        local temp
        while true; do
            temp=${spinstr#?}
            printf "\r%s   [%c] Waiting for environment deactivation...%s" "${CYAN}" "$spinstr" "${NC}"
            spinstr=$temp${spinstr%"$temp"}
            sleep $delay
        done
    }

    # Start background spinner
    show_waiting &
    SPINNER_PID=$!

    # Setup cleanup function for spinner
    cleanup_spinner() {
        if [[ -n "$SPINNER_PID" ]]; then
            kill $SPINNER_PID 2>/dev/null || true
            wait $SPINNER_PID 2>/dev/null || true
            printf "\r                                                    \r"  # Clear spinner line
        fi
    }

    # Setup trap to ensure spinner cleanup on script exit
    trap cleanup_spinner EXIT INT TERM

    # Try to deactivate with timeout
    if timeout 30 bash -c 'deactivate 2>/dev/null || true'; then
        cleanup_spinner
        printf "\r%s✅ Environment deactivation completed%s\n" "${GREEN}" "${NC}"
    else
        cleanup_spinner
        printf "\r%s❌ Environment deactivation timed out after 30 seconds%s\n" "${RED}" "${NC}"
        echo -e "${YELLOW}💡 This usually means processes are still using the environment.${NC}"
        echo -e "${CYAN}   Run this command to force cleanup:${NC}"
        echo -e "${CYAN}   pkill -f '$VIRTUAL_ENV'${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Proceeding with activation anyway...${NC}"
        # Clear VIRTUAL_ENV to force reactivation
        unset VIRTUAL_ENV
    fi

    # Clear the trap
    trap - EXIT INT TERM
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
EOF

    chmod +x activate.sh
    log_success "Activation script created: ./activate.sh"
}

# Function to create virtual environment and install dependencies
setup_environment() {
    local has_cuda=$1
    local has_metal=$2

    log_step "Creating virtual environment..."
    uv venv

    # Activate the environment for installation
    source .venv/bin/activate
    source .env

    log_step "Installing dependencies..."

    # First, create lock file if it doesn't exist
    if [ ! -f "uv.lock" ]; then
        log_info "Creating dependency lock file..."
        uv lock
    fi

    if [ "$has_cuda" = true ]; then
        log_info "Installing with CUDA support..."
        log_info "This may take several minutes on first install..."
        verbose_log "Installing complete CUDA stack locally in venv"

        # Install complete CUDA runtime stack locally in venv with all dependencies
        echo -e "${CYAN}   → Running: uv sync --extra all${NC}"
        if uv sync --extra all; then
            log_success "Local CUDA runtime installed in venv with all dependencies"
        else
            log_warning "CUDA installation failed, falling back to CPU-only"
            echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
            uv sync --extra all-cpu
            has_cuda=false
            # Update .env file for CPU-only
            create_env_file false false
        fi
    elif [ "$has_metal" = true ]; then
        log_info "Installing with Metal support (Apple Silicon)..."
        log_info "This may take several minutes on first install..."
        verbose_log "Installing JAX with Metal backend"

        # Install with Metal support for Apple Silicon
        echo -e "${CYAN}   → Running: uv sync --extra all-macos${NC}"
        if uv sync --extra all-macos; then
            log_success "Metal acceleration installed successfully"
        else
            log_warning "Metal installation failed, falling back to CPU-only"
            echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
            uv sync --extra all-cpu
            has_metal=false
            # Update .env file for CPU-only
            create_env_file false false
        fi
    elif [ "$IS_MACOS" = true ]; then
        log_info "Installing CPU-only version for macOS..."
        log_info "This may take several minutes on first install..."
        echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
        uv sync --extra all-cpu
    else
        log_info "Installing CPU-only version with all dependencies..."
        log_info "This may take several minutes on first install..."
        echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
        uv sync --extra all-cpu
    fi

    log_success "Dependencies installed successfully"
    return 0
}

# Function to verify installation
verify_installation() {
    local has_cuda=$1

    log_step "Verifying installation..."

    # Test JAX installation
    python << PYTHON_EOF
import sys
import traceback

try:
    import jax
    import jax.numpy as jnp
    import flax
    import optax

    print(f"✅ Core dependencies verified:")
    print(f"   JAX: {jax.__version__}")
    print(f"   Flax: {flax.__version__}")
    print(f"   Optax: {optax.__version__}")

    # Test basic functionality
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x**2)
    print(f"✅ Basic computation test: {float(y)}")

    # Test devices
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']

    print(f"✅ Available devices: {len(devices)} total")
    if gpu_devices:
        print(f"✅ GPU devices detected: {len(gpu_devices)}")
        # Simple GPU test
        try:
            with jax.default_device(gpu_devices[0]):
                z = jnp.array([1., 2., 3.])
                w = jnp.dot(z, z)
            print(f"✅ GPU computation test: {float(w)}")
        except Exception as e:
            print(f"⚠️  GPU test warning: {e}")
    else:
        print("ℹ️  No GPU devices (CPU-only mode)")

    print("✅ Installation verification complete!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installation may be incomplete")
    sys.exit(1)
except Exception as e:
    print(f"❌ Verification error: {e}")
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

    local verify_status=$?
    if [ $verify_status -eq 0 ]; then
        log_success "Installation verified successfully"
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# Function to display setup summary
display_summary() {
    local has_cuda=$1
    local has_metal=$2

    echo ""
    echo -e "${GREEN}🎉 Datarax Development Environment Setup Complete!${NC}"
    echo "================================================="
    echo ""
    echo -e "${BLUE}📁 Files Created:${NC}"
    echo -e "${CYAN}   .venv/                 Virtual environment${NC}"
    echo -e "${CYAN}   .env                   Environment configuration${NC}"
    echo -e "${CYAN}   activate.sh            Unified activation script${NC}"
    echo -e "${CYAN}   uv.lock                Dependency lock file${NC}"
    echo ""
    echo -e "${BLUE}🚀 Quick Start:${NC}"
    echo -e "${YELLOW}   source ./activate.sh   ${NC}# Activate environment (use 'source'!)"
    echo -e "${CYAN}   uv run pytest tests/   ${NC}# Run tests to verify setup"
    echo ""

    if [ "$has_cuda" = true ]; then
        echo -e "${GREEN}🎮 GPU Support: ✅ CUDA Enabled${NC}"
        echo "   Your environment is ready for GPU-accelerated development!"
    elif [ "$has_metal" = true ]; then
        echo -e "${GREEN}🍎 GPU Support: ✅ Metal Enabled (Apple Silicon)${NC}"
        echo "   Your environment is ready for Metal-accelerated development!"
    elif [ "$IS_MACOS" = true ]; then
        echo -e "${BLUE}💻 GPU Support: ❌ CPU-Only Mode (macOS)${NC}"
        if [ "$IS_APPLE_SILICON" = true ]; then
            echo "   For Metal acceleration on Apple Silicon, re-run with: ./setup.sh --metal --force"
        else
            echo "   Intel Mac detected - CPU-only mode (no GPU acceleration available)"
        fi
    else
        echo -e "${BLUE}💻 GPU Support: ❌ CPU-Only Mode${NC}"
        echo "   For GPU support, ensure NVIDIA drivers and CUDA are installed,"
        echo "   then re-run with: ./setup.sh --force"
    fi
    echo ""
    echo -e "${PURPLE}📖 For more information, see README.md${NC}"
}

# Main execution function
main() {
    echo -e "${PURPLE}🚀 Datarax Development Environment Setup${NC}"
    echo "==============================================="
    echo ""

    # Display platform information
    if [ "$IS_MACOS" = true ]; then
        if [ "$IS_APPLE_SILICON" = true ]; then
            log_info "Platform: macOS (Apple Silicon - $ARCH_TYPE)"
        else
            log_info "Platform: macOS (Intel - $ARCH_TYPE)"
        fi
    else
        log_info "Platform: Linux ($ARCH_TYPE)"
    fi
    echo ""

    # Pre-flight checks
    ensure_uv_installed

    # Detect GPU/Metal capability
    HAS_CUDA=false
    HAS_METAL=false

    if [ "$IS_MACOS" = true ]; then
        # On macOS, check for Metal support
        if detect_metal_support; then
            HAS_METAL=true
        fi
    else
        # On Linux, check for CUDA support
        if detect_cuda_support; then
            HAS_CUDA=true
        fi
    fi

    # Check if environment already exists and handle appropriately
    if [ -d ".venv" ] && [ "$FORCE_REINSTALL" != true ]; then
        log_warning "Virtual environment already exists"
        echo "Use --force to reinstall or source ./activate.sh to use existing environment"
        exit 1
    fi

    # Perform cleanup
    perform_cleaning

    # Create configuration files
    create_env_file "$HAS_CUDA" "$HAS_METAL"
    create_activation_script

    # Setup environment and install dependencies
    if ! setup_environment "$HAS_CUDA" "$HAS_METAL"; then
        log_error "Failed to setup environment"
        exit 1
    fi

    # Verify installation works
    if ! verify_installation "$HAS_CUDA"; then
        log_error "Installation verification failed"
        exit 1
    fi

    # Show summary
    display_summary "$HAS_CUDA" "$HAS_METAL"
}

# Run main function with all arguments
main "$@"
