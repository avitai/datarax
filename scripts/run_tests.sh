#!/bin/bash

# Test runner script for Datarax
# Automatically detects GPU and runs tests with appropriate settings
# Usage: ./run_tests.sh [options] [pytest arguments]

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH=$PYTHONPATH:$SCRIPT_DIR

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# Function to safely check for GPU availability
check_gpu() {
    uv run python -c "
import jax
try:
    # Try to get GPU count, catching any JAX configuration errors
    count = jax.local_device_count('gpu')
    print(count)
except Exception:
    # If any error occurs (e.g. specialized hardware not found), return 0
    print(0)
" 2>/dev/null
}

echo "Checking for GPU availability..."
GPU_COUNT=$(check_gpu)

HAS_GPU=false
if [ "$GPU_COUNT" -gt 0 ]; then
    echo "✅ GPU detected (Count: $GPU_COUNT)! Will run tests on both CPU and GPU."
    HAS_GPU=true
else
    echo "ℹ️ No GPU detected or JAX GPU support not configured. Running tests on CPU only."
fi

# Function to run tests with specified device
run_tests_on_device() {
    local device=$1
    shift  # Remove the first argument (device)
    local extra_args=("$@")  # All remaining arguments as array

    echo ""
    echo "======================================================"
    echo "Running tests on $device"
    echo "======================================================"

    if [ "$device" = "gpu" ]; then
        # For GPU tests, set JAX to use CUDA
        export JAX_PLATFORMS="cuda"
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Limit memory usage to avoid OOM errors

        # Run tests with GPU enabled
        # We catch the exit code to ensure we continue even if some tests fail?
        # No, set -e is on. But for test runner, typically we want to return failure at end.

        # We need to temporarily disable set -e to capture failure
        set +e
        uv run pytest --device=gpu "${extra_args[@]}"
        local status=$?
        set -e

        if [ $status -ne 0 ]; then
            echo "❌ Tests failed on GPU"
            return $status
        fi
    else
        # For CPU tests, reset environment and force CPU
        unset JAX_PLATFORMS
        export JAX_PLATFORMS="cpu"

        set +e
        uv run pytest --device=cpu "${extra_args[@]}"
        local status=$?
        set -e

        if [ $status -ne 0 ]; then
            echo "❌ Tests failed on CPU"
            return $status
        fi
    fi
    return 0
}

# Parse command line arguments
PYTEST_ARGS=()
RUN_INTEGRATION=false
RUN_E2E=false
RUN_BENCHMARK=false
FORCE_DEVICE=""

for arg in "$@"; do
    if [ "$arg" = "--integration" ]; then
        RUN_INTEGRATION=true
    elif [ "$arg" = "--end-to-end" ]; then
        RUN_E2E=true
    elif [ "$arg" = "--benchmark" ]; then
        RUN_BENCHMARK=true
    elif [ "$arg" = "--all" ]; then
        RUN_INTEGRATION=true
        RUN_E2E=true
        RUN_BENCHMARK=true
    elif [[ "$arg" == "--device="* ]]; then
        # Handle --device=cpu/gpu argument explicitly if passed
        FORCE_DEVICE="${arg#*=}"
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# Build the test arguments based on what's selected
TEST_ARGS=("${PYTEST_ARGS[@]}")

if [ "$RUN_INTEGRATION" = true ]; then
    TEST_ARGS+=("--integration")
fi

if [ "$RUN_E2E" = true ]; then
    TEST_ARGS+=("--end-to-end")
fi

if [ "$RUN_BENCHMARK" = true ]; then
    TEST_ARGS+=("--benchmark")
fi

# Track overall success
EXIT_CODE=0

# If a specific device is forced, run only on that
if [ -n "$FORCE_DEVICE" ]; then
    run_tests_on_device "$FORCE_DEVICE" "${TEST_ARGS[@]}"
    EXIT_CODE=$?
else
    # Otherwise run on CPU first
    run_tests_on_device "cpu" "${TEST_ARGS[@]}"
    CPU_STATUS=$?

    if [ $CPU_STATUS -ne 0 ]; then
        EXIT_CODE=$CPU_STATUS
    fi

    # If GPU is available and CPU tests passed (or we want to run both regardless? usually fail fast)
    # Let's run GPU tests even if CPU failed, to see full picture, or maybe fail fast?
    # Original script ran both if available.

    if [ "$HAS_GPU" = true ]; then
        run_tests_on_device "gpu" "${TEST_ARGS[@]}"
        GPU_STATUS=$?
        if [ $GPU_STATUS -ne 0 ]; then
            EXIT_CODE=$GPU_STATUS
        fi
    fi
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================"
    echo "✅ All tests completed successfully"
    echo "======================================================"
else
    echo "======================================================"
    echo "❌ Some tests failed"
    echo "======================================================"
fi

exit $EXIT_CODE
