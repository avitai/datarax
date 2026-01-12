#!/bin/bash

# Unified test script for Datarax
# Automatically detects GPU and runs tests with appropriate settings
# - Runs all tests on CPU by default
# - If GPU is available, also runs tests on GPU

set -e

# Check if GPU is available using JAX
echo "Checking for GPU availability..."
HAS_GPU=false
GPU_COUNT=$(uv run python -c "import jax; print(jax.local_device_count('gpu'))")

if [ "$GPU_COUNT" -gt 0 ]; then
    echo "✅ GPU detected! Will run tests on both CPU and GPU."
    HAS_GPU=true
else
    echo "ℹ️ No GPU detected. Running tests on CPU only."
fi

# Set up environment
current_dir=$(pwd)
export PYTHONPATH=$PYTHONPATH:$current_dir

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
        uv run pytest --device=gpu "${extra_args[@]}"
    else
        # For CPU tests, reset environment and force CPU
        unset JAX_PLATFORMS
        export JAX_PLATFORMS="cpu"

        # Run tests on CPU
        uv run pytest --device=cpu "${extra_args[@]}"
    fi
}

# Parse command line arguments
PYTEST_ARGS=""
RUN_INTEGRATION=false
RUN_E2E=false
RUN_BENCHMARK=false

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
    else
        PYTEST_ARGS="$PYTEST_ARGS $arg"
    fi
done

# Build the test arguments based on what's selected
TEST_ARGS="$PYTEST_ARGS"

if [ "$RUN_INTEGRATION" = true ]; then
    TEST_ARGS="$TEST_ARGS --integration"
fi

if [ "$RUN_E2E" = true ]; then
    TEST_ARGS="$TEST_ARGS --end-to-end"
fi

if [ "$RUN_BENCHMARK" = true ]; then
    TEST_ARGS="$TEST_ARGS --benchmark"
fi

# Always run tests on CPU first
run_tests_on_device "cpu" "$TEST_ARGS"

# If GPU is available, also run tests on GPU
if [ "$HAS_GPU" = true ]; then
    run_tests_on_device "gpu" "$TEST_ARGS"
fi

echo ""
echo "======================================================"
echo "All tests completed"
echo "======================================================"
