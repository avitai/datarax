#!/bin/bash

# This script runs tests with JAX properly configured
# Dependencies can be installed with: pip install -e ".[test-cpu]"
# This installs both tensorflow-datasets and huggingface datasets for TFDS and HF tests

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if we want to run all tests on the specified device
FORCE_DEVICE=""
if [[ "$*" == *"--device=cpu"* ]]; then
    FORCE_DEVICE="cpu"
    echo "Forcing CPU device as specified in command arguments"
    export JAX_PLATFORMS="cpu"
elif [[ "$*" == *"--device=gpu"* ]]; then
    FORCE_DEVICE="gpu"
    echo "Forcing GPU device as specified in command arguments"
    export JAX_PLATFORMS="cuda"
fi

# If no specific device requested, check for GPU availability and run on both if available
if [ -z "$FORCE_DEVICE" ]; then
    # Use the more complete test script that handles GPU detection
    echo "No specific device requested. Automatically detecting and using available devices..."
    bash "$SCRIPT_DIR/run_tests_with_gpu_auto.sh" "$@"
    exit $?
fi

# Run all tests at once on specified device only
echo "Running all tests on $FORCE_DEVICE..."
uv run pytest "$@"

echo "All tests completed."

# Alternative command if you want to run without uv
# python -m pytest "$@"
