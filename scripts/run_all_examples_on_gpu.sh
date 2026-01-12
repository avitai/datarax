#!/bin/bash

# Script to run all examples on GPU
# Usage: bash run_all_examples_on_gpu.sh

set -e

# Set up environment for GPU
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 # Limit memory usage to avoid OOM errors

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$PROJECT_ROOT/examples"
cd "$PROJECT_ROOT"

echo "====================================================================="
echo "Running all examples on GPU"
echo "====================================================================="

# Function to run an example file
run_example() {
    example_file="$1"
    echo ""
    echo "---------------------------------------------------------------------"
    echo "Running: $example_file"
    echo "---------------------------------------------------------------------"

    if python "$example_file"; then
        echo "✅ $example_file completed successfully"
        return 0
    else
        echo "❌ $example_file failed"
        return 1
    fi
}

# Find all Python files in the examples directory and run them
failed_examples=()
success_count=0
total_count=0

for example in $(find "$EXAMPLES_DIR" -name "*.py" | grep -v "__pycache__" | grep -v "__init__.py" | grep -v "config_example.py" | sort); do
    if run_example "$example"; then
        ((success_count++))
    else
        failed_examples+=("$example")
    fi
    ((total_count++))
done

echo ""
echo "====================================================================="
echo "Example execution summary"
echo "====================================================================="
echo "Total examples: $total_count"
echo "Successfully ran: $success_count"
echo "Failed: $((total_count - success_count))"

if [ ${#failed_examples[@]} -gt 0 ]; then
    echo ""
    echo "Failed examples:"
    for failed in "${failed_examples[@]}"; do
        echo "  - $failed"
    done
    exit 1
else
    echo ""
    echo "All examples ran successfully!"
    exit 0
fi
