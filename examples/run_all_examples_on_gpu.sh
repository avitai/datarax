#!/bin/bash

# Script to run all examples on GPU
# Usage: bash run_all_examples_on_gpu.sh

# Set up environment for GPU
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 # Limit memory usage to avoid OOM errors

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "====================================================================="
echo "Running selected examples on GPU"
echo "====================================================================="

# Function to run an example file
run_example() {
    example_file="$1"
    echo ""
    echo "---------------------------------------------------------------------"
    echo "Running: $example_file"
    echo "---------------------------------------------------------------------"

    python "$example_file"
    local result=$?

    if [ $result -eq 0 ]; then
        echo "✅ $example_file completed successfully"
        return 0
    else
        echo "❌ $example_file failed with exit code $result"
        return 1
    fi
}

# Select specific examples that are more likely to work on GPU
examples=(
    "rng_test_example.py"
    "advanced_optimized_transforms.py"
    "simplified_optimized_transforms.py"
    "simple_transform_test.py"
    "test_rotation.py"
    "optimized_transform_example.py"
    "image_transforms_example.py"
)

# Run the selected examples
failed_examples=()
success_count=0
total_count=0

for example in "${examples[@]}"; do
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
