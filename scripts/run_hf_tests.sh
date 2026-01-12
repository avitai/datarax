#!/bin/bash

# Force JAX to use CPU to avoid GPU memory issues and segmentation faults
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run HuggingFace datasets compatibility tests
echo "Running HuggingFace datasets compatibility tests..."
python examples/hf_datasets_test.py 2>&1 | tee logs/hf_datasets_test_results.log

# Run the example
echo "Running HuggingFace example..."
python examples/hf_example.py 2>&1 | tee logs/hf_example_results.log

# Run the unit tests
echo "Running HuggingFace unit tests..."
uv run pytest tests/test_hf_source.py -v 2>&1 | tee logs/hf_unit_tests_results.log

# Run the integration tests
echo "Running HuggingFace integration tests..."
uv run pytest tests/test_hf_pipeline_integration.py -v 2>&1 | tee logs/hf_integration_tests_results.log

echo "All HuggingFace tests completed!"
