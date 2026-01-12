#!/bin/bash

# Force JAX to use CPU to avoid GPU memory issues and segmentation faults
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run benchmark tests with verbose output
uv run pytest -v tests/benchmark_*.py "$@" --benchmark-autosave --benchmark-columns=mean,stddev,min,max
