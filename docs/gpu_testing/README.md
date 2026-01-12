# Datarax GPU Testing Guide

This document describes how to run Datarax tests on GPU hardware.

## Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 12.x installed
3. Python 3.11 virtual environment with JAX GPU support
4. Datarax development dependencies installed

## Setting Up the Environment

The easiest way to set up the environment is to use the setup script:

```bash
# Set up the development environment with automatic GPU detection
./setup.sh

# Activate the virtual environment
source activate.sh
```

Alternatively, for manual GPU environment setup:

```bash
bash scripts/setup_jax_gpu_env.sh
source .venv/bin/activate
```

## Running GPU Tests

We provide a dedicated script for running tests on GPU:

```bash
# Run all GPU-specific tests
bash scripts/run_gpu_tests.sh
```

This script will:

1. Check for GPU availability
2. Set up the required environment variables
3. Run examples that previously had GPU issues
4. Run selected tests with GPU support

## Manual GPU Testing

If you want more control over which tests to run on GPU, you can:

```bash
# Set the environment to use CUDA
export JAX_PLATFORMS="cuda"

# Run all tests with GPU device selection
uv run pytest --device=gpu

# Run a specific test on GPU
uv run pytest --device=gpu tests/operators/
```

## Troubleshooting

If you encounter issues with GPU tests:

1. **Verify GPU is detected**:

   ```bash
   uv run python scripts/check_gpu.py
   ```

2. **Check CUDA installation**:

   ```bash
   nvidia-smi
   ```

3. **Memory issues**: Adjust memory fraction if tests fail due to OOM errors:

   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
   ```

4. **GPU acceleration not used**: Ensure JAX is using the GPU:

   ```bash
   JAX_PLATFORMS=cuda python -c "import jax; print(jax.devices())"
   ```

## Recent Improvements

We recently made several improvements to the GPU testing system:

1. **Fixed Test Selection Issue**: Previously, the `run_gpu_tests.py` script was only running tests marked with `@pytest.mark.gpu`, but few tests utilized this marker. The script has been updated to:
   - Run specific test files that are relevant for GPU testing
   - Use the `--device=gpu` option to enable GPU testing
   - Force JAX to use CUDA by setting the `JAX_PLATFORMS` environment variable

2. **Created Dedicated Shell Script**: Added a convenient `scripts/run_gpu_tests.sh` script that:
   - Verifies GPU availability before running tests
   - Sets required environment variables automatically
   - Provides a clean output of test results

3. **Better Error Handling**: Improved error reporting and test outcome verification

For a detailed summary of all GPU testing improvements, see the GPU testing documentation in the main docs directory.

## Adding New GPU Tests

Currently, Datarax does not have many tests explicitly marked with `@pytest.mark.gpu`, which means most tests are run regardless of the device type.

To add a new GPU-specific test:

1. Add the GPU marker to your test:

   ```python
   @pytest.mark.gpu
   def test_my_gpu_feature():
       # Test code here
   ```

2. Update the `run_gpu_tests.py` script if needed to include your new test files.

## Testing Status

The GPU testing pipeline has been updated to ensure all tests can run successfully on GPU hardware. This helps catch GPU-specific issues early in the development process.
