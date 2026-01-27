# Datarax GPU Testing Guide

This document describes how to run Datarax tests on GPU hardware.

## Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 12.x installed
3. Python 3.11 virtual environment with JAX GPU support
4. Datarax development dependencies installed

## Setting Up the Environment

The recommended way to set up the environment is to use the main setup script:

```bash
# Set up the development environment with automatic GPU detection
./setup.sh

# Activate the virtual environment (loads .env with CUDA configuration)
source activate.sh
```

This approach:

- Automatically detects NVIDIA GPUs and configures CUDA
- Creates `.env` file with proper `LD_LIBRARY_PATH` for CUDA libraries
- Installs all dependencies including GPU support via `uv sync --extra all`
- Creates `activate.sh` that loads environment configuration

!!! warning "Legacy Script"
    The `scripts/setup_jax_gpu_env.sh` script is a legacy alternative that uses different
    conventions (`.venv311` directory, manual JAX CUDA installation). It is not recommended
    for new setups. Use `./setup.sh` instead.

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

## How GPU Testing Works

The GPU testing infrastructure consists of:

1. **Pytest `--device` Option**: The `conftest.py` provides a `--device` flag that accepts `cpu`, `gpu`, `tpu`, or `all`. When `--device=gpu` is specified, TPU-specific tests are skipped.

2. **Shell Script** (`scripts/run_gpu_tests.sh`):
   - Verifies GPU availability using `scripts/check_gpu.py`
   - Sets required environment variables (`JAX_PLATFORMS=cuda`)
   - Runs pytest with `--device=gpu` flag

3. **Python Script** (`scripts/run_gpu_tests.py`):
   - Provides more fine-grained control over GPU test execution
   - Runs GPU-relevant test directories (distributed, sharding, benchmarks)
   - Can also run example files on GPU

4. **Test Markers**: Tests can use `@pytest.mark.gpu` or `@pytest.mark.gpu_required` to indicate GPU requirements. Currently, most tests run on any device, with only a few explicitly marked as GPU-specific.

## Adding New GPU Tests

Most Datarax tests are device-agnostic and run on whatever JAX backend is available. Use GPU markers when a test:

- **Requires GPU** (would fail on CPU): Use `@pytest.mark.gpu_required`
- **Benefits from GPU** (runs faster): Use `@pytest.mark.gpu`

### Example: GPU-Required Test

```python
import pytest
import jax

@pytest.mark.gpu_required
def test_multi_gpu_sharding():
    """Test that requires multiple GPU devices."""
    devices = jax.devices("gpu")
    if len(devices) < 2:
        pytest.skip("Requires at least 2 GPUs")
    # Test multi-GPU functionality
```

### Example: GPU-Beneficial Test

```python
import pytest

@pytest.mark.gpu
def test_large_batch_processing():
    """Test that benefits from GPU acceleration."""
    # This test runs on any device but is faster on GPU
    pass
```

### Test File Location

Place GPU-intensive tests in appropriate directories:

- `tests/distributed/` - Multi-device and sharding tests
- `tests/sharding/` - Data sharding tests
- `tests/benchmarks/` - Performance benchmarks

## Testing Status

The GPU testing infrastructure supports:

- **Automatic device detection**: Tests adapt to available hardware
- **Selective test execution**: Use `--device=gpu` to focus on GPU-relevant tests
- **Memory management**: Environment variables control GPU memory allocation

### Running Full GPU Test Suite

```bash
# Run all tests on GPU
JAX_PLATFORMS=cuda uv run pytest --device=gpu tests/

# Run with memory limits (useful for shared GPUs)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 JAX_PLATFORMS=cuda uv run pytest --device=gpu tests/
```

For more testing information, see the [Testing Guide](testing_guide.md).
