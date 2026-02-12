# Installation Guide

This guide covers installing Datarax and its dependencies.

## Requirements

Datarax requires:

- Python 3.11 or higher
- JAX 0.6.1 or higher
- Flax 0.12 or higher

**Supported Platforms:**

- Linux (x86_64) - with optional CUDA GPU support
- macOS (Intel x86_64) - CPU-only
- macOS (Apple Silicon M1/M2/M3+) - with optional Metal acceleration

## Basic Installation

Install Datarax with pip:

```bash
pip install datarax
```

This will install Datarax with minimal dependencies (CPU-only).

## Installation with Optional Dependencies

Datarax provides several optional dependency groups:

```bash
# Install with all optional dependencies (Linux with GPU)
pip install datarax[all]

# Install with all dependencies (CPU-only, works on all platforms)
pip install datarax[all-cpu]

# Install with all dependencies (macOS with Metal acceleration)
pip install datarax[all-macos]

# Install with specific optional dependencies
pip install datarax[docs]     # Documentation dependencies
pip install datarax[data]     # Data loading (HuggingFace, TFDS, etc.)
pip install datarax[test]     # Testing dependencies
pip install datarax[metal]    # Metal acceleration (Apple Silicon only)
```

## Platform-Specific Installation

### Linux with NVIDIA GPU (CUDA)

To use Datarax with CUDA-enabled GPUs:

1. Ensure you have compatible NVIDIA drivers installed
2. Install with GPU support:

```bash
# Install Datarax with CUDA 12 support
pip install datarax[all]

# Or install JAX with CUDA separately
pip install "jax[cuda12]>=0.6.1"
```

This will install the appropriate CUDA and cuDNN dependencies.

### macOS (Intel)

Intel Macs run in CPU-only mode:

```bash
# Install Datarax for macOS (CPU-only)
pip install datarax[all-cpu]
```

### macOS (Apple Silicon - M1/M2/M3+)

Apple Silicon Macs can use Metal for GPU-like acceleration:

```bash
# Option 1: CPU-only (simpler, always works)
pip install datarax[all-cpu]

# Option 2: With Metal acceleration (recommended for performance)
pip install datarax[all-macos]
```

**Note:** Metal acceleration requires:

- macOS 12.0 (Monterey) or later
- Apple Silicon chip (M1, M2, M3, or newer)

To verify Metal is working:

```python
import jax
print(jax.devices())
# Should show: [METAL:0] or similar
```

## Development Installation

For development, install Datarax from source:

```bash
git clone https://github.com/avitai/datarax.git
cd datarax

# Use the automated setup script (recommended)
./setup.sh

# Or install manually based on your platform:
# Linux with GPU:
pip install -e ".[all]"

# macOS (Apple Silicon with Metal):
pip install -e ".[all-macos]"

# Any platform (CPU-only):
pip install -e ".[all-cpu]"
```

## Environment Setup

Datarax works well with environment management tools:

### Using the Setup Script (Recommended)

The included `setup.sh` script automatically detects your platform and configures the environment:

```bash
# Clone the repository
git clone https://github.com/avitai/datarax.git
cd datarax

# Run setup with auto-detection
./setup.sh

# Or with specific options:
./setup.sh --cpu-only    # Force CPU-only mode
./setup.sh --metal       # Enable Metal (Apple Silicon only)
./setup.sh --force       # Reinstall existing environment
./setup.sh --help        # Show all options

# Activate the environment
source ./activate.sh
```

### Using uv Manually

[uv](https://github.com/astral-sh/uv) is the recommended package manager for Datarax development:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a new environment
uv venv
source .venv/bin/activate

# Install based on your platform:
# Linux with CUDA:
uv pip install -e ".[all]"

# macOS with Metal:
uv pip install -e ".[all-macos]"

# CPU-only (any platform):
uv pip install -e ".[all-cpu]"
```

### Using conda/mamba

```bash
conda create -n datarax python=3.11
conda activate datarax

# Install based on your platform
pip install datarax[all-cpu]  # CPU-only
# or
pip install datarax[all-macos]  # macOS with Metal
```

## Verifying Installation

You can verify your installation by running:

```python
import datarax as jf
print(f"Datarax version: {jf.__version__}")
```

Or run a simple test pipeline:

```python
import jax.numpy as jnp
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

# Create sample data
data = [{"image": jnp.ones((28, 28)), "label": i % 10} for i in range(10)]

# Create a simple pipeline using the DAG-based API
config = MemorySourceConfig()
source = MemorySource(config, data)
pipeline = from_source(source, batch_size=2)

# Iterate through the pipeline
for i, batch in enumerate(pipeline):
    print(f"Batch {i}: shape = {batch['image'].shape}")
```

## Troubleshooting

### Common Issues

#### JAX GPU Detection Issues (Linux/CUDA)

If JAX doesn't detect your NVIDIA GPU:

1. Verify CUDA installation: `nvidia-smi`
2. Check JAX can see GPU devices:

   ```python
   import jax
   print(jax.devices())
   ```

3. Set environment variables:

   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_ALLOCATOR=platform
   ```

#### Metal Not Working (macOS Apple Silicon)

If Metal acceleration isn't working on Apple Silicon:

1. Verify you have an Apple Silicon Mac:

   ```bash
   uname -m  # Should show "arm64"
   ```

2. Check that jax-metal is installed:

   ```bash
   pip list | grep jax-metal
   ```

3. Verify JAX sees the Metal device:

   ```python
   import jax
   print(jax.devices())
   # Should show [METAL:0] or similar
   ```

4. If using the setup script, ensure you used the `--metal` flag:

   ```bash
   ./setup.sh --metal --force
   ```

5. Ensure you're on macOS 12.0+ (Monterey or later)

#### Memory Issues

For GPU/Metal memory management:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Use only 75% of GPU memory
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Don't preallocate memory
```

#### Version Conflicts

If you encounter package conflicts:

```bash
# Linux with GPU:
pip install --upgrade "datarax[all]" --force-reinstall

# macOS:
pip install --upgrade "datarax[all-cpu]" --force-reinstall

# macOS with Metal:
pip install --upgrade "datarax[all-macos]" --force-reinstall
```

#### TensorFlow Issues on macOS

TensorFlow on macOS is CPU-only. If you encounter TensorFlow-related errors:

```bash
# Ensure you're using the CPU version
export TF_CPP_MIN_LOG_LEVEL=1
```

#### Deep Lake + TensorFlow OpenSSL Crash

**Problem**: When Deep Lake and TensorFlow are both installed, importing Deep Lake *after* TensorFlow kills the process with a fatal C-level assertion:

```
Fatal error condition occurred in .../aws-c-cal/.../openssl_platform_init.c:641:
    strncmp(openssl_prefix, runtime_version, strlen(openssl_prefix)) == 0
```

This is not a Python exception — it terminates the process immediately and cannot be caught with `try/except`.

**Cause**: Deep Lake's native extension bundles [aws-c-cal](https://github.com/awslabs/aws-c-cal), which performs a strict OpenSSL version check at initialization. TensorFlow ships its own BoringSSL (an OpenSSL fork). When TensorFlow loads first, BoringSSL occupies the OpenSSL symbol space. When Deep Lake's `aws-c-cal` then checks the OpenSSL version string, it finds BoringSSL's version instead — assertion failure.

**Solution**: Import `deeplake` **before** `tensorflow`. When Deep Lake loads first, `aws-c-cal` initializes against the system's real OpenSSL. TensorFlow's BoringSSL then loads into a separate symbol space without conflict.

```python
# Correct: Deep Lake first
import deeplake
import tensorflow as tf  # safe — BoringSSL loads into separate space

# Wrong: TensorFlow first — FATAL CRASH
import tensorflow as tf
import deeplake  # aws-c-cal finds BoringSSL, assertion fails, process dies
```

!!! note "Handled automatically in the test suite"
    Datarax's `tests/conftest.py` pre-imports Deep Lake before TensorFlow so that
    `uv run pytest` and `uv run pytest --all-suites` work without issues.
    If you write standalone scripts that use both libraries, ensure the import order
    is correct.

This is an upstream issue between Deep Lake (v4.x, `aws-c-cal` v0.8.1) and TensorFlow (BoringSSL). No environment variable or configuration can bypass the assertion — import order is the only workaround as of February 2026.

### Getting Help

If you encounter issues:

1. Check the [GitHub issues](https://github.com/avitai/datarax/issues) for similar problems
2. Create a new issue with:

   - Datarax version
   - JAX version
   - Python version
   - Platform (Linux, macOS Intel, macOS Apple Silicon)
   - CUDA version (if using Linux GPU)
   - Error messages and a minimal code example
