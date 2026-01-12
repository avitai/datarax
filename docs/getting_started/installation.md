# Installation Guide

This guide covers installing Datarax and its dependencies.

## Requirements

Datarax requires:

- Python 3.11 or higher
- JAX 0.4.38 or higher
- Flax 0.10 or higher

## Basic Installation

Install Datarax with pip:

```bash
pip install datarax
```

This will install Datarax with minimal dependencies.

## Installation with Optional Dependencies

Datarax provides several optional dependency groups:

```bash
# Install with all optional dependencies
pip install datarax[all]

# Install with specific optional dependencies
pip install datarax[docs]     # Documentation dependencies
pip install datarax[hf]       # HuggingFace datasets support
pip install datarax[tfds]     # TensorFlow datasets support
pip install datarax[viz]      # Visualization dependencies
pip install datarax[benchmark] # Benchmarking tools
```

## GPU Support

To use Datarax with CUDA-enabled GPUs:

1. Ensure you have compatible NVIDIA drivers installed
2. Install JAX with CUDA support

```bash
# For CUDA 12
pip install "jax[cuda12_pip]~=0.4.38"

# Or for CUDA 11
pip install "jax[cuda11_pip]~=0.4.38"
```

This will install the appropriate CUDA and cuDNN dependencies.

## Development Installation

For development, install Datarax from source:

```bash
git clone https://github.com/datarax/datarax.git
cd datarax
pip install -e ".[dev,docs,test]"
```

## Environment Setup

Datarax works well with environment management tools:

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is the recommended package manager for Datarax development:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a new environment
uv venv
source .venv/bin/activate

# Install Datarax and dependencies
uv pip install -e ".[all]"
```

### Using conda/mamba

```bash
conda create -n datarax python=3.11
conda activate datarax
pip install datarax
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

#### JAX GPU Detection Issues

If JAX doesn't detect your GPU:

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

#### Memory Issues

For GPU memory management:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75  # Use only 75% of GPU memory
```

#### Version Conflicts

If you encounter package conflicts:

```bash
pip install --upgrade "datarax[all]" --force-reinstall
```

### Getting Help

If you encounter issues:

1. Check the [GitHub issues](https://github.com/datarax/datarax/issues) for similar problems
2. Create a new issue with:
   - Datarax version
   - JAX version
   - Python version
   - CUDA version (if using GPU)
   - Error messages and a minimal code example
