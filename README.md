# Datarax: A Data Pipeline Framework for JAX

[![CI](https://github.com/avitai/datarax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/ci.yml)
[![Test Coverage](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml)
[![codecov](https://codecov.io/gh/avitai/datarax/branch/main/graph/badge.svg)](https://codecov.io/gh/avitai/datarax)
[![Build](https://github.com/avitai/datarax/actions/workflows/build-verification.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/build-verification.yml)
[![Summary](https://github.com/avitai/datarax/actions/workflows/summary.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/summary.yml)

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

> **Note:** This project is in early development. API may change. Expect breaking changes.

**Datarax** (*Data + Array/JAX*) is a high-performance, extensible data pipeline framework specifically engineered for JAX-based machine learning workflows. It simplifies and accelerates the development of efficient and scalable data loading, preprocessing, and augmentation pipelines for JAX, leveraging the full potential of JAX's Just-In-Time (JIT) compilation, automatic differentiation, and hardware acceleration capabilities.

## Key Features

- **High Performance:** Leverages JAX's JIT compilation and XLA backend to achieve near-optimal data processing speeds on CPUs, GPUs, and TPUs.
- **JAX-Native Design:** All core components and operations are designed with JAX's functional programming paradigm and immutable data structures (PyTrees) in mind.
- **Scalability:** Supports efficient data loading and processing for large datasets and distributed training scenarios, including multi-host and multi-device setups.
- **Extensibility:** Easily define and integrate custom data sources, transformations, and augmentation operations.
- **Usability:** Provides a clear, intuitive Python API and a flexible configuration system (TOML-based) for defining and managing pipelines.
- **Determinism:** Pipeline runs are deterministic by default, crucial for reproducibility in research and production.
- **Caching Optimization:** Multiple caching strategies for performance improvement, including function caching, transformer caching, and checkpointing.
- **Complete Feature Set:** Supports common data pipeline operations including diverse data source handling, advanced transformations, data augmentation, batching, sharding, checkpointing, and caching.
- **Ecosystem Integration:** Facilitates smooth integration with other JAX libraries like Flax, Optax, and Orbax.

## Installation

**IMPORTANT:** Datarax uses `uv` as its package manager for all installation, development, and deployment tasks.

```bash
# Install uv if you don't have it already
pip install uv

# Run the automatic setup script (creates environment & installs dependencies)
./setup.sh

# Activate the environment
source activate.sh

# Install via uv
uv pip install datarax

# Install with optional dependencies
uv pip install datarax[data]     # For data loading (HF, TFDS, Audio/Image libs)
uv pip install datarax[gpu]      # For GPU support (CUDA 12)

# For development
uv pip install datarax[dev]

# For all dependencies
uv pip install datarax[all]
```

## Development Environment

Datarax development requires using `uv` for all package management operations. See the [Development Environment Guide](docs/dev_guide.md) for detailed instructions on setting up a development environment, including:

- Creating a virtual environment with `uv venv`
- Installing dependencies through `pyproject.toml`
- Running tests through pytest
- Building and packaging Datarax

## Current Status

Datarax is currently in active development with a **Flax NNX-based architecture** that provides robust state management and checkpointing:

### Core Architecture

- **NNX Module System**: All components built on `flax.nnx.Module` for robust state management
- **Integrated Checkpointing**: Seamless Orbax integration for stateful pipeline persistence
- **Type Safety**: Complete type annotations and runtime validation
- **Composability**: Modular design enabling flexible pipeline construction

### Implemented Components

- **Core NNX Modules**: DataraxModule, OperatorModule, StructuralModule
- **Data Sources**: MemorySource, TFDSSource, HFSource (inheriting from StructuralModule/DataSourceModule)
- **Operators**: Element-wise operators, MapOperator (inheriting from OperatorModule)
- **DAG Execution Engine**: `DAGExecutor` and `pipeline` API for constructing flexible, graph-based data processing flows.
- **Node System**: A rich set of nodes for building pipelines:
  - `DataSourceNode`: entry point for data
  - `OperatorNode`: for transformations
  - `BatchNode` & `RebatchNode`: for batching control
  - `ShuffleNode`, `CacheNode`: for data management
  - Control flow nodes: `Sequential`, `Parallel`, `Branch`, `Merge`
- **Stateful Components**:
  - MemorySourceModule/ArrayRecordSourceModule/HFSourceModule for diverse data ingestion
  - Range/ShuffleSamplerModule with reproducible random state
  - DefaultBatcherModule with buffer state management
  - ArraySharderModule for device-aware data distribution
- **External Integrations**:
  - HuggingFace Datasets with stateful iteration
  - TensorFlow Datasets with checkpoint support
- **Advanced Features**:
  - Complete caching strategies with state preservation
  - Differentiable rebatching (`DifferentiableRebatchImpl`)
  - NNXCheckpointHandler for production-grade checkpointing

Upcoming features include:

- Image transformation library with JAX-native operations
- Advanced sharding strategies for multi-device and multi-host scenarios
- Performance optimization suite with benchmarking tools
- Extended monitoring and metrics capabilities
- Additional external data source integrations

## Quick Start

Here's a simple example of using Datarax's DAG-based architecture to create a data pipeline for image classification:

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.typing import Element


def normalize(element: Element, key: jax.Array | None = None) -> Element:
    """Normalize the image in the element."""
    # element.data is a dict-like PyTree containing the actual data arrays
    # Return a new Element with updated data (immutable update)
    # update_data provides a clean API for partial updates
    return element.update_data({"image": element.data["image"] / 255.0})


def apply_augmentation(element: Element, key: jax.Array) -> Element:
    """Apply a simple augmentation (flipping) to the image."""
    key1, _ = jax.random.split(key)
    flip_horizontal = jax.random.bernoulli(key1, 0.5)

    # Use jax.lax.cond for JAX-compatible conditional execution
    def flip_image(img):
        return jnp.flip(img, axis=1)

    def no_flip(img):
        return img

    new_image = jax.lax.cond(
        flip_horizontal,
        flip_image,
        no_flip,
        element.data["image"]
    )
    return element.update_data({"image": new_image})


# Create some dummy data (28x28 images)
num_samples = 1000
image_shape = (28, 28, 1)
data = {
    "image": np.random.randint(0, 255, (num_samples, *image_shape)).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

# Create the data source using config-based API
# MemorySource is a standard DataSource implementation for in-memory data
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

# Create operators using the unified ElementOperator API
# Normalizer is deterministic (no random key needed)
normalizer_config = ElementOperatorConfig(stochastic=False)
normalizer = ElementOperator(normalizer_config, fn=normalize, rngs=nnx.Rngs(0))

# Augmenter is stochastic: requires stream_name for proper RNG management
augmenter_config = ElementOperatorConfig(stochastic=True, stream_name="augmentations")
augmenter = ElementOperator(augmenter_config, fn=apply_augmentation, rngs=nnx.Rngs(42))

# Create the data pipeline using the DAG-based API
# from_source() initializes the pipeline with:
# 1. The DataSourceNode (data loading)
# 2. A BatchNode (automatic batching)
# The >> operator chains transformation operators
pipeline = (
    from_source(source, batch_size=32)
    >> OperatorNode(normalizer)
    >> OperatorNode(augmenter)
)

# Alternative: Method Chaining
# You can also build the pipeline using the fluent .add() method:
# pipeline = (
#     from_source(source, batch_size=32)
#     .add(OperatorNode(normalizer))
#     .add(OperatorNode(augmenter))
# )

# Create an iterator and process batches
# The pipeline handles data streaming, batching, state management, and execution
for i, batch in enumerate(pipeline):
    if i >= 3:
        break

    # Get the shape and stats for each component in the batch
    # batch['key'] provides direct access to data arrays
    image_batch = batch["image"]
    label_batch = batch["label"]

    print(f"Batch {i}:")
    print(f"  Image shape: {image_batch.shape}")
    print(f"  Label batch size: {label_batch.shape[0]}")
    print(f"  Image min/max: {image_batch.min():.3f}/{image_batch.max():.3f}")

print("Pipeline processing completed!")
```

### Advanced Pipeline

For more complex workflows, Datarax supports branching and parallel execution:

```python
from datarax.dag.nodes import Parallel, Merge, Branch

# Define additional operators
def invert(element, key=None):
    return element.update_data({"image": 1.0 - element.data["image"]})

def is_high_contrast(element):
    # Condition: check if image variance is high
    return jnp.var(element.data["image"]) > 0.1

# Build a complex DAG:
# 1. Source -> Batching
# 2. Parallel: Normal version AND Inverted version
# 3. Merge: Average them (Simple Ensemble)
# 4. Branch: Apply extra noise ONLY if high contrast, otherwise normalize again
complex_pipeline = (
    from_source(source, batch_size=32)
    >> (OperatorNode(normalizer) | OperatorNode(invert))
    >> Merge("mean")
    >> Branch(
           condition=is_high_contrast,
           true_path=OperatorNode(augmenter),
           false_path=OperatorNode(normalizer)
       )
)
```

## Documentation

For complete documentation, please visit [datarax.readthedocs.io](https://datarax.readthedocs.io).

- [Installation Guide](https://datarax.readthedocs.io/en/latest/installation/)
- [User Guide](https://datarax.readthedocs.io/en/latest/user_guide/)
- [API Reference](https://datarax.readthedocs.io/en/latest/api_reference/)
- [Examples](https://datarax.readthedocs.io/en/latest/examples/)
- [Contributing](https://datarax.readthedocs.io/en/latest/contributing/)


## Testing

Datarax uses a complete test suite with support for both CPU and GPU testing:

- Tests are organized to mirror the `src/datarax` package structure for easier navigation
- All GitHub CI workflows run tests exclusively on CPU
- Local test runs automatically use GPU when available
- The testing infrastructure handles environment configuration to ensure consistency

To run tests:

```bash
# Run all tests using CPU only (most stable)
JAX_PLATFORMS=cpu python -m pytest

# Run specific test module
JAX_PLATFORMS=cpu python -m pytest tests/sources/test_memory_source.py
```

For more information on the test organization and how to run tests, see the [Testing Guide](tests/README.md).

## License

Datarax is licensed under the [MIT License](LICENSE).
