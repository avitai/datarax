# Datarax: High-Performance JAX Data Pipelines

[![CI](https://github.com/avitai/datarax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/ci.yml)
[![Test Coverage](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml)
[![codecov](https://codecov.io/gh/avitai/datarax/branch/main/graph/badge.svg)](https://codecov.io/gh/avitai/datarax)

Datarax is a high-performance, extensible data pipeline framework specifically
engineered for JAX-based machine learning workflows. It simplifies and accelerates
the development of efficient and scalable data loading, preprocessing, and
augmentation pipelines for JAX, leveraging the full potential of JAX's
Just-In-Time (JIT) compilation, automatic differentiation, and hardware
acceleration capabilities.

## Key Features

- **High Performance:** Leverages JAX's JIT compilation and XLA backend to
achieve near-optimal data processing speeds on CPUs, GPUs, and TPUs.
- **JAX-Native Design:** All core components and operations are designed with JAX's
functional programming paradigm and immutable data structures (PyTrees) in mind.
- **Scalability:** Supports efficient data loading and processing for large datasets
and distributed training scenarios, including multi-host and multi-device setups.
- **Extensibility:** Easily define and integrate custom data sources, transformations,
and augmentation operations.
- **Usability:** Provides a clear, intuitive Python API and a flexible configuration
system for defining and managing pipelines.
- **Determinism:** Pipeline runs are deterministic by default, crucial for reproducibility.
- **Complete Feature Set:** Supports common operators, advanced transformations,
batching, sharding, checkpointing, and caching.
- **Ecosystem Integration:** Facilitates smooth integration with other JAX libraries
like Flax, Optax, and Orbax.

## Quick Navigation

- [Batching](batching/index.md) - Batch creation and management
- [Benchmarking](benchmarking/index.md) - Performance measurement tools
- [Checkpoint](checkpoint/index.md) - State persistence and recovery
- [Command Line Interface](cli/index.md) - CLI tools
- [Config](config/index.md) - Configuration management
- [Control](control/index.md) - Pipeline control flow
- [Core Components](core/index.md) - Core abstractions
- [DAG](dag/index.md) - Directed acyclic graph execution
- [Distributed](distributed/index.md) - Multi-device processing
- [Memory](memory/index.md) - Memory management
- [Monitoring](monitoring/index.md) - Metrics and observability
- [Operators](operators/index.md) - Data transformation operators
- [Performance](performance/index.md) - Performance optimization
- [Root](root/index.md) - Root module types
- [Samplers](samplers/index.md) - Data sampling strategies
- [Sharding](sharding/index.md) - Data sharding
- [Sources](sources/index.md) - Data source adapters
- [Utilities](utils/index.md) - Utility functions

## Installation

```bash
# Install via uv (recommended)
uv pip install datarax

# Install with optional dependencies
uv pip install datarax[data]     # For data loading (HF, TFDS)
uv pip install datarax[gpu]      # For GPU support

# Or locally for development
pip install -e .
```

## Quick Start

Here's a simple example of using Datarax's DAG-based architecture:

```python
import jax
import jax.numpy as jnp
from flax import nnx
from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig

# 1. Define operations
def normalize(element, key=None):
    return element.update_data({"image": element.data["image"] / 255.0})

# 2. Create data source
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=my_data_dict, rngs=nnx.Rngs(0))

# 3. Create operators
normalizer = ElementOperator(
    ElementOperatorConfig(),
    fn=normalize,
    rngs=nnx.Rngs(0)
)

# 4. Build pipeline
pipeline = (
    from_source(source, batch_size=32)
    >> OperatorNode(normalizer)
)

# 5. Run pipeline
for batch in pipeline:
    process(batch)
```

## Documentation Structure

- **API Reference** - Complete API documentation
- **Module Documentation** - Detailed documentation for each module
- **Examples** - Usage examples and tutorials
- **Migration Guides** - Guides for migrating between versions

## Contributing

To contribute to the documentation:

1. Add docstrings to your Python code
2. Run the documentation generator: `python scripts/generate_docs.py`
3. Build the documentation: `mkdocs build`
4. Preview locally: `mkdocs serve`
