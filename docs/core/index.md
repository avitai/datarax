# Core Components

Core abstractions and building blocks that form the foundation of Datarax pipelines. These modules define the protocols, base classes, and data structures used throughout the framework.

## Component Overview

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Element & Batch** | Data containers | `Element`, `Batch`, `Metadata` |
| **Config** | Typed configuration | `OperatorConfig`, `StructuralConfig` |
| **Modules** | Base abstractions | `DataraxModule`, `OperatorModule` |
| **Protocols** | Interface contracts | `DataSourceModule`, `SamplerModule` |

`★ Insight ─────────────────────────────────────`

- **Element** wraps a single data sample with state and metadata
- **Batch** is a dictionary of batched JAX arrays
- All modules inherit from `DataraxModule` for consistent behavior
- Protocols enable duck-typing with `isinstance()` checks

`─────────────────────────────────────────────────`

## Architecture

```
DataraxModule (base)
├── OperatorModule      → Transformations (learnable)
├── DataSourceModule    → Data loading
├── BatcherModule       → Batching logic
├── SamplerModule       → Index sampling
└── SharderModule       → Device sharding
```

## Quick Start

```python
from datarax.core import Element, Batch
from datarax.core.config import OperatorConfig

# Create an element
element = Element(
    data={"image": jnp.zeros((32, 32, 3))},
    state={"step": 0},
    metadata=Metadata(index=0),
)

# Access and update immutably
new_element = element.replace(
    data={"image": element.data["image"] / 255.0}
)
```

## Modules

### Data Structures

- [element_batch](element_batch.md) - `Element` and `Batch` data containers
- [metadata](metadata.md) - Metadata handling and field selection

### Configuration

- [config](config.md) - Configuration base classes and validation

### Base Classes

- [module](module.md) - `DataraxModule` base class
- [operator](operator.md) - `OperatorModule` for transformations
- [data_source](data_source.md) - `DataSourceModule` for data loading
- [batcher](batcher.md) - `BatcherModule` for batch creation
- [sampler](sampler.md) - `SamplerModule` for index sampling
- [sharder](sharder.md) - `SharderModule` for device sharding

### Specialized

- [cross_modal](cross_modal.md) - Cross-modal data handling
- [modality](modality.md) - Data modality definitions (image, text, audio)
- [structural](structural.md) - Structural utilities and patterns

## See Also

- [Types & Protocols](../root/index.md) - Type definitions
- [Configuration Guide](config.md) - Detailed config documentation
- [DAG Executor](../dag/dag_executor.md) - Using core components in pipelines
