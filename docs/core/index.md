# Core Components

Core abstractions and building blocks that form the foundation of Datarax pipelines. These modules define the protocols, base classes, and data structures used throughout the framework.

## Component Overview

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| **Element & Batch** | Data containers | `Element`, `Batch`, `Maybe`, `batch_ops` |
| **Config** | Typed configuration | `OperatorConfig`, `StructuralConfig` |
| **Modules** | Base abstractions | `DataraxModule`, `OperatorModule` |
| **Protocols** | Interface contracts | `DataSourceModule`, `SamplerModule` |

!!! note "Key points"

    - **Element** is one record: its values, its state and its identity (index, epoch, draw)
    - **Batch** holds records along a leading axis; `batch["image"]` reads its data
    - **Maybe** is a field a record may lack: its values and a `present` flag per record
    - **batch_ops** builds, slices, regroups and pads batches, as pure functions
    - All modules inherit from `DataraxModule` for consistent behavior
    - Protocols enable duck-typing with `isinstance()` checks

## Architecture

```
DataraxModule (base)
├── OperatorModule          → Transformations (learnable)
└── StructuralModule        → Non-parametric structural processors
    ├── DataSourceModule    → Data loading
    ├── BatcherModule       → Batching logic
    └── SamplerModule       → Index sampling
```

## Quick Start

```python
import jax.numpy as jnp
from datarax.core import Element, batch_ops

# A record, and an immutable update of it
element = Element({"image": jnp.zeros((32, 32, 3))}, state={"step": jnp.int32(0)})
new_element = element.update_data({"image": element.data["image"] / 255.0})

# Records stacked into a batch; each row is keyed by its identity
batch = batch_ops.from_stacked(batch_ops.stack([element, new_element]))
assert batch["image"].shape == (2, 32, 32, 3)
```

## Modules

### Data Structures

- [element_batch](element_batch.md) - `Element` and `Batch` data containers
- [index_words](index_words.md) - 64-bit record indices as two uint32 words
- [batch_ops](batch_ops.md) - Building, slicing, regrouping and padding batches
- [maybe](maybe.md) - Missing (`Maybe`), masked (`MASKED`) and padded values
- [state_keys](state_keys.md) - The per-record state entries datarax writes
- [spec](spec.md) - Element specs: data as given or as JAX arrays, and batch validation
- [prng](prng.md) - The named `nnx.Rngs` streams and per-record key derivation

### Configuration

- [config](config.md) - Configuration base classes and validation

### Base Classes

- [module](module.md) - `DataraxModule` base class
- [operator](operator.md) - `OperatorModule` for transformations
- [data_source](data_source.md) - `DataSourceModule` for data loading
- [batcher](batcher.md) - `BatcherModule` for batch creation
- [sampler](sampler.md) - `SamplerModule` for index sampling

### Specialized

- [cross_modal](cross_modal.md) - Cross-modal data handling
- [modality](modality.md) - Base classes for single-modality (per-field) operators
- [structural](structural.md) - Structural utilities and patterns

## Real-World Examples

- [DADA Learned Augmentation](../examples/advanced/differentiable/dada-learned-augmentation.md) - `nnx.Param` and `nnx.value_and_grad` through augmentation operators for policy search
- [Learned ISP Pipeline](../examples/advanced/differentiable/learned-isp-guide.md) - Custom `ModalityOperator` subclasses with learnable parameters for ISP stages
- [DDSP Audio Synthesis](../examples/advanced/differentiable/ddsp-audio-synthesis.md) - Custom `OperatorModule` subclasses for audio, showcasing extensibility to any domain

## See Also

- [Types & Protocols](../root/index.md) - Type definitions
- [Configuration Guide](config.md) - Detailed config documentation
- DAG Executor - Using core components in pipelines
