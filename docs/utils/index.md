# Utilities

Utility modules providing common functionality across Datarax. These are
low-level helpers used internally and available for advanced use cases.

## Available Utilities

| Utility | Purpose | Key Functions |
|---------|---------|---------------|
| **PRNG** | Random numbers | `nnx.Rngs` creation from seed |
| **PyTree Utils** | Batch tree operations | Batch size, split, concatenate |
| **External** | Library adapters | Integration helpers |
| **Cache** | Dataset cache layout | Resolve/structure cache dirs |
| **Multirate** | Signal alignment | Align streams at different rates |

!!! note "Key points"

    - PRNG utilities create `nnx.Rngs` with consistent seeding
    - PyTree utils operate on `Element`/`Batch` structures
    - Use these for custom operators and extensions

## Quick Start

```python
from datarax.utils.prng import create_rngs
from datarax.utils.pytree_utils import get_batch_size

# Create named RNG streams from a seed
rngs = create_rngs(42)

# Inspect a batch produced by a pipeline
size = get_batch_size(batch)
```

## Modules

- [prng](prng.md) - `nnx.Rngs` creation utilities for JAX
- [pytree_utils](pytree_utils.md) - Batch PyTree manipulation and transformation
- [external](external.md) - External library adapters and integrations
- [cache](cache.md) - Dataset cache-layout helpers
- [multirate](multirate.md) - Multirate signal-alignment helpers

## Batch PyTree Operations

```python
from datarax.utils.pytree_utils import (
    add_batch_dimension,
    split_batch_for_devices,
    concatenate_batch_sequence,
)

# Promote a single element to a batch of size 1
batch = add_batch_dimension(element)

# Split a batch across devices, then recombine
shards = split_batch_for_devices(batch, num_splits=4)
merged = concatenate_batch_sequence(shards)
```

## See Also

- [Types & Protocols](../root/index.md) - Type definitions
- [JAX Documentation](https://jax.readthedocs.io/) - JAX PyTrees
