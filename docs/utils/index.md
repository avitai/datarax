# Utilities

Utility modules providing common functionality across Datarax. These are low-level helpers used internally and available for advanced use cases.

## Available Utilities

| Utility | Purpose | Key Functions |
|---------|---------|---------------|
| **PRNG** | Random numbers | Seed management, key splitting |
| **PyTree Utils** | Tree operations | Flatten, unflatten, map |
| **External** | Library adapters | Integration helpers |

`★ Insight ─────────────────────────────────────`

- PRNG utilities wrap JAX random for consistent seeding
- PyTree utils handle nested data structures
- Use these for custom operators and extensions

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.utils import create_rng_key, split_key
from datarax.utils.pytree_utils import tree_map_with_path

# Create and split RNG keys
key = create_rng_key(42)
key1, key2 = split_key(key, 2)

# Map over PyTree with path info
def scale_images(path, value):
    if "image" in path:
        return value / 255.0
    return value

scaled = tree_map_with_path(scale_images, data)
```

## Modules

- [prng](prng.md) - Pseudo-random number generation utilities for JAX
- [pytree_utils](pytree_utils.md) - PyTree manipulation and transformation
- [external](external.md) - External library adapters and integrations

## PyTree Operations

```python
from datarax.utils.pytree_utils import (
    flatten_dict,
    unflatten_dict,
    tree_select,
)

# Flatten nested dict
flat = flatten_dict({"a": {"b": 1, "c": 2}})
# {"a.b": 1, "a.c": 2}

# Select subset
subset = tree_select(data, keys=["image", "label"])
```

## See Also

- [Types & Protocols](../root/index.md) - Type definitions
- [JAX Documentation](https://jax.readthedocs.io/) - JAX PyTrees
