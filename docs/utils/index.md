# Utilities

Utility modules providing common functionality across Datarax. These are
low-level helpers used internally and available for advanced use cases.

## Available Utilities

| Utility | Purpose | Key Functions |
|---------|---------|---------------|
| **External** | Library adapters | Integration helpers |
| **Cache** | Dataset cache layout | Resolve/structure cache dirs |
| **Multirate** | Signal alignment | Align streams at different rates |

!!! note "Key points"

    - Batch operations live in the core layer: [batch_ops](../core/batch_ops.md)
    - Randomness helpers live in the core layer: [prng](../core/prng.md)

## Modules

- [external](external.md) - External library adapters and integrations
- [cache](cache.md) - Dataset cache-layout helpers
- [multirate](multirate.md) - Multirate signal-alignment helpers

## See Also

- [Types & Protocols](../root/index.md) - Type definitions
- [JAX Documentation](https://jax.readthedocs.io/) - JAX PyTrees
