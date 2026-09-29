# Types & Protocols

Core type definitions and protocols used throughout Datarax. These provide type safety, enable IDE autocompletion, and define the interfaces that components must implement.

## Type Categories

| Category | Types | Purpose |
|----------|-------|---------|
| **Data Containers** | `Element`, `Batch` (from `datarax`) | A record and a batch of records |
| **Dict Aliases** | `DataDict` | A source's field-name-to-array mapping |
| **JAX Types** | `ArrayShape`, `PRNGKey` | JAX-specific type aliases |
| **Function Types** | `ArrayTransform` | Callable signatures |
| **Protocols** | `CheckpointableIterator` (built on substrax's `Checkpointable`) | Interface definitions |

!!! note "Key points"

    - **Type aliases** provide semantic meaning (e.g., `PRNGKey` vs `jax.Array`)
    - **Protocols** use `@runtime_checkable` for `isinstance()` checks
    - `Element` and `Batch` are classes, imported from `datarax` or `datarax.core`

## Quick Reference

### Data Containers

```python
import jax.numpy as jnp

from datarax import Batch, Element
from datarax.core import batch_ops

# Element: one record, its state and its identity
element = Element({"image": jnp.zeros((32, 32, 3))}, state={"augmented": jnp.array(False)})

# Batch: records along a leading axis, read by field name
batch: Batch = batch_ops.from_arrays({"image": jnp.zeros((64, 32, 32, 3))})
image = batch["image"]
```

### Dictionary Type Aliases

```python
from datarax.typing import DataDict

# DataDict: Maps field names to JAX arrays
data: DataDict = {"image": array, "label": labels}
```

### Function Types

```python
from datarax.typing import ArrayTransform

# ArrayTransform: jax.Array -> jax.Array
def scale(arr: jax.Array) -> jax.Array:
    return arr * 2.0
```

### Protocols

```python
from substrax.typing import Checkpointable

# Implement the Checkpointable protocol (substrax owns it; every Datarax
# module, pipeline and iterator satisfies it)
class MyModule:
    def get_state(self) -> dict[str, Any]:
        return {"data": self.data}

    def set_state(self, state: dict[str, Any]) -> None:
        self.data = state["data"]

# Runtime checking works
obj = MyModule()
assert isinstance(obj, Checkpointable)  # True!
```

## JAX-Specific Types

```python
from datarax.typing import PRNGKey, ArrayShape

# PRNGKey: JAX random key
key: PRNGKey = jax.random.key(42)

# ArrayShape: Tuple of dimensions
shape: ArrayShape = (32, 224, 224, 3)
```

## Modules

- [typing](typing.md) - Complete API reference for all type definitions

## See Also

- [Configuration System](../core/config.md) - Config dataclasses
- [Checkpoint](../checkpoint/index.md) - Using the Checkpointable protocol
- [Element Operator](../operators/element_operator.md) - Using Element type
