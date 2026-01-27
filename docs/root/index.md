# Types & Protocols

Core type definitions and protocols used throughout Datarax. These provide type safety, enable IDE autocompletion, and define the interfaces that components must implement.

## Type Categories

| Category | Types | Purpose |
|----------|-------|---------|
| **Data Containers** | `Element`, `Batch`, `Metadata` | Core data structures |
| **Dict Aliases** | `DataDict`, `StateDict`, `MetadataDict` | Dictionary type shortcuts |
| **JAX Types** | `ArrayShape`, `PRNGKey` | JAX-specific type aliases |
| **Function Types** | `ElementTransform`, `BatchTransform`, etc. | Callable signatures |
| **Protocols** | `Checkpointable`, `CheckpointableIterator` | Interface definitions |

`★ Insight ─────────────────────────────────────`

- **Type aliases** provide semantic meaning (e.g., `PRNGKey` vs `jax.Array`)
- **Protocols** use `@runtime_checkable` for `isinstance()` checks
- Function types ensure correct signatures across the codebase
- Import from `datarax.typing` for all public types

`─────────────────────────────────────────────────`

## Quick Reference

### Data Containers

```python
from datarax.typing import Element, Batch, Metadata

# Element: Single data sample with state and metadata
element = Element(
    data={"image": jnp.zeros((32, 32, 3))},
    state={"augmented": False},
    metadata=Metadata(index=0),
)

# Batch: Collection of elements (batched arrays)
batch: Batch = {"image": jnp.zeros((64, 32, 32, 3))}
```

### Dictionary Type Aliases

```python
from datarax.typing import DataDict, StateDict, MetadataDict

# DataDict: Maps field names to JAX arrays
data: DataDict = {"image": array, "label": labels}

# StateDict: Maps names to any state values
state: StateDict = {"step": 100, "learning_rate": 0.001}

# MetadataDict: Maps names to metadata values
meta: MetadataDict = {"source": "train", "index": 42}
```

### Function Types

```python
from datarax.typing import ElementTransform, BatchTransform, ArrayTransform

# ElementTransform: Element -> Element
def my_transform(element: Element) -> Element:
    return element.replace(data=process(element.data))

# BatchTransform: Batch -> Batch
def normalize(batch: Batch) -> Batch:
    return {k: v / 255.0 for k, v in batch.items()}

# ArrayTransform: jax.Array -> jax.Array
def scale(arr: jax.Array) -> jax.Array:
    return arr * 2.0
```

### Protocols

```python
from datarax.typing import Checkpointable

# Implement the Checkpointable protocol
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
from datarax.typing import PRNGKey, ArrayShape, ScanFn, CondFn

# PRNGKey: JAX random key
key: PRNGKey = jax.random.key(42)

# ArrayShape: Tuple of dimensions
shape: ArrayShape = (32, 224, 224, 3)

# ScanFn: For jax.lax.scan operations
def scan_body(carry, element) -> tuple[Any, Element]:
    return new_carry, processed_element

# CondFn: For jax.lax.cond operations
def should_augment(x) -> bool:
    return x["training"]
```

## Modules

- [typing](typing.md) - Complete API reference for all type definitions

## See Also

- [Configuration System](../core/config.md) - Config dataclasses
- [Checkpoint Handlers](../checkpoint/handlers.md) - Using Checkpointable protocol
- [Element Operator](../operators/element_operator.md) - Using Element type
