# Datarax Type Issues and Solutions Guide

This guide documents common pyright type issues encountered when developing Datarax and their solutions. Issues are organized by context: JAX operations, Flax NNX state management, and Grain-style data handling.

## Quick Reference

| Issue | Context | Solution |
|-------|---------|----------|
| [Union type narrowing](#1-union-type-narrowing-for-element-data) | JAX/Python | Use `isinstance()` before accessing `.shape` |
| [Variable access patterns](#2-flax-nnx-variable-access-patterns) | Flax NNX | Use `[...]` slice notation or `.get_value()` |
| [JAX Array vs bool](#3-jax-array-vs-python-bool) | JAX | Explicit `bool()` conversion |
| [PRNG key creation](#4-prng-key-creation-and-type-handling) | JAX | Use `jax.random.key()` not `PRNGKey()` |
| [Module wrapping for DAG](#5-dag-node-type-requirements) | Datarax | Wrap in appropriate Node classes |
| [Batch dimension requirements](#6-batch-dimension-requirements) | JAX | Ensure arrays have shape `(N, ...)` |
| [Element/Batch PyTree access](#7-elementbatch-pytree-access) | Datarax | Type narrow after dictionary access |

---

## JAX Context Issues

### 1. Union Type Narrowing for Element Data

**Issue**: Accessing array properties on `Element.data` without type narrowing.

Datarax `Element.data` is typed as `PyTree` which can contain nested structures. When accessing values, pyright doesn't know the concrete type:

```python
# ❌ Error: Cannot access .shape on PyTree type
element = Element(data={"image": jnp.ones((28, 28))})
shape = element.data["image"].shape  # pyright error

# ❌ Also wrong: assertIsInstance doesn't narrow types
self.assertIsInstance(element.data["image"], jax.Array)
element.data["image"].shape  # Still errors
```

**Solution**: Extract the value first, then use `isinstance()` for type narrowing:

```python
# ✅ Correct: Extract and narrow
value = element.data["image"]
assert isinstance(value, jax.Array)
assert value.shape == (28, 28)  # Now pyright knows value is jax.Array

# ✅ For nested structures
nested = element.data["features"]["embedding"]
assert isinstance(nested, jax.Array)
print(f"Embedding shape: {nested.shape}")
```

**Key Insight**: Python's `isinstance()` is a type guard that pyright recognizes for narrowing. `unittest.TestCase.assertIsInstance()` does NOT provide this narrowing.

### 2. Flax NNX Variable Access Patterns

**Issue**: Incorrect access patterns for `nnx.Variable` values.

Datarax uses Flax NNX for stateful components. Variables wrap values and require specific access patterns:

```python
# ❌ Error: Cannot use Variable directly as array
class MyModule(nnx.Module):
    def __init__(self):
        self.count = nnx.Variable(0)

    def increment(self):
        self.count = self.count + 1  # Wrong: creates new Variable, doesn't update

# ❌ Error: Accessing raw Variable in arithmetic
result = self.count * 2  # May work at runtime but type-unsafe
```

**Solution**: Use slice notation `[...]` or `.get_value()`/`.set_value()` methods:

```python
# ✅ Correct: Use slice notation or get/set methods
class MyModule(nnx.Module):
    def __init__(self):
        self.count = nnx.Variable(0)

    def increment(self):
        # Option 1: Slice notation (preferred)
        self.count[...] = self.count[...] + 1

        # Option 2: get_value/set_value methods
        self.count.set_value(self.count.get_value() + 1)

# ✅ For Datarax Batch objects
batch = Batch(elements)
data = batch.data.get_value()  # Returns the PyTree
batch.data.set_value(new_data)  # Updates the Variable
```

**Variable Types in Datarax**:

The `Batch` class stores data in `nnx.Variable`:

- `batch.data` → `nnx.Variable[PyTree]` (use `.get_value()`)
- `batch.states` → `nnx.Variable[PyTree]` (use `.get_value()`)
- `batch.batch_state` → `nnx.Variable[dict]` (use `.get_value()`)

### 3. JAX Array vs Python Bool

**Issue**: JAX comparison operations return arrays, not Python booleans. The correct solution depends on the execution context.

**Context 1: Datarax Filter Predicates (Python iteration)**

Datarax filter predicates are called during Python-level iteration, NOT inside JIT. Here, `bool()` conversion works:

```python
# ✅ Correct for Datarax filter predicates
def filter_condition(element: Element) -> bool:
    """Called in Python iteration - bool() works here."""
    score = element.data.get("score")
    if score is None:
        return True
    assert isinstance(score, jax.Array)
    return bool(score > 0.5)  # Safe: not inside JIT

# Usage in Datarax pipeline
pipeline.filter(filter_condition)  # Predicate called in Python loop
```

**Context 2: Inside JIT-Traced Functions**

Inside `@jax.jit`, `@nnx.jit`, or other JAX transforms, `bool()` will raise `TracerBoolConversionError`:

```python
# ❌ FAILS inside JIT
@jax.jit
def bad_transform(x):
    if bool(jnp.sum(x) > 0):  # TracerBoolConversionError!
        return x * 2
    return x

# ✅ Use jnp.where for element-wise conditionals
@jax.jit
def good_transform(x):
    return jnp.where(x > 0, x * 2, x)

# ✅ Use jax.lax.cond for branch selection
@jax.jit
def good_branch(x):
    return jax.lax.cond(
        jnp.sum(x) > 0,  # Keep as array - lax.cond handles it
        lambda arr: arr * 2,
        lambda arr: arr,
        x
    )
```

**Context 3: Pyright Type Checking (static analysis)**

Even outside JIT, pyright may complain about the return type. Use explicit conversion:

```python
# ❌ Pyright error: Expression of type "Array" cannot be assigned to "bool"
def predicate(element: Element) -> bool:
    return jnp.mean(element.data["value"]) > 0.5

# ✅ Satisfies pyright
def predicate(element: Element) -> bool:
    value = element.data["value"]
    assert isinstance(value, jax.Array)
    return bool(jnp.mean(value) > 0.5)
```

**Summary**:

| Context | `bool()` Conversion | Alternative |
|---------|---------------------|-------------|
| Datarax filter predicates | ✅ Works | - |
| Inside `@jax.jit` | ❌ Raises error | `jax.lax.cond`, `jnp.where` |
| Inside `jax.lax.cond` | ❌ Don't convert | Keep as array |
| Pyright static check | ✅ Needed for types | - |

### 4. PRNG Key Creation and Type Handling

**Issue**: Using legacy `jax.random.PRNGKey()` instead of modern `jax.random.key()`, or incorrect handling of PRNG keys in different contexts.

JAX has two ways to create PRNG keys (see [JEP 9263](https://docs.jax.dev/en/latest/jep/9263-typed-keys.html)). The correct approach depends on context.

**Context 1: Creating New PRNG Keys**

Use `jax.random.key()` for all new key creation:

```python
# ✅ Modern pattern - creates typed key (JAX 0.4.16+)
key = jax.random.key(42)  # Returns PRNGKeyArray with key<impl> dtype

# ❌ Legacy pattern - creates uint32 array
key = jax.random.PRNGKey(42)  # Returns shape (2,) uint32 array
```

**Why modern keys are preferred**:

- Scalar shape instead of `(2,)` trailing dimension
- Carries RNG implementation info in the dtype
- Prevents accidental arithmetic operations on keys
- Better type safety with explicit key type

**Context 2: Type Annotations**

Use `jax.Array` for PRNG key type hints (PRNGKeyArray is a subclass):

```python
# ✅ Correct: Use jax.Array for key parameters
def stochastic_transform(
    element: Element,
    key: jax.Array | None = None
) -> Element:
    if key is None:
        key = jax.random.key(0)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, element.data["image"].shape)
    ...

# ❌ Avoid: PRNGKey type is deprecated for annotations
def bad_transform(key: jax.random.PRNGKey) -> ...:  # Don't use PRNGKey as type
    ...
```

**Context 3: Serialization and Checkpointing**

PRNG keys require special handling for serialization. Use `key_data()`/`wrap_key_data()`:

```python
import jax.random

# ✅ Correct: Serialize with key_data(), restore with wrap_key_data()
key = jax.random.key(42)

# Saving
key_bytes = jax.random.key_data(key)  # Extract raw array data
# key_bytes is a regular jax.Array, safe for serialization

# Restoring
restored_key = jax.random.wrap_key_data(key_bytes)  # Reconstruct typed key

# ❌ Wrong: Directly serializing the key object
# pickle.dumps(key)  # May not preserve key type information
```

Datarax checkpoint handlers use this pattern internally:

```python
# From datarax/checkpoint/handlers.py pattern
state = {
    'rng_key': jax.random.key_data(module.rngs.default.key[...]),
    'rng_count': module.rngs.default.count[...]
}
```

**Context 4: Flax NNX Modules (Recommended for Stateful Random)**

For modules that need persistent random state, use `nnx.Rngs`:

```python
# ✅ Datarax operator pattern - pass rngs to super().__init__()
from datarax.core import OperatorModule
from datarax.operators import OperatorConfig

class StochasticOperator(OperatorModule):
    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs,
        name: str | None = None,
    ):
        # DataraxModule stores self.rngs automatically
        super().__init__(config, rngs=rngs, name=name)
        # No need for self.rngs = rngs - base class already handles it!

    def apply(self, batch: Batch, key: jax.Array | None = None) -> Batch:
        # Access self.rngs from base class
        dropout_key = self.rngs.dropout()  # Get key, auto-advance state
        noise_key = self.rngs.noise()      # Separate stream
        ...
```

**Important**: Datarax's `DataraxModule` stores `self.rngs = rngs` in its `__init__`, so subclasses should NOT duplicate this assignment. The Flax NNX base `nnx.Module.__init__()` takes no parameters.

```python
# Creating nnx.Rngs - accepts int or jax.Array
rngs = nnx.Rngs(default=42)           # From int seed
rngs = nnx.Rngs(default=jax.random.key(42))  # From typed key

# Using named streams for different random operations
rngs = nnx.Rngs(dropout=42, noise=123)  # Multiple named streams
```

Flax NNX internally converts legacy keys to typed keys, so both work.

**Context 5: Legacy Compatibility**

When interfacing with code that uses `PRNGKey()`, both key types work with JAX random functions:

```python
# Both work with jax.random functions
legacy_key = jax.random.PRNGKey(42)
modern_key = jax.random.key(42)

# JAX handles both transparently
jax.random.normal(legacy_key, (10,))  # Works
jax.random.normal(modern_key, (10,))  # Works

# ✅ Converting legacy to modern (if needed)
modern_key = jax.random.wrap_key_data(
    jax.random.key_data(legacy_key)
)
```

**Summary**:

| Context | Recommended Approach | Notes |
|---------|---------------------|-------|
| Creating keys | `jax.random.key(seed)` | Modern typed key |
| Type annotations | `key: jax.Array \| None` | PRNGKeyArray is subclass |
| Serialization | `key_data()` / `wrap_key_data()` | Required for checkpoints |
| NNX modules | `nnx.Rngs` | Managed state, auto-advance |
| Legacy interop | Both types work | JAX handles transparently |

**Datarax Convention**: Use `key: jax.Array | None = None` for optional PRNG arguments. For NNX modules, prefer `nnx.Rngs` for managed random state with automatic key advancement.

---

## Datarax-Specific Issues

### 5. DAG Node Type Requirements

**Issue**: Passing raw modules to DAG executor instead of wrapped nodes.

The DAG system requires specific node wrapper types:

```python
from datarax.sources import MemorySource
from datarax.dag import DAGExecutor

# ❌ Error: MemorySource is not assignable to Node
executor = DAGExecutor()
executor.add(MemorySource(data))  # Type error
```

**Solution**: Wrap modules in appropriate Node classes:

```python
from datarax.dag.nodes import (
    DataSourceNode, OperatorNode, BatchNode,
    SamplerNode, SharderNode, CacheNode,
    PrefetchNode, ShuffleNode
)

# ✅ Correct: Use node wrappers
executor = DAGExecutor()
executor.add(DataSourceNode(MemorySource(data)))
executor.add(OperatorNode(MyTransform(config)))
executor.add(BatchNode(batcher))
```

**Node Class Reference**:

| Module Type | Node Wrapper | Purpose |
|-------------|--------------|---------|
| `DataSourceModule` | `DataSourceNode` | Data sources (Memory, TFDS, HF) |
| `OperatorModule` | `OperatorNode` | Transformations |
| `BatcherModule` | `BatchNode` | Batching operations |
| `SamplerModule` | `SamplerNode` | Index sampling |
| `SharderModule` | `SharderNode` | Data sharding |
| N/A (built-in) | `CacheNode` | Result caching |
| N/A (built-in) | `PrefetchNode` | Async prefetching |
| N/A (built-in) | `ShuffleNode` | Shuffling |

### 6. Batch Dimension Requirements

**Issue**: Scalar arrays where batch dimension is expected.

Datarax batch operations expect arrays with explicit batch dimensions:

```python
# ❌ Error: Scalar has no batch dimension
element = Element(data={"value": jnp.array(5)})  # Shape ()
batch = Batch([element, element])  # Will fail to stack

# ❌ Error: Inconsistent batch dimensions in Batch.from_parts
data = {
    "image": jnp.ones((32, 224, 224, 3)),
    "label": jnp.ones((16,))  # Wrong batch size!
}
batch = Batch.from_parts(data, states={})  # Validation error
```

**Solution**: Ensure all arrays have consistent batch dimensions:

```python
# ✅ Correct: Use explicit dimensions
element = Element(data={"value": jnp.array([5])})  # Shape (1,)

# ✅ Correct: Consistent batch dimensions
data = {
    "image": jnp.ones((32, 224, 224, 3)),  # (batch, H, W, C)
    "label": jnp.ones((32,)),               # (batch,)
    "mask": jnp.ones((32, 224, 224))        # (batch, H, W)
}
states = {"count": jnp.zeros((32,))}        # (batch,)
batch = Batch.from_parts(data, states)
```

**Validation**: `Batch.from_parts(..., validate=True)` checks:

- All data arrays have same batch size (axis 0)
- All state arrays have same batch size
- Metadata list length matches batch size

### 7. Element/Batch PyTree Access

**Issue**: Type errors when accessing nested PyTree structures.

`Element.data` is typed as `PyTree` and `Batch.data` is an `nnx.Variable` wrapping a PyTree. Both can contain arbitrarily nested structures, causing type errors on access:

```python
# ❌ Error: Nested access without type narrowing
element = Element(data={"features": {"visual": jnp.ones((224, 224, 3))}})
shape = element.data["features"]["visual"].shape  # Multiple type errors

# ❌ Error: Batch.data is nnx.Variable, not dict
batch = Batch(elements)
batch.data["image"]  # Wrong: data is Variable, not dict
```

**Solution for Element**: Extract and narrow at each level, or use `jax.tree.map`:

```python
# ✅ Option 1: Step-by-step narrowing
features = element.data["features"]
assert isinstance(features, dict)
visual = features["visual"]
assert isinstance(visual, jax.Array)
print(visual.shape)

# ✅ Option 2: Use jax.tree.map for transformations
def normalize(x):
    if isinstance(x, jax.Array):
        return x / 255.0
    return x

normalized_data = jax.tree.map(normalize, element.data)
```

**Solution for Batch**: Use the provided access methods:

```python
batch = Batch(elements)

# ✅ Option 1: Dict-like access for flat structures (most common)
image = batch["image"]  # Uses __getitem__, returns jax.Array directly
label = batch["label"]

# ✅ Option 2: get_data() convenience method
data_dict = batch.get_data()  # Returns PyTree, same as batch.data.get_value()

# ✅ Option 3: Direct Variable access for NNX compatibility
data_pytree = batch.data.get_value()  # Returns PyTree
# Or with slice notation:
data_pytree = batch.data[...]

# ✅ Option 4: jax.tree.map for nested PyTree operations
def get_shape(x):
    return x.shape if isinstance(x, jax.Array) else None

shapes = jax.tree.map(get_shape, batch.get_data())
```

**Batch Access Pattern Reference**:

| Pattern | Returns | Use When |
|---------|---------|----------|
| `batch["key"]` | `jax.Array` | Simple flat dict access |
| `batch.get_data()` | `PyTree` | Need full data dict |
| `batch.data.get_value()` | `PyTree` | NNX-style, same as get_data() |
| `batch.data[...]` | `PyTree` | NNX slice notation |
| `batch.get_states()` | `PyTree` | Access state arrays |
| `batch.get_element(i)` | `Element` | Extract single element |

---

## Flax NNX Context Issues

### 8. Module State in JIT Compilation

**Issue**: Mutable state captured in closures causes tracer errors.

NNX modules have mutable state that must be handled carefully with JAX transforms:

```python
# ❌ Error: Mutable module in closure
class Counter(nnx.Module):
    def __init__(self):
        self.count = nnx.Variable(0)

counter = Counter()

@jax.jit
def increment():
    counter.count[...] += 1  # Captured mutable state!
    return counter.count[...]
```

**Solution**: Pass modules as arguments, use NNX transforms:

```python
# ✅ Correct: Use nnx.jit and pass module as argument
@nnx.jit
def increment(counter: Counter):
    counter.count[...] += 1
    return counter.count[...]

result = increment(counter)

# ✅ For Datarax operators
class MyOperator(OperatorModule):
    def apply(self, batch: Batch, key: jax.Array | None = None) -> Batch:
        # State updates happen through self.* Variables
        self.call_count[...] += 1
        return batch
```

### 9. Custom Variable Types and Filtering

**Issue**: Type errors when filtering state by Variable type.

NNX uses Variable subclasses for state categorization:

```python
# ❌ Error: Filter returns Union type
state = nnx.state(model)
params = state.filter(nnx.Param)  # Type is State, not narrowed

# ❌ Error: Wrong filter order (subclass after superclass)
params, batch_stats = state.split(nnx.Param, nnx.BatchStat)
# If a variable is subclass of Param, it matches Param first!
```

**Solution**: Use explicit filtering with proper ordering:

```python
# ✅ Correct: Filter returns typed State
state = nnx.state(model)
params: nnx.State = state.filter(nnx.Param)

# ✅ Correct: Most specific filters first
class SpecialParam(nnx.Param): pass

# Filter order: specific → general
special, regular = state.split(SpecialParam, nnx.Param)

# ✅ For Datarax modules
from datarax.core import DataraxModule

class MyModule(DataraxModule):
    def __init__(self):
        super().__init__()
        self.weight = nnx.Param(jnp.ones((10, 10)))
        self.running_mean = nnx.BatchStat(jnp.zeros((10,)))

    def get_trainable_state(self) -> nnx.State:
        return nnx.state(self).filter(nnx.Param)
```

---

## Grain-Style Data Handling

### 10. RandomAccessDataSource Protocol

**Issue**: Custom data sources missing required protocol methods.

Datarax sources follow Grain's `RandomAccessDataSource` protocol:

```python
# ❌ Error: Missing __len__ or __getitem__
class MySource:
    def __init__(self, data):
        self.data = data

    def get_item(self, idx):  # Wrong method name
        return self.data[idx]
```

**Solution**: Implement the full protocol:

```python
from typing import Protocol, Generic, TypeVar

T = TypeVar('T')

# ✅ Correct: Full protocol implementation
class MySource(Generic[T]):
    def __init__(self, data: list[T]):
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> T:
        return self._data[index]

    # Optional: For batched reads
    def _getitems(self, indices: list[int]) -> list[T]:
        return [self._data[i] for i in indices]
```

### 11. Transform Return Types

**Issue**: Transforms returning wrong types.

Datarax transforms must return the same type they receive:

```python
# ❌ Error: Returns dict instead of Element
def my_transform(element: Element) -> Element:
    return element.data  # Returns dict, not Element!

# ❌ Error: Modifies in place (Elements are immutable)
def bad_transform(element: Element) -> Element:
    element.data["new_key"] = value  # Element is immutable!
    return element
```

**Solution**: Use Element's update methods:

```python
# ✅ Correct: Use Element.update_data() or Element.replace()
def my_transform(element: Element) -> Element:
    new_value = process(element.data["image"])
    return element.update_data({"processed": new_value})

# ✅ For state updates
def stateful_transform(element: Element) -> Element:
    return element.update_state({"processed": True})

# ✅ For complete replacement
def replace_transform(element: Element) -> Element:
    return element.replace(
        data={"new": jnp.ones((10,))},
        state={"flag": True}
    )
```

---

## Datarax Module Hierarchy

Understanding the type hierarchy helps with proper annotations:

```text
DataraxModule (base, extends nnx.Module)
├── OperatorModule              # Parametric transformations
│   ├── ElementOperator         # Per-element transforms
│   └── BatchMixOperator        # Cross-element transforms
├── StructuralModule            # Non-parametric processors
│   ├── BatcherModule           # Batch creation
│   ├── SamplerModule           # Index sampling
│   └── SharderModule           # Data sharding
└── DataSourceModule            # Data sources
    ├── MemorySourceModule      # In-memory data
    ├── TfdsSourceModule        # TensorFlow Datasets
    └── HfSourceModule          # HuggingFace Datasets
```

---

## Type Checking Best Practices

### For Test Files

1. **Always narrow before accessing array properties**:

   ```python
   value = result["key"]
   assert isinstance(value, jax.Array)
   assert value.shape == expected_shape
   ```

2. **Use pytest over unittest** for better type inference

3. **Add `# type: ignore` sparingly** with explanatory comments

### For Implementation Files

1. **Use explicit type annotations** on public APIs:

   ```python
   def process(
       data: dict[str, jax.Array],
       key: jax.Array | None = None
   ) -> dict[str, jax.Array]:
   ```

2. **Import from datarax.typing** for consistency:

   ```python
   from datarax.typing import Element, Batch, PRNGKey, DataDict
   ```

3. **Use Protocol types** for interface definitions

4. **Document Variable access patterns** in docstrings

---

## Related Documentation

**Datarax Documentation**:

- [Contributing Guide](contributing_guide.md) - Code style requirements
- [Developer Guide](dev_guide.md) - Development setup and practices

**External Resources**:

- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html) - Official NNX guide
- [JAX Documentation](https://docs.jax.dev/en/latest/) - JAX type system and patterns
- [Grain Documentation](https://github.com/google/grain) - Data loading patterns
