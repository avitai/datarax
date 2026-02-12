# Error Messages Guide

Common Datarax errors, what causes them, and how to fix them.

## Configuration Errors

### `ValueError: Stochastic modules require stream_name for RNG management`

**Cause**: You set `stochastic=True` without providing a `stream_name`.

```python
# Wrong
config = OperatorConfig(stochastic=True)

# Fix: add stream_name
config = OperatorConfig(stochastic=True, stream_name="augment")
```

### `ValueError: Deterministic modules should not specify stream_name`

**Cause**: You provided a `stream_name` but left `stochastic=False`.

```python
# Wrong
config = OperatorConfig(stochastic=False, stream_name="augment")

# Fix: either remove stream_name or set stochastic=True
config = OperatorConfig(stochastic=False)
# or
config = OperatorConfig(stochastic=True, stream_name="augment")
```

### `ValueError: batch_strategy must be 'vmap' or 'scan'`

**Cause**: Invalid `batch_strategy` value.

```python
# Wrong
config = OperatorConfig(batch_strategy="parallel")

# Fix: use "vmap" or "scan"
config = OperatorConfig(batch_strategy="vmap")   # parallel, default
config = OperatorConfig(batch_strategy="scan")    # sequential, low memory
```

### `FrozenInstanceError: Cannot modify frozen StructuralConfig`

**Cause**: Attempting to modify a `StructuralConfig` after initialization. Structural configs are frozen (immutable) after `__post_init__` completes.

```python
# Wrong
config = StructuralConfig(stochastic=True, stream_name="sample")
config.stochastic = False  # FrozenInstanceError!

# Fix: create a new config instead
config = StructuralConfig(stochastic=False)
```

### `ValueError: Cannot specify both batch_stats_fn and precomputed_stats`

**Cause**: These are mutually exclusive -- use either dynamic computation or static values.

```python
# Wrong
config = DataraxModuleConfig(
    batch_stats_fn=my_fn,
    precomputed_stats={"mean": 0.5},
)

# Fix: choose one
config = DataraxModuleConfig(batch_stats_fn=my_fn)
# or
config = DataraxModuleConfig(precomputed_stats={"mean": 0.5})
```

## Operator Errors

### `NotImplementedError: must implement apply()`

**Cause**: You subclassed `OperatorModule` but didn't implement the `apply()` method.

```python
# Wrong
class MyOp(OperatorModule):
    pass  # Missing apply()

# Fix: implement apply
class MyOp(OperatorModule):
    def apply(self, element, *, rngs=None):
        return element  # your logic here
```

## Memory Errors

### XLA OOM during vmap

**Symptom**: Warning like `Allocation of X exceeds 90% of memory` or XLA OOM error.

**Cause**: `batch_strategy="vmap"` materializes all batch elements in parallel, which can exceed GPU memory for large operators (e.g., CREPE CNN, synthesis chains).

```python
# Problem: 16 parallel forward passes = 16x memory
config = OperatorConfig(batch_strategy="vmap")

# Fix: use scan for sequential processing (O(1) memory)
config = OperatorConfig(batch_strategy="scan")
```

!!! tip
    `scan` processes elements one at a time while keeping full JIT compilation
    benefits. It's slower but uses constant memory regardless of batch size.

## Source Errors

### `ValueError: num_records is required`

**Cause**: `EpochAwareSamplerConfig` requires `num_records` to be set.

```python
# Wrong
config = EpochAwareSamplerConfig()

# Fix: provide num_records
config = EpochAwareSamplerConfig(num_records=1000)
```

### `ValueError: num_records must be positive`

**Cause**: `num_records` must be > 0.

```python
# Wrong
config = EpochAwareSamplerConfig(num_records=0)

# Fix
config = EpochAwareSamplerConfig(num_records=100)
```

### `ValueError: Cannot determine length for infinite epochs`

**Cause**: Calling `len()` on a sampler with `num_epochs=-1` (infinite).

```python
sampler = EpochAwareSamplerModule(
    EpochAwareSamplerConfig(num_records=100, num_epochs=-1),
    rngs=nnx.Rngs(0),
)
len(sampler)  # ValueError!

# Fix: don't call len() on infinite samplers, or use finite epochs
sampler = EpochAwareSamplerModule(
    EpochAwareSamplerConfig(num_records=100, num_epochs=10),
    rngs=nnx.Rngs(0),
)
print(len(sampler))  # 1000
```

## Import Errors

### `ModuleNotFoundError: No module named 'tensorflow_datasets'`

**Cause**: TFDS sources require `tensorflow-datasets` to be installed.

```bash
uv pip install tensorflow-datasets
```

### `ModuleNotFoundError: No module named 'datasets'`

**Cause**: HuggingFace sources require the `datasets` library.

```bash
uv pip install datasets
```

## Debugging Tips

1. **Check config validation**: Most errors are caught at config construction. Create configs early to fail fast.
2. **Use `batch_strategy="scan"` for OOM**: If an operator causes memory issues, switch to scan before reducing batch size.
3. **Check `stream_name` consistency**: Each stochastic operator needs a unique `stream_name` for reproducible RNG streams.
