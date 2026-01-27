# Pyright Type Issues and Solutions Guide

This document captures common pyright type issues encountered in the Datarax codebase and their solutions.

## Common Pyright Issues and Their Solutions

### 1. Type Narrowing for Union Types (Most Common)

**Issue**: Accessing properties on union types without type narrowing

```python
# Element type is dict[str, ArrayLike | dict[str, ArrayLike]]
result["key"].shape  # Error: dict[str, ArrayLike] doesn't have .shape
```

**Solution**: Use isinstance() checks for type narrowing

```python
# Extract value first, then narrow type
value = result["key"]
assert isinstance(value, (jax.Array, jnp.ndarray))
assert value.shape == (32, 32)  # Now safe to access .shape
```

**Key Insight**: pyright needs explicit type narrowing for union types. The `isinstance()` check provides this narrowing, while `hasattr()` or unittest's `assertIsInstance()` do NOT provide type narrowing for pyright.

### 2. Unittest vs Pytest

**Issue**: Tests using unittest.TestCase instead of pytest

```python
# Old pattern - unittest
class TestModule(unittest.TestCase):
    def test_something(self):
        self.assertTrue(condition)
        self.assertIsInstance(obj, type)  # Doesn't provide type narrowing!
```

**Solution**: Convert to pytest

```python
# New pattern - pytest
class TestModule:
    def test_something(self):
        assert condition
        assert isinstance(obj, type)  # Provides type narrowing!
```

### 3. Module Type Hierarchies

**Issue**: Incorrect types in DAG operations

```python
# Error: MockDataSource is not assignable to Node
executor.add(MockDataSource(data))
```

**Solution**: Wrap modules in appropriate Node classes

```python
from datarax.dag.nodes import DataSourceNode, OperatorNode

# Correct: Wrap data source in DataSourceNode
executor.add(DataSourceNode(MockDataSource(data)))

# For operators, use OperatorNode
executor.parallel([OperatorNode(MyOperator(config))])
```

### 4. Batch Dimension Requirements

**Issue**: Scalar arrays when batch dimension expected

```python
# Error: apply_batch expects arrays with batch dimension
element = {"value": jnp.array(5)}  # Scalar - no dimensions
```

**Solution**: Ensure arrays have at least one dimension

```python
element = {"value": jnp.array([5])}  # Shape (1,) - has batch dimension
```

### 5. Condition Functions Returning Arrays Instead of Bool

**Issue**: Condition functions returning JAX arrays instead of Python bool

```python
def condition(x):
    return jnp.mean(x["value"]) > 0  # Returns Array, not bool
```

**Solution**: Explicitly convert to bool

```python
def condition(x):
    value = x["value"]
    assert isinstance(value, (jax.Array, jnp.ndarray))
    return bool(jnp.mean(value) > 0)  # Explicitly convert to bool
```

## Datarax Module Hierarchy

Understanding the module hierarchy helps with proper type annotations:

```text
DataraxModule (base)
├── OperatorModule         # Parametric, differentiable transformations
│   └── ModalityOperator   # Modality-specific operators (image, text, etc.)
├── StructuralModule       # Non-parametric structural processors
│   ├── BatcherModule      # Batch creation
│   ├── SamplerModule      # Index sampling
│   └── SharderModule      # Data sharding
└── CheckpointableIteratorModule  # Data sources with checkpoint support
```

## DAG Node Classes

For DAG operations, use the appropriate node wrapper classes:

| Purpose       | Node Class       | Module Type                            |
|---------------|------------------|----------------------------------------|
| Data sources  | `DataSourceNode` | Any data source                        |
| Operators     | `OperatorNode`   | `OperatorModule` or `ModalityOperator` |
| Batching      | `BatchNode`      | `BatcherModule`                        |
| Sampling      | `SamplerNode`    | `SamplerModule`                        |
| Sharding      | `SharderNode`    | `SharderModule`                        |
| Caching       | `CacheNode`      | N/A (built-in)                         |
| Prefetching   | `PrefetchNode`   | N/A (built-in)                         |
| Shuffling     | `ShuffleNode`    | N/A (built-in)                         |

## Type Checking Best Practices

### For Test Files

1. **Always add type narrowing** before accessing array properties:

   ```python
   # Pattern to follow
   array_value = result["key"]
   assert isinstance(array_value, (jnp.ndarray, jax.Array))
   # Now safe to use array_value.shape, array_value.dtype, etc.
   ```

2. **Use pytest over unittest** for better type inference

3. **Add type: ignore sparingly** with explanatory comments when intentional relaxed typing is used

### For Implementation Files

1. **Be explicit about accepted types** in docstrings even if runtime is more lenient

2. **Consider defensive programming** but document the behavior

3. **Use Protocol types** for better interface definitions

## Key Takeaways

1. **Type narrowing with isinstance() is essential** for union types

2. **Pytest provides better type inference** than unittest

3. **Wrap modules in Node classes** for DAG operations

4. **Batch dimensions are required** for JAX batch operations

5. **Explicit bool conversion** needed for condition functions

This guide should be referenced when encountering similar pyright issues in the future.
