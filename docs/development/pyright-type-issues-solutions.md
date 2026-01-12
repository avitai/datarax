# Pyright Type Issues and Solutions Guide

!!! note "Internal Development Document"
    This document captures historical type-fixing efforts. Some class names referenced
    (e.g., `FieldTransformerModule`, `TransformerModule`) may refer to deprecated or
    renamed components. The general patterns and solutions remain applicable.

This document captures key findings from fixing pyright type issues across the Datarax codebase, particularly in test files.

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
# Correct: Wrap in DataSourceNode
executor.add(DataSourceNode(MockDataSource(data)))
# For transformers
executor.parallel([TransformNode(MockTransformer())])
```

### 4. FieldTransformerModule's Relaxed Typing

**Finding**: FieldTransformerModule intentionally accepts more relaxed types than its signature suggests.

**Design Pattern**:
- Type signature declares `Element` for type safety
- Runtime implementation checks `isinstance(data_element, dict)` and returns unchanged if not
- This defensive programming allows graceful handling of non-dict inputs

**Solution**: Use `# type: ignore[arg-type]` with explanatory comment
```python
transformer.transform(data)  # type: ignore[arg-type] # FieldTransformer accepts relaxed types
```

### 5. Batch Dimension Requirements

**Issue**: Scalar arrays when batch dimension expected
```python
# Error: transform_batch expects arrays with batch dimension
element = {"value": jnp.array(5)}  # Scalar - no dimensions
```

**Solution**: Ensure arrays have at least one dimension
```python
element = {"value": jnp.array([5])}  # Shape (1,) - has batch dimension
```

### 6. Condition Functions Returning Arrays Instead of Bool

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

## TransformerModule's Mixed Type Handling

**Key Finding**: TransformerModule.transform_batch handles mixed JAX and non-JAX types by:
1. Splitting data into JAX arrays and non-JAX data
2. Processing JAX arrays with vmap
3. Preserving non-JAX data
4. Merging results back together

**Limitation**: Struggles with nested dictionaries containing only non-JAX values.

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

## Summary Statistics from This Session

- **test_pytree_utils.py**: Fixed 37 pyright errors → 0 errors
- **test_transformer_modules.py**: Fixed 8 errors + converted from unittest → 0 errors
- **test_sequential_parallel.py**: Fixed 17 errors → 0 errors
- **test_field_transformer.py**: Fixed 88 errors → 0 errors
- **test_image_transforms.py**: Fixed 80 errors → 0 errors
- **test_dag_monitor.py**: Fixed 13 errors → 0 errors

**Total**: Fixed 243 pyright errors across 6 test files

## Key Takeaways

1. **Type narrowing with isinstance() is essential** for union types
2. **Pytest provides better type inference** than unittest
3. **Some modules intentionally use relaxed typing** for defensive programming
4. **Wrap modules in Node classes** for DAG operations
5. **Batch dimensions are required** for JAX batch operations
6. **Explicit bool conversion** needed for condition functions

This guide should be referenced when encountering similar pyright issues in the future.
