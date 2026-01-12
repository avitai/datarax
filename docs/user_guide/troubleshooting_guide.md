# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with Datarax's NNX-based architecture, particularly around checkpointing, state management, and module integration.

## Checkpointing Issues

### State Serialization Errors

**Problem**: Custom objects or complex state structures fail to serialize with Orbax.

```python
# Error example
ValueError: TypeHandler lookup failed for: type=<class 'custom_module.CustomClass'>
```

**Solution**: Implement proper serialization methods in your custom modules.

```python
from datarax.core import DataraxModule
import flax.nnx as nnx

class CustomModule(DataraxModule):
    def __init__(self, custom_data, name="custom"):
        super().__init__(name=name)
        # Store serializable data in NNX variables
        self.serializable_data = nnx.Variable(self._to_serializable(custom_data))

    def _to_serializable(self, data):
        """Convert custom objects to serializable format."""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        elif isinstance(data, (list, tuple)):
            return [self._to_serializable(item) for item in data]
        else:
            return data

    def get_serializable_state(self):
        """Override to handle complex state serialization."""
        state = super().get_serializable_state()
        # Ensure all state is serializable
        return self._clean_state_for_serialization(state)

    def _clean_state_for_serialization(self, state):
        """Remove or convert non-serializable objects."""
        cleaned_state = {}
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                cleaned_state[key] = value
            elif hasattr(value, 'tolist'):  # JAX arrays
                cleaned_state[key] = value
            else:
                # Convert complex objects
                cleaned_state[key] = str(value)  # or custom conversion
        return cleaned_state
```

### Custom State Structure Compatibility

**Problem**: NNX's `replace_by_pure_dict` fails with custom nested state structures.

```python
# Error example
ValueError: key in pure_dict not available in state: ('custom_state', 'nested_key')
```

**Solution**: Flatten state structure or use standard NNX Variable patterns.

```python
class CompatibleModule(DataraxModule):
    def __init__(self, name="compatible"):
        super().__init__(name=name)
        # Use flat structure with NNX Variables
        self.position = nnx.Variable(0)
        self.buffer_size = nnx.Variable(100)
        self.internal_seed = nnx.Variable(42)

    def get_state(self):
        """Return flat state structure compatible with NNX."""
        state = super().get_state()
        # Avoid nested custom dictionaries
        return state

    def set_state(self, state):
        """Handle flat state restoration."""
        super().set_state(state)
        # Additional validation if needed
        self._validate_state()

    def _validate_state(self):
        """Validate state consistency after restoration."""
        assert isinstance(self.position[...], int)
        assert self.buffer_size[...] > 0
```

### Iterator State Management

**Problem**: Iterator state becomes inconsistent after checkpointing.

```python
# Problem: Iterator doesn't resume from correct position
iterator = pipeline.create_iterator()
# ... consume some batches ...
state = pipeline.get_state()
pipeline.set_state(state)
# Iterator might restart from beginning
```

**Solution**: Implement proper iterator state tracking.

```python
from datarax.core.module import CheckpointableIteratorModule

class RobustIteratorModule(CheckpointableIteratorModule):
    def __init__(self, data, name="robust_iterator"):
        super().__init__(name=name)
        self.data = nnx.Variable(data)
        self.position = nnx.Variable(0)
        self.epoch = nnx.Variable(0)
        self._iterator = None

    def create_iterator(self):
        """Create iterator that tracks position."""
        self.reset_iterator()
        return self

    def reset_iterator(self):
        """Reset iterator to current position."""
        self._iterator = iter(self.data[...][self.position[...]:])

    def __next__(self):
        if self._iterator is None:
            self.reset_iterator()

        try:
            item = next(self._iterator)
            self.position[...] = self.position[...] + 1
            return item
        except StopIteration:
            self.epoch[...] = self.epoch[...] + 1
            self.position[...] = 0
            self.reset_iterator()
            raise

    def get_state(self):
        """Include iterator position in state."""
        state = super().get_state()
        state.update({
            'iterator_position': self.position[...],
            'iterator_epoch': self.epoch[...]
        })
        return state

    def set_state(self, state):
        """Restore iterator position."""
        super().set_state(state)
        if 'iterator_position' in state:
            self.position[...] = state['iterator_position']
        if 'iterator_epoch' in state:
            self.epoch[...] = state['iterator_epoch']
        # Reset iterator to correct position
        self.reset_iterator()
```

## State Management Issues

### Variable Access Patterns

**Problem**: Incorrect access to NNX Variable values causing AttributeError.

```python
# Wrong: Accessing Variable directly
sampler.current_position = 10  # AttributeError

# Wrong: Not using proper accessor
if sampler.buffer_size > 0:  # Comparing Variable object, not value
```

**Solution**: Use slice notation `variable[...]` or `variable.get_value()` for Variable access and modification.

```python
# Correct: Access Variable values with slice notation (Flax 0.12.0+)
sampler.current_position[...] = 10

# Correct: Compare Variable values
if sampler.buffer_size[...] > 0:
    process_buffer()

# Correct: Initialize Variables properly
class SamplerModule(DataraxModule):
    def __init__(self, buffer_size=100):
        super().__init__()
        self.buffer_size = nnx.Variable(buffer_size)  # Store value in Variable
        self.current_position = nnx.Variable(0)
```

> **Note**: The `.value` attribute is deprecated as of Flax 0.12.0. Use `variable[...]` for Array variables or `variable.get_value()` for other types.

### PRNG State Consistency

**Problem**: Random number generation becomes inconsistent after state restoration.

```python
# Problem: PRNG state not properly restored
rngs = nnx.Rngs(42)
sampler = ShuffleSamplerModule(rngs=rngs)
# ... use sampler ...
state = sampler.get_state()
# ... restore state ...
sampler.set_state(state)
# Random sequence might not continue correctly
```

**Solution**: Implement proper PRNG state management.

```python
class StatefulSamplerModule(SamplerModule):
    def __init__(self, seed=0, name="stateful_sampler"):
        super().__init__(name=name)
        self.rngs = nnx.Rngs(default=seed)
        self.original_seed = nnx.Variable(seed)

    def get_state(self):
        """Include PRNG state in checkpoint."""
        state = super().get_state()
        state.update({
            'rng_state': self.rngs.default.key[...],
            'rng_count': self.rngs.default.count[...],
            'original_seed': self.original_seed[...]
        })
        return state

    def set_state(self, state):
        """Restore PRNG state."""
        super().set_state(state)
        if 'rng_state' in state and 'rng_count' in state:
            # Restore exact PRNG state
            self.rngs.default.key[...] = state['rng_state']
            self.rngs.default.count[...] = state['rng_count']
        if 'original_seed' in state:
            self.original_seed[...] = state['original_seed']
```

## Module Integration Issues

### Module Registration

**Problem**: Custom modules not recognized by checkpointing system.

```python
# Error example
ValueError: Unknown type: "<class 'custom.CustomSamplerModule'>"
```

**Solution**: Ensure proper inheritance and registration.

```python
from datarax.core import SamplerModule
import flax.nnx as nnx

# Correct: Inherit from appropriate Datarax base class
class CustomSamplerModule(SamplerModule):
    def __init__(self, custom_param=10, name="custom_sampler"):
        # Always call super().__init__
        super().__init__(name=name)

        # Use NNX Variables for state
        self.custom_param = nnx.Variable(custom_param)
        self.internal_state = nnx.Variable({})

    def sample(self, data):
        """Implement required interface."""
        # Custom sampling logic
        return data[::self.custom_param[...]]
```

### Type Handler Registration

**Problem**: Complex custom types need explicit Orbax type handlers.

```python
# For very complex custom types, register handlers
import orbax.checkpoint as ocp

class CustomTypeHandler:
    def serialize(self, value):
        # Convert to serializable format
        return {'data': value.to_dict(), 'type': 'custom'}

    def deserialize(self, serialized):
        # Reconstruct from serialized format
        return CustomType.from_dict(serialized['data'])

# Register handler
handler_registry = ocp.type_handlers.TypeHandlerRegistry()
handler_registry.register(CustomType, CustomTypeHandler())
```

## Performance Issues

### Memory Leaks in Checkpointing

**Problem**: Memory usage grows over time with frequent checkpointing.

```python
# Problem: Accumulating checkpoint data
checkpoints = []
for i in range(1000):
    state = pipeline.get_state()
    checkpoints.append(state)  # Memory leak!
```

**Solution**: Implement checkpoint rotation and cleanup.

```python
class CheckpointManager:
    def __init__(self, max_checkpoints=5):
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def save_checkpoint(self, pipeline):
        """Save checkpoint with automatic cleanup."""
        state = pipeline.get_state()
        timestamp = time.time()

        checkpoint = {
            'state': state,
            'timestamp': timestamp
        }

        self.checkpoints.append(checkpoint)

        # Cleanup old checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            # Explicitly delete to help garbage collection
            del old_checkpoint['state']
            del old_checkpoint

        return len(self.checkpoints) - 1  # Return checkpoint index

    def restore_checkpoint(self, pipeline, index=-1):
        """Restore from specific checkpoint (default: latest)."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        checkpoint = self.checkpoints[index]
        pipeline.set_state(checkpoint['state'])
        return checkpoint['timestamp']
```

### Large State Serialization

**Problem**: Large pipeline states cause slow checkpointing.

```python
# Problem: Serializing large data buffers
class LargeBufferModule(DataraxModule):
    def __init__(self, buffer_size=1000000):
        super().__init__()
        self.large_buffer = nnx.Variable(jnp.zeros(buffer_size))  # Too large!
```

**Solution**: Implement efficient state management strategies.

```python
class EfficientBufferModule(DataraxModule):
    def __init__(self, buffer_size=1000000):
        super().__init__()
        # Only store essential state
        self.buffer_size = nnx.Variable(buffer_size)
        self.buffer_position = nnx.Variable(0)
        self.buffer_seed = nnx.Variable(42)
        # Don't store actual buffer data in state
        self._buffer = None

    def get_serializable_state(self):
        """Only serialize essential state."""
        state = super().get_serializable_state()
        # Remove large buffers from serialization
        state_copy = {}
        for key, value in state.items():
            if key.startswith('_buffer'):
                continue  # Skip large internal buffers
            state_copy[key] = value
        return state_copy

    def set_state(self, state):
        """Restore state and rebuild buffers."""
        super().set_state(state)
        # Rebuild buffer from essential state
        self._rebuild_buffer()

    def _rebuild_buffer(self):
        """Rebuild large buffers from essential state."""
        if self._buffer is None:
            self._buffer = jnp.zeros(self.buffer_size[...])
        # Apply any necessary initialization based on state
```

## Debugging Tools

### State Inspection

```python
def inspect_module_state(module, depth=0):
    """Recursively inspect module state for debugging."""
    indent = "  " * depth
    print(f"{indent}{module.__class__.__name__}:")

    if hasattr(module, 'get_state'):
        state = module.get_state()
        for key, value in state.items():
            if isinstance(value, (int, float, str, bool)):
                print(f"{indent}  {key}: {value}")
            elif hasattr(value, 'shape'):
                print(f"{indent}  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{indent}  {key}: {type(value)}")

# Usage
inspect_module_state(pipeline)
```

### Checkpoint Validation

```python
def validate_checkpoint_integrity(original_module, restored_module):
    """Validate that checkpoint restoration was successful."""
    orig_state = original_module.get_state()
    rest_state = restored_module.get_state()

    # Check state keys match
    assert set(orig_state.keys()) == set(rest_state.keys()), \
        f"State keys mismatch: {orig_state.keys()} vs {rest_state.keys()}"

    # Check state values
    for key in orig_state.keys():
        orig_val = orig_state[key]
        rest_val = rest_state[key]

        if hasattr(orig_val, 'shape'):
            assert jnp.allclose(orig_val, rest_val), f"Array mismatch for key: {key}"
        else:
            assert orig_val == rest_val, f"Value mismatch for key: {key}"

    print("✅ Checkpoint validation passed!")

# Usage
original_state = pipeline.get_state()
pipeline.set_state(original_state)
validate_checkpoint_integrity(original_pipeline, pipeline)
```

## Best Practices Summary

1. **Always use NNX Variables**: Store mutable state in `nnx.Variable` objects
2. **Access with slice notation**: Use `variable[...]` for Arrays or `variable.get_value()` for other types (`.value` is deprecated in Flax 0.12.0+)
3. **Implement clean serialization**: Override `get_serializable_state()` for complex objects
4. **Validate after restoration**: Include validation in your restoration workflow
5. **Manage memory**: Implement checkpoint rotation for long-running processes
6. **Keep state flat**: Avoid deeply nested custom state structures
7. **Test thoroughly**: Create unit tests for your checkpointing functionality

## See Also

- [Checkpointing Guide](checkpointing_guide.md)
- [NNX Best Practices](nnx_best_practices.md)
- [API Reference](../core/index.md)
