# Datarax vs Google Grain: Critical Conceptual Differences

This directory contains detailed examples demonstrating the fundamental architectural differences between the `datarax` (Datarax) stateful approach and Google Grain's stateless framework.

> **These scripts are architectural-pattern mock-ups.** They illustrate the
> stateful-vs-stateless *design* using plain `flax.nnx` + `numpy` (with `jax`,
> `optax`, and `psutil` for a few demos). They contain **no `datarax` and no
> `grain` imports** — they do not exercise the real libraries. Treat every code
> snippet, API name, and performance number below as an illustration of the
> pattern, not as a runnable Datarax or Grain API reference.

## 📁 Directory Structure

```
examples/comparison/
├── README.md                           # This file
├── quick_demo.py                       # Quick 100-line demo of core difference
├── 02_stateful_transformations.py      # Transform pipeline comparison
├── 03_distributed_memory_efficient.py  # Distributed & memory comparison
└── 04_end_to_end_pipeline.py           # Complete pipeline comparison
```

## 🎯 Core Conceptual Differences

### 1. **Stateful vs Stateless Architecture**

| Aspect | Datarax (Stateful) | Google Grain (Stateless) |
|--------|--------------------------|--------------------------|
| **State Management** | Internal via NNX Variables | External, passed around |
| **Checkpointing** | Iterator-level `get_state()`/`set_state()` | Manual state collection |
| **Iteration** | Natural Python `__iter__`/`__next__` | Generator with state tuples |
| **PRNG Handling** | Internal `nnx.Rngs` management | External key passing |
| **Code Complexity** | Clean, encapsulated | More boilerplate |

### 2. **Key Advantages of Datarax's Approach**

#### ✅ **Cleaner APIs**
- No need to pass state between function calls
- Natural Python iteration protocol
- Automatic state tracking with NNX

#### ✅ **Better Performance** *(illustrative figures)*
- ~1.5-2x faster iteration (no tuple unpacking overhead)
- ~30-50% memory savings with shared memory management
- Better JIT compilation due to cleaner boundaries

#### ✅ **Robust Checkpointing**
- Checkpointing lives on the **iterator**, not the pipeline: `iter(pipeline)`
  returns a `PipelineIterator` exposing `get_state()`, `set_state()`, and `close()`
- One-line checkpoint: `state = iterator.get_state()`
- One-line restore: `iterator.set_state(state)`
- All component states captured in that iteration session

#### ✅ **Learnable Components**
- Transformations can have trainable parameters
- Automatic gradient computation with `nnx.grad`
- Seamless integration with optimizers

## 🚀 Running the Examples

### Prerequisites

```bash
# Install dependencies
pip install jax jaxlib flax numpy psutil

# Or using uv (recommended)
uv pip install jax jaxlib flax numpy psutil
```

### Quick Demo
Quick 100-line demo showing the core difference between stateful and stateless.

```bash
python examples/comparison/quick_demo.py
```

**Key Insights:**
- Shows external vs internal state management in ~100 lines
- Demonstrates automatic checkpointing
- Highlights cleaner NNX-based code

### Example 2: Stateful Transformations
Shows learnable transformations and automatic state management.

```bash
python examples/comparison/02_stateful_transformations.py
```

**Key Insights:**
- Transformations as NNX modules
- Learnable parameters in transforms
- Automatic batch normalization state
- Clean PRNG handling for augmentations

### Example 3: Distributed & Memory Efficient
Demonstrates multi-worker coordination and memory optimization.

```bash
python examples/comparison/03_distributed_memory_efficient.py
```

**Key Insights:**
- Automatic worker coordination
- Illustrative 30-50% memory savings with sharing
- Clean JAX sharding integration
- Built-in performance monitoring

### Example 4: End-to-End Pipeline
Walks through a complete pipeline comparison from source to trained step.

```bash
python examples/comparison/04_end_to_end_pipeline.py
```

**Key Insights:**
- Full source → transform → batch → train flow in both styles
- Stateful vs. stateless composition across a realistic pipeline
- Combined view of checkpointing, iteration, and state management

## 📊 Performance Comparison Summary

> The figures in this section are **illustrative** of the architectural pattern
> and are not measured Datarax vs. Grain benchmarks.

| Metric | Datarax | Grain | Improvement |
|--------|--------------|-------|-------------|
| **Iteration Speed** | 0.12s/1000 samples | 0.24s/1000 samples | 2x faster |
| **Memory Usage (4 workers)** | 150 MB | 250 MB | 40% less |
| **Checkpoint Size** | Minimal | Larger | ~30% smaller |
| **Lines of Code** | 50 | 80 | 37% less |
| **State Management** | Automatic | Manual | ∞ better |

## 🔬 When to Use Datarax Over Grain

### ✅ **Use Datarax When:**

1. **You need stateful components** - Running statistics, learnable parameters, complex state
2. **You want clean checkpointing** - Production ML training with fault tolerance
3. **You need memory efficiency** - Large-scale data with multiple workers
4. **You prefer Pythonic APIs** - Natural iteration, no tuple unpacking
5. **You're using JAX transformations** - Better integration with `jit`, `grad`, `vmap`
6. **You need monitoring** - Built-in statistics and progress tracking

### ⚠️ **Consider Grain When:**

1. **You prefer pure functional style** - No mutable state
2. **You have simple, stateless pipelines** - Basic data loading without complex transforms
3. **You're already using Grain** - Migration cost might not be worth it

## 🎓 Technical Deep Dive

### NNX Variables vs External State

**Datarax (NNX):**
```python
class StatefulComponent(nnx.Module):
    def __init__(self):
        self.count = nnx.Variable(0)  # Internal state

    def process(self, data):
        self.count.value += 1  # Automatic tracking
        return data
```

**Grain (External):**
```python
def process(data, state):
    new_state = {"count": state["count"] + 1}
    return data, new_state  # Must return state
```

### Checkpoint/Restore Pattern

**Datarax:**
```python
# Open an iteration session (returns a PipelineIterator)
iterator = iter(pipeline)

# Save
checkpoint = iterator.get_state()

# Restore into a session
iterator.set_state(checkpoint)

# Release session resources when done
iterator.close()
```

> There is no pipeline-level `get_state()`/`set_state()`. State is scoped to the
> `PipelineIterator` returned by `iter(pipeline)`.

**Grain:**
```python
# Save (manual collection)
checkpoint = {
    "loader": loader_state,
    "transform1": transform1_state,
    "transform2": transform2_state,
    # ... must track everything manually
}

# Restore (manual distribution)
loader = create_loader(checkpoint["loader"])
transform1.set_state(checkpoint["transform1"])
# ... restore each component
```

## 📈 Production Considerations

### Reliability
- **Datarax**: Automatic state tracking reduces bugs
- **Grain**: Manual state management prone to errors

### Scalability
- **Datarax**: Better memory efficiency at scale
- **Grain**: Higher memory overhead with multiple workers

### Maintainability
- **Datarax**: Less code, cleaner interfaces
- **Grain**: More boilerplate, complex state flow

### Performance *(illustrative figures)*
- **Datarax**: ~1.5-2x faster, better JIT compilation
- **Grain**: Overhead from state passing

## 🚦 Migration Guide

If migrating from Grain to Datarax:

1. **Convert stateless functions to NNX modules**
2. **Replace external state with NNX Variables**
3. **Use the iterator's `get_state()`/`set_state()` for checkpointing**
4. **Leverage automatic PRNG management**
5. **Simplify worker coordination code**

## 📝 Conclusion

The datarax approach with stateful NNX modules provides significant advantages over Grain's stateless design (figures below are **illustrative** of the pattern, not measured benchmarks):

- **~50% less code** for equivalent functionality
- **~2x better performance** in typical scenarios
- **~40% memory savings** with multiple workers
- **Automatic everything** - state, checkpoints, statistics
- **Production-ready** - robust, maintainable, scalable

These examples demonstrate that the stateful approach is a **strong fit** for real-world machine learning pipelines.

---

For questions or contributions, please open an issue in the datarax repository.
