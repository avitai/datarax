# Composite Operator

The `CompositeOperatorModule` enables **composing multiple operators** into sophisticated pipelines using 11 different composition strategies. It's the foundation for building complex data augmentation and transformation workflows.

## Composition Strategies

| Strategy | Description |
|----------|-------------|
| **Sequential** | Chain operators: output of one → input of next |
| **Conditional Sequential** | Chain with per-operator conditions |
| **Dynamic Sequential** | Runtime-modifiable chain |
| **Parallel** | Apply all operators to same input, merge outputs |
| **Weighted Parallel** | Parallel with learnable weights |
| **Conditional Parallel** | Parallel with per-operator conditions |
| **Ensemble Mean/Sum/Max/Min** | Parallel + reduction |
| **Branching** | Route through different paths based on input |

`★ Insight ─────────────────────────────────────`

- CompositeOperator uses **JAX-compatible patterns** throughout
- Integer-based branching with `jax.lax.switch` (not dict lookups)
- Fixed-shape outputs for vmap compatibility
- All strategies work inside `jax.jit` and `jax.vmap`

`─────────────────────────────────────────────────`

## Quick Start

### Sequential Composition

Chain operators where each output feeds into the next:

```python
from datarax.operators import CompositeOperatorModule
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositionStrategy,
)

# Create child operators
normalize = create_normalize_op()
augment = create_augment_op()

config = CompositeOperatorConfig(
    strategy=CompositionStrategy.SEQUENTIAL,
    operators=[normalize, augment],
)
pipeline = CompositeOperatorModule(config)
```

### Parallel Composition

Apply multiple operators to the same input and merge results:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.PARALLEL,
    operators=[op_a, op_b, op_c],
    merge_strategy="concat",  # or "stack", "sum", "mean", "dict"
    merge_axis=-1,
)
parallel_op = CompositeOperatorModule(config)
```

### Ensemble with Reduction

Combine multiple model outputs with reduction:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.ENSEMBLE_MEAN,
    operators=[model_a, model_b, model_c],
)
ensemble = CompositeOperatorModule(config)
# Output is element-wise mean of all operator outputs
```

### Conditional Branching

Route data through different paths based on conditions:

```python
def router(data):
    """Return integer index of operator to use."""
    # Must return int or JAX scalar (not strings!)
    return 0 if data["type"] == "image" else 1

config = CompositeOperatorConfig(
    strategy=CompositionStrategy.BRANCHING,
    operators=[image_processor, text_processor],
    router=router,
)
branched = CompositeOperatorModule(config)
```

## Weighted Parallel (Learnable)

Create learnable weighted combinations:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.WEIGHTED_PARALLEL,
    operators=[op_a, op_b],
    weights=[0.5, 0.5],
    learnable_weights=True,  # Weights become trainable parameters
)
weighted = CompositeOperatorModule(config, rngs=nnx.Rngs(0))

# Access weights for training
current_weights = weighted.weights.get_value()
```

## Dynamic Sequential

Modify the operator chain at runtime:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
    operators=[op_a, op_b],
)
dynamic = CompositeOperatorModule(config)

# Modify at runtime
dynamic.add_operator(op_c)
dynamic.remove_operator(1)
dynamic.reorder_operators([1, 0, 2])
```

## JAX Compatibility Notes

!!! warning "Important for JIT/vmap"

    - **Router functions must return integers**, not strings
    - All code paths must return the **same PyTree structure**
    - Conditions should use `jax.lax.cond`, not Python `if`

```python
# ✅ Correct: Integer-based routing
def router(x): return 0 if condition else 1

# ❌ Wrong: String-based routing (breaks tracing)
def router(x): return "path_a" if condition else "path_b"
```

## See Also

- [Element Operator](element_operator.md) - Single-element transformations
- [Operator Strategies](sequential.md) - Strategy implementations
- [DAG Control Flow](../dag/control_flow.md) - DAG-level branching
- [Operators Tutorial](../examples/core/operators-tutorial.md)

---

## API Reference

::: datarax.operators.composite_operator
