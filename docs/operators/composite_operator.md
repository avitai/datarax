# Composite Operator

The `CompositeOperatorModule` enables **composing multiple operators** into sophisticated pipelines using 11 different composition strategies. It's the foundation for building complex data augmentation and transformation workflows.

## Composition Strategies

| Strategy | Description |
|----------|-------------|
| **Sequential** | Chain operators: output of one → input of next |
| **Conditional Sequential** | Chain with per-operator conditions |
| **Dynamic Sequential** | Runtime-modifiable chain |
| **Parallel** | Apply all operators to same input, merge outputs |
| **Weighted Parallel** | Weighted sum of named fields, with static, learnable or per-record weights |
| **Conditional Parallel** | Parallel with per-operator conditions |
| **Ensemble Mean/Sum/Max/Min** | Parallel + reduction |
| **Branching** | Route through different paths based on input |

!!! note "Key points"

    - CompositeOperator uses **JAX-compatible patterns** throughout
    - Integer-based branching with `jax.lax.switch` (not dict lookups)
    - Fixed-shape outputs for vmap compatibility
    - All strategies work inside `jax.jit` and `jax.vmap`

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

## Weighted Parallel

A weighted parallel composite runs every operator on the same record and replaces the fields
named in `mix_fields` with the weighted sum of the operators' outputs. Every other field passes
through from the input unchanged, dtype included. `mix_fields` defaults to the fields the
operators declare they write (`target_key` or `field_key`) and is required when an operator
declares none.

Static weights form a linear combination, such as DDSP's harmonic-plus-noise sum:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.WEIGHTED_PARALLEL,
    operators=[harmonic, noise],
    weights=[1.0, 0.1],
    mix_fields=("audio",),
)
```

Learnable weights are logits, initialized to `log(weights / sum(weights))` and mixed with
`softmax(logits / temperature)`, the relaxation DARTS and Faster AutoAugment use to learn which
operation to apply:

```python
config = CompositeOperatorConfig(
    strategy=CompositionStrategy.WEIGHTED_PARALLEL,
    operators=[brightness, contrast],  # both declare field_key="image"
    weights=[0.5, 0.5],
    learnable_weights=True,
    temperature=1.0,
)
weighted = CompositeOperatorModule(config, rngs=nnx.Rngs(0))

# The mixture the composite currently applies
current_weights = nnx.softmax(weighted.weight_logits[...] / config.temperature)
```

With `weight_key="op_weights"` the weights come from each record instead, for example
Gumbel-Softmax weights from an upstream policy.

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
# After add + remove, only 2 operators remain
dynamic.reorder_operators([1, 0])
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
- DAG Control Flow - DAG-level branching
- [Operators Tutorial](../examples/core/operators-tutorial.md)

---

## API Reference

::: datarax.operators.composite_operator
