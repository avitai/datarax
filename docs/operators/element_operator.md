# Element Operator

The `ElementOperator` is Datarax's most commonly used operator for element-level transformations. Unlike `MapOperator` (which transforms individual array leaves), `ElementOperator` provides access to the **full Element structure** - including data, state, and metadata - enabling coordinated transformations across multiple fields.

## Key Concepts

`★ Insight ─────────────────────────────────────`

- **ElementOperator** works with entire `Element` objects, not individual arrays
- User functions receive `fn(element, key) -> element` signature
- Use `element.replace()` for immutable updates (Pythonic JAX pattern)
- Supports both deterministic and stochastic modes via configuration

`─────────────────────────────────────────────────`

## When to Use ElementOperator

| Use Case | Example |
|----------|---------|
| **Coordinated transformations** | Flip an image AND its segmentation mask together |
| **Multi-field processing** | Normalize image based on mask statistics |
| **State tracking** | Update element state based on transformation |
| **Metadata-aware processing** | Apply different augmentations based on metadata |

## Quick Start

```python
import flax.nnx as nnx
from datarax.operators import ElementOperator
from datarax.core.config import ElementOperatorConfig

# Define a transformation function
def normalize(element, key):
    """Normalize image values to [0, 1]."""
    new_data = {"image": element.data["image"] / 255.0}
    return element.replace(data=new_data)

# Create operator (deterministic mode)
config = ElementOperatorConfig(stochastic=False)
op = ElementOperator(config, fn=normalize, rngs=nnx.Rngs(0))

# Apply to an element
result = op.apply(element.data, element.state, element.metadata)
```

## Stochastic Transformations

For random augmentations, use stochastic mode with a stream name:

```python
import jax

def add_noise(element, key):
    """Add random Gaussian noise to image."""
    noise = jax.random.normal(key, element.data["image"].shape) * 0.1
    new_data = {"image": element.data["image"] + noise}
    return element.replace(data=new_data)

config = ElementOperatorConfig(stochastic=True, stream_name="augment")
op = ElementOperator(config, fn=add_noise, rngs=nnx.Rngs(42))
```

## Coordinated Augmentations

One of ElementOperator's key strengths is applying the **same random decision** to multiple fields:

```python
import jax.lax

def flip_both(element, key):
    """Randomly flip image and mask together."""
    should_flip = jax.random.uniform(key) < 0.5

    new_data = jax.lax.cond(
        should_flip,
        lambda: {
            "image": element.data["image"][..., ::-1],
            "mask": element.data["mask"][..., ::-1]
        },
        lambda: element.data,
    )
    return element.replace(data=new_data)

config = ElementOperatorConfig(stochastic=True, stream_name="flip")
flip_op = ElementOperator(config, fn=flip_both, rngs=nnx.Rngs(0))
```

## Integration with DAG Pipelines

ElementOperator integrates seamlessly with Datarax's DAG execution:

```python
from datarax.dag import from_source
from datarax.dag.nodes import OperatorNode

# Build a pipeline with ElementOperator
pipeline = (
    from_source(my_source, batch_size=32)
    >> OperatorNode(normalize_op)
    >> OperatorNode(flip_op)
)

# Iterate over batches
for batch in pipeline:
    train_step(batch)
```

## See Also

- [Operators Overview](index.md) - All available operators
- [MapOperator](map_operator.md) - For per-array-leaf transformations
- [CompositeOperator](composite_operator.md) - For chaining operators
- [DAG Executor](../dag/dag_executor.md) - Pipeline execution
- [Operators Tutorial](../examples/core/operators-tutorial.md) - Hands-on examples

---

## API Reference

::: datarax.operators.element_operator
