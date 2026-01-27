# Operators

Data transformation operators for building processing pipelines. Operators are the workhorses of Datarax - they transform, augment, and process data as it flows through your pipeline.

## Operator Types

| Type | Description | Use Case |
|------|-------------|----------|
| **ElementOperator** | Full element access | Coordinated multi-field transforms |
| **MapOperator** | Per-array-leaf transforms | Simple field-wise operations |
| **CompositeOperator** | Compose multiple operators | Complex augmentation pipelines |
| **BatchMixOperator** | Batch-level mixing | MixUp, CutMix augmentation |

`★ Insight ─────────────────────────────────────`

- **Deterministic** operators are pure functions (same input → same output)
- **Stochastic** operators use RNG and require `stream_name` in config
- Use `>>` operator to chain: `op1 >> op2 >> op3`
- All operators work inside `jax.jit` for performance

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.operators import ElementOperator, CompositeOperatorModule
from datarax.core.config import ElementOperatorConfig

# Simple element transformation
def normalize(element, key):
    return element.replace(
        data={"image": element.data["image"] / 255.0}
    )

config = ElementOperatorConfig(stochastic=False)
op = ElementOperator(config, fn=normalize)
```

## Core Operators

Base operator types for building data pipelines:

- [element_operator](element_operator.md) - Element-level transformations with full access
- [map_operator](map_operator.md) - Per-array-leaf map transformations
- [composite_operator](composite_operator.md) - Compose multiple operators (11 strategies)
- [batch_mix_operator](batch_mix_operator.md) - MixUp and CutMix augmentation
- [probabilistic_operator](probabilistic_operator.md) - Apply operators with probability
- [selector_operator](selector_operator.md) - Random operator selection

## Image Operators

Specialized operators for image data transformations:

- [brightness_operator](brightness_operator.md) - Brightness adjustments
- [contrast_operator](contrast_operator.md) - Contrast modifications
- [dropout_operator](dropout_operator.md) - Pixel dropout regularization
- [noise_operator](noise_operator.md) - Gaussian/uniform noise injection
- [patch_dropout_operator](patch_dropout_operator.md) - Patch-level dropout
- [rotation_operator](rotation_operator.md) - Image rotation transforms
- [functional](functional.md) - Functional image operations (stateless)

## Composition Strategies

Strategies for combining operators in `CompositeOperatorModule`:

- [sequential](sequential.md) - Chain operators in sequence
- [parallel](parallel.md) - Apply operators in parallel, merge results
- [branching](branching.md) - Conditional routing based on data
- [ensemble](ensemble.md) - Ensemble multiple operators with reduction
- [merging](merging.md) - Output merging strategies (concat, sum, mean)
- [base](base.md) - Base strategy interface

## See Also

- [Element Operator Guide](element_operator.md) - Detailed element operator docs
- [Composite Operator Guide](composite_operator.md) - Composition strategies
- [DAG Executor](../dag/dag_executor.md) - Using operators in pipelines
- [Operators Tutorial](../examples/core/operators-tutorial.md)
