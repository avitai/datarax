# Advanced Operators Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 min |
| **Prerequisites** | Operators Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master advanced operators in Datarax: probabilistic application,
random operator selection, and spatial dropout. These operators
enable sophisticated augmentation pipelines and AutoAugment-style
learned policies.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Apply transformations probabilistically with `ProbabilisticOperator`
2. Randomly select from multiple operators with `SelectorOperator`
3. Use patch-based occlusion with `PatchDropoutOperator`
4. Build AutoAugment-style augmentation pipelines
5. Understand the JAX compatibility patterns used

## Coming from PyTorch?

| PyTorch torchvision | Datarax |
|---------------------|---------|
| `transforms.RandomApply([t], p=0.5)` | `ProbabilisticOperator(config, ...)` |
| `transforms.RandomChoice([t1, t2])` | `SelectorOperator(config, ...)` |
| `transforms.RandomErasing()` | `PatchDropoutOperator(config, ...)` |

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| Custom with `tf.random.uniform < p` | `ProbabilisticOperator` |
| Custom with `tf.cond` | `SelectorOperator` |
| `tf.image.random_erasing` (TF Addons) | `PatchDropoutOperator` |

## Files

- **Python Script**: [`examples/core/09_advanced_operators_tutorial.py`](https://github.com/avitai/datarax/blob/main/examples/core/09_advanced_operators_tutorial.py)
- **Jupyter Notebook**: [`examples/core/09_advanced_operators_tutorial.ipynb`](https://github.com/avitai/datarax/blob/main/examples/core/09_advanced_operators_tutorial.ipynb)

## Quick Start

### Run the Python Script

```bash
python examples/core/09_advanced_operators_tutorial.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/core/09_advanced_operators_tutorial.ipynb
```

## Key Concepts

### ProbabilisticOperator

Wraps any operator and applies it with a configured probability.

```python
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)

# Wrap operator with 50% probability
prob_brightness = ProbabilisticOperator(
    ProbabilisticOperatorConfig(
        operator=brightness_op,
        probability=0.5,
    ),
    rngs=nnx.Rngs(augment=42),
)
```

**Behavior by Probability:**

| Probability | Mode | Behavior |
|-------------|------|----------|
| `p = 0.0` | Deterministic | Never apply (passthrough) |
| `0 < p < 1` | Stochastic | Apply with probability p |
| `p = 1.0` | Deterministic | Always apply |

**Terminal Output:**
```
Effect of probability values:
  p=0.00 (deterministic): mean delta = +0.0000
  p=0.25 (stochastic   ): mean delta = +0.0500
  p=0.50 (stochastic   ): mean delta = +0.1000
  p=0.75 (stochastic   ): mean delta = +0.1500
  p=1.00 (deterministic): mean delta = +0.2000
```

### SelectorOperator

Randomly selects ONE operator from a list to apply per sample.

```python
from datarax.operators.selector_operator import (
    SelectorOperator,
    SelectorOperatorConfig,
)

selector = SelectorOperator(
    SelectorOperatorConfig(
        operators=[op_bright, op_contrast, op_noise],
        weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
    ),
    rngs=nnx.Rngs(augment=100),
)
```

**Terminal Output:**
```
SelectorOperator created:
  Operators: [Brightness, Contrast, Noise]
  Weights: [50%, 30%, 20%]
  Always stochastic: True
```

### PatchDropoutOperator

Applies patch-based occlusion by dropping random rectangular regions.

```python
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)

patch_dropout = PatchDropoutOperator(
    PatchDropoutOperatorConfig(
        field_key="image",
        num_patches=4,
        patch_size=(8, 8),
        drop_value=0.0,  # Black patches
        stochastic=True,
    ),
    rngs=nnx.Rngs(patch=42),
)
```

**Configuration:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_patches` | Number of patches to drop | 4 |
| `patch_size` | Size as (height, width) | (8, 8) |
| `drop_value` | Fill value for dropped regions | 0.0 |

## Building AutoAugment-Style Pipelines

Combine operators to build sophisticated augmentation pipelines:

```python
# 1. Probabilistically apply brightness (60%)
prob_bright = ProbabilisticOperator(
    ProbabilisticOperatorConfig(operator=bright, probability=0.6),
    rngs=nnx.Rngs(augment=100),
)

# 2. Probabilistically apply contrast (60%)
prob_contrast = ProbabilisticOperator(
    ProbabilisticOperatorConfig(operator=contrast, probability=0.6),
    rngs=nnx.Rngs(augment=200),
)

# 3. Randomly select: noise (70%) or patch dropout (30%)
final_selector = SelectorOperator(
    SelectorOperatorConfig(
        operators=[noise, patch],
        weights=[0.7, 0.3],
    ),
    rngs=nnx.Rngs(augment=300),
)

# Build pipeline
pipeline = (
    from_source(source, batch_size=32)
    .add(OperatorNode(prob_bright))
    .add(OperatorNode(prob_contrast))
    .add(OperatorNode(final_selector))
)
```

## JAX Compatibility Patterns

All advanced operators use specific patterns for JAX compatibility:

| Pattern | Used In | Purpose |
|---------|---------|---------|
| `jax.lax.cond` | ProbabilisticOperator | Conditional execution |
| `jax.lax.switch` | SelectorOperator | Multi-way branching |
| `jax.lax.fori_loop` | PatchDropoutOperator | Loop over patches |
| Pre-generated random params | All | vmap compatibility |

**Why These Patterns?**

1. **No Python if statements**: Traced JAX values can't be used in Python conditionals
2. **Pre-generated randoms**: Avoids RNG state mutations inside vmap
3. **Fixed output shapes**: vmap requires consistent shapes across branches

## Results

Running the tutorial produces:

```
============================================================
Advanced Operators Tutorial
============================================================

1. ProbabilisticOperator (p=0.5):
   Output mean: 0.5987

2. SelectorOperator (3 operators):
   Output mean: 0.5623

3. PatchDropoutOperator (4 patches):
   Output mean: 0.4521 (lower due to black patches)

============================================================
Tutorial completed successfully!
============================================================
```

## Operator Summary

| Operator | Purpose | Key Config |
|----------|---------|------------|
| `ProbabilisticOperator` | Apply with probability | `probability` (0-1) |
| `SelectorOperator` | Random selection | `operators`, `weights` |
| `PatchDropoutOperator` | Spatial dropout | `num_patches`, `patch_size` |

## Use Cases

| Use Case | Operator(s) |
|----------|-------------|
| Stochastic augmentation | `ProbabilisticOperator` |
| AutoAugment policies | `SelectorOperator` + `ProbabilisticOperator` |
| Occlusion robustness | `PatchDropoutOperator` |
| Test-time augmentation | All, with various settings |

## Next Steps

- [Composition Strategies](composition-strategies-tutorial.md) - Combine operators
- [MixUp/CutMix Tutorial](../advanced/augmentation/mixup-cutmix-tutorial.md) - Batch mixing
- [Performance Guide](../advanced/performance/optimization-guide.md) - Optimization

## API Reference

- [`ProbabilisticOperator`](../../operators/probabilistic_operator.md)
- [`SelectorOperator`](../../operators/selector_operator.md)
- [`PatchDropoutOperator`](../../operators/patch_dropout_operator.md)
