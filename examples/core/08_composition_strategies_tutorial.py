# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Composition Strategies Deep Dive

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~30 min |
| **Prerequisites** | Operators Tutorial, Pipeline Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master the 11 composition strategies in Datarax for combining operators.
This tutorial covers sequential chaining, parallel application, ensemble
reductions, and dynamic branching - all with JAX vmap/JIT compatibility.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Chain operators with Sequential strategies (basic, conditional, dynamic)
2. Apply operators in parallel with different merge modes
3. Use weighted combinations for learnable augmentation
4. Build ensemble reductions (mean, sum, max, min)
5. Route data through branches based on conditions
6. Write vmap/JIT-compatible composition patterns
"""

# %% [markdown]
"""
## Coming from PyTorch?

| PyTorch | Datarax |
|---------|---------|
| `transforms.Compose([t1, t2])` | `CompositeOperatorModule(..., strategy=SEQUENTIAL)` |
| `transforms.RandomChoice([t1, t2])` | `CompositeOperatorModule(..., strategy=BRANCHING)` |
| `transforms.RandomApply([t], p=0.5)` | `CompositeOperatorModule(..., strategy=COND_SEQ)` |
| Manual weighted ensemble | `CompositeOperatorModule(..., strategy=WEIGHTED_PARALLEL)` |

## Coming from TensorFlow?

| TensorFlow | Datarax |
|------------|---------|
| `tf.keras.Sequential([l1, l2])` | `CompositeOperatorModule(..., strategy=SEQUENTIAL)` |
| `tf.keras.layers.Average([o1, o2])` | `CompositeOperatorModule(..., strategy=ENSEMBLE_MEAN)` |
| Custom conditional logic | `CompositeOperatorModule(..., strategy=CONDITIONAL_*)` |
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[data]"
```
"""

# %%
# Imports
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
"""
## Part 1: Setup - Create Sample Data and Base Operators

We'll create a small image dataset and several operators to demonstrate
different composition strategies.
"""

# %%
# Create sample image data
np.random.seed(42)
num_samples = 100
data = {
    "image": np.random.randint(0, 256, (num_samples, 32, 32, 3)).astype(np.float32) / 255.0,
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
print(f"Dataset: {num_samples} samples, shape {data['image'].shape}")


# %%
# Define helper operators for composition demos
def make_brightness_op(delta: float, seed: int = 0) -> BrightnessOperator:
    """Create a brightness operator with fixed delta."""
    return BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(delta, delta),  # Fixed delta
            stochastic=False,
        ),
        rngs=nnx.Rngs(seed),
    )


def make_contrast_op(factor: float, seed: int = 0) -> ContrastOperator:
    """Create a contrast operator with fixed factor."""
    return ContrastOperator(
        ContrastOperatorConfig(
            field_key="image",
            contrast_range=(factor, factor),  # Fixed factor
            stochastic=False,
        ),
        rngs=nnx.Rngs(seed),
    )


def make_noise_op(std: float, seed: int = 0) -> NoiseOperator:
    """Create a noise operator."""
    return NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=std,
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=seed),
    )


print("Helper functions created for building operators")

# %% [markdown]
"""
## Part 2: Sequential Strategies

Sequential strategies chain operators: output of one becomes input of next.

### Strategy Comparison

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `SEQUENTIAL` | Simple chain: op₁ → op₂ → op₃ | Standard augmentation pipeline |
| `CONDITIONAL_SEQUENTIAL` | Chain with per-op conditions | Skip ops based on data properties |
| `DYNAMIC_SEQUENTIAL` | Runtime-modifiable chain | Adaptive pipelines |
"""

# %%
# SEQUENTIAL: Basic chaining
# brightness(+0.1) → contrast(1.2) → result

bright_op = make_brightness_op(0.1, seed=1)
contrast_op = make_contrast_op(1.2, seed=2)

sequential_composite = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[bright_op, contrast_op],
    ),
    rngs=nnx.Rngs(0),
)

# Test it
source1 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(10))
pipeline = from_source(source1, batch_size=16).add(OperatorNode(sequential_composite))
batch = next(iter(pipeline))

print("SEQUENTIAL Strategy:")
print("  Chain: Brightness(+0.1) → Contrast(×1.2)")
print("  Input range: [0.0, 1.0]")
print(f"  Output range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

# %%
# CONDITIONAL_SEQUENTIAL: Skip operators based on conditions
# Only apply brightness if image mean < 0.5


def is_dark(data):
    """Condition: True if average pixel value < 0.5."""
    return jnp.mean(data["image"]) < 0.5


def always_true(data):  # noqa: ARG001
    """Always apply this operator."""
    return True


bright_op2 = make_brightness_op(0.2, seed=3)
contrast_op2 = make_contrast_op(1.1, seed=4)

conditional_seq = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
        operators=[bright_op2, contrast_op2],
        conditions=[is_dark, always_true],  # Brightness only if dark
    ),
    rngs=nnx.Rngs(0),
)

print()
print("CONDITIONAL_SEQUENTIAL Strategy:")
print("  - Brightness: Only if mean < 0.5 (dark images)")
print("  - Contrast: Always applied")
print("  Note: Conditions use jax.lax.cond for JIT compatibility")

# %% [markdown]
"""
## Part 3: Parallel Strategies

Parallel strategies apply ALL operators to the SAME input, then merge outputs.

### Merge Modes

| Mode | Description | Output Shape |
|------|-------------|--------------|
| `"concat"` | Concatenate along axis | (N, H, W, C×num_ops) |
| `"stack"` | Stack into new dimension | (num_ops, N, H, W, C) |
| `"sum"` | Element-wise sum | Same as input |
| `"mean"` | Element-wise mean | Same as input |
| `"dict"` | Keep separate in dict | `{op_0: ..., op_1: ...}` |
"""

# %%
# PARALLEL: Apply multiple augmentations to same input, merge results

op_bright = make_brightness_op(0.15, seed=10)
op_contrast = make_contrast_op(1.3, seed=11)
op_noise = make_noise_op(0.05, seed=12)

# Merge with mean - creates averaged augmentation
parallel_mean = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.PARALLEL,
        operators=[op_bright, op_contrast, op_noise],
        merge_strategy="mean",  # Average the three versions
    ),
    rngs=nnx.Rngs(0),
)

source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(20))
pipeline = from_source(source2, batch_size=16).add(OperatorNode(parallel_mean))
batch = next(iter(pipeline))

print("PARALLEL Strategy (merge='mean'):")
print("  Operators: [Brightness, Contrast, Noise]")
print(f"  Output shape: {batch['image'].shape} (same as input)")
print("  Output is the mean of all three augmented versions")

# %%
# PARALLEL with dict merge - keep all versions separate
parallel_dict = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.PARALLEL,
        operators=[op_bright, op_contrast],
        merge_strategy="dict",  # Keep outputs separate
    ),
    rngs=nnx.Rngs(0),
)

source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(30))
pipeline = from_source(source3, batch_size=8).add(OperatorNode(parallel_dict))
batch = next(iter(pipeline))

print()
print("PARALLEL Strategy (merge='dict'):")
print(f"  Output keys: {list(batch['image'].keys())}")
print("  Each operator output accessible separately")

# %% [markdown]
"""
## Part 4: Weighted Parallel Strategy

Apply operators in parallel with learnable or fixed weights.
Useful for AutoAugment-style learned augmentation policies.
"""

# %%
# WEIGHTED_PARALLEL: Weighted combination of augmentations
op1 = make_brightness_op(0.2, seed=40)
op2 = make_contrast_op(1.4, seed=41)
op3 = make_noise_op(0.03, seed=42)

weighted_parallel = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.WEIGHTED_PARALLEL,
        operators=[op1, op2, op3],
        weights=[0.5, 0.3, 0.2],  # 50% brightness, 30% contrast, 20% noise
        learnable_weights=False,  # Set True for gradient-based learning
    ),
    rngs=nnx.Rngs(0),
)

source4 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(40))
pipeline = from_source(source4, batch_size=16).add(OperatorNode(weighted_parallel))
batch = next(iter(pipeline))

print("WEIGHTED_PARALLEL Strategy:")
print("  Weights: [0.5 (Brightness), 0.3 (Contrast), 0.2 (Noise)]")
print("  Output: weighted sum of augmented versions")
print(f"  Shape: {batch['image'].shape}")

# %% [markdown]
"""
## Part 5: Ensemble Strategies

Ensemble strategies apply operators in parallel, then reduce with
mathematical operations. Ideal for model ensemble predictions.

### Reduction Modes

| Strategy | Reduction | Formula |
|----------|-----------|---------|
| `ENSEMBLE_MEAN` | Average | (op₁ + op₂ + ... + opₙ) / n |
| `ENSEMBLE_SUM` | Sum | op₁ + op₂ + ... + opₙ |
| `ENSEMBLE_MAX` | Maximum | max(op₁, op₂, ..., opₙ) |
| `ENSEMBLE_MIN` | Minimum | min(op₁, op₂, ..., opₙ) |
"""

# %%
# ENSEMBLE_MEAN: Average of multiple augmentations
ensemble_ops = [
    make_brightness_op(0.1, seed=50),
    make_brightness_op(-0.1, seed=51),
    make_contrast_op(1.2, seed=52),
]

ensemble_mean = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.ENSEMBLE_MEAN,
        operators=ensemble_ops,
    ),
    rngs=nnx.Rngs(0),
)

source5 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(50))
pipeline = from_source(source5, batch_size=16).add(OperatorNode(ensemble_mean))
batch = next(iter(pipeline))

print("ENSEMBLE_MEAN Strategy:")
print("  Operators: [Bright+0.1, Bright-0.1, Contrast×1.2]")
print("  Output: element-wise mean of 3 versions")

# %%
# ENSEMBLE_MAX: Take maximum across augmentations
# Useful for conservative augmentation (keeps brightest values)
ensemble_max = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.ENSEMBLE_MAX,
        operators=[
            make_brightness_op(0.0, seed=60),  # Original
            make_brightness_op(0.2, seed=61),  # Brighter
        ],
    ),
    rngs=nnx.Rngs(0),
)

source6 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(60))
pipeline = from_source(source6, batch_size=16).add(OperatorNode(ensemble_max))
batch = next(iter(pipeline))

print()
print("ENSEMBLE_MAX Strategy:")
print("  Takes pixel-wise maximum across augmented versions")
print("  Useful for: highlight preservation, conservative blending")

# %% [markdown]
"""
## Part 6: Branching Strategy

Route data through different operator branches based on conditions.
Uses `jax.lax.switch` for JIT-compatible branching.

### Key Concept: Integer Routing

The router function MUST return an integer index (0, 1, 2, ...) to select
which operator to apply. This is required for JAX tracing compatibility.

```python
def router(data):
    # Return integer index, not string or boolean
    label = data["label"]
    return jax.lax.cond(label > 5, lambda: 1, lambda: 0)
```
"""


# %%
# BRANCHING: Route based on label
def label_router(data):
    """Route based on label value.

    Returns:
        0 if label <= 5 (bright augmentation)
        1 if label > 5 (contrast augmentation)
    """
    label = data["label"]
    # Must use jax.lax operations for traced values
    return jax.lax.cond(label > 5, lambda: 1, lambda: 0)


branch_ops = [
    make_brightness_op(0.2, seed=70),  # Branch 0: for labels 0-5
    make_contrast_op(1.4, seed=71),  # Branch 1: for labels 6-9
]

branching = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.BRANCHING,
        operators=branch_ops,
        router=label_router,
        default_branch=0,  # Fallback if router fails
    ),
    rngs=nnx.Rngs(0),
)

source7 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(70))
pipeline = from_source(source7, batch_size=16).add(OperatorNode(branching))
batch = next(iter(pipeline))

print("BRANCHING Strategy:")
print("  Router: label <= 5 → Brightness, label > 5 → Contrast")
print(f"  Batch labels: {batch['label'][:8]}...")
print("  Each sample routed to appropriate augmentation branch")

# %% [markdown]
"""
## Part 7: Conditional Parallel Strategy

Apply operators in parallel, but only include those where condition is True.
False-condition operators return identity (for vmap compatibility).
"""


# %%
# CONDITIONAL_PARALLEL: Apply subset of operators based on conditions
def has_high_variance(data_dict):
    """Apply if image has high variance (needs smoothing)."""
    return jnp.var(data_dict["image"]) > 0.05


def has_low_mean(data_dict):
    """Apply if image is dark (needs brightening)."""
    return jnp.mean(data_dict["image"]) < 0.4


cond_parallel = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
        operators=[
            make_noise_op(0.02, seed=80),  # Smoothing noise (low std)
            make_brightness_op(0.15, seed=81),  # Brightening
        ],
        conditions=[has_high_variance, has_low_mean],
        merge_strategy="mean",
    ),
    rngs=nnx.Rngs(0),
)

print("CONDITIONAL_PARALLEL Strategy:")
print("  - Noise: only if variance > 0.05")
print("  - Brightness: only if mean < 0.4")
print("  Result: mean of applicable operators")

# %% [markdown]
"""
## Part 8: Building Complex Pipelines

Combine composition strategies for sophisticated augmentation pipelines.
"""

# %%
# Build a production-ready augmentation pipeline:
# 1. First, normalize (always)
# 2. Then, apply random augmentation branch


def normalize_op():
    """Create normalization operator."""

    def normalize_fn(element, key=None):
        del key  # Unused - deterministic operator
        image = element.data["image"]
        # Simple min-max normalization
        normalized = (image - jnp.min(image)) / (jnp.max(image) - jnp.min(image) + 1e-8)
        return element.update_data({"image": normalized})

    return ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=normalize_fn,
        rngs=nnx.Rngs(0),
    )


# Augmentation ensemble: average of multiple augmentations
aug_ensemble = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.ENSEMBLE_MEAN,
        operators=[
            make_brightness_op(0.1, seed=90),
            make_contrast_op(1.1, seed=91),
        ],
    ),
    rngs=nnx.Rngs(0),
)

# Full pipeline: normalize → augment
full_pipeline_op = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[normalize_op(), aug_ensemble],
    ),
    rngs=nnx.Rngs(0),
)

source8 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(90))
pipeline = from_source(source8, batch_size=32).add(OperatorNode(full_pipeline_op))

# Process all data
total_samples = 0
for batch in pipeline:
    total_samples += batch["image"].shape[0]

print("Complex Pipeline (Nested Composition):")
print("  SEQUENTIAL [")
print("    Normalize,")
print("    ENSEMBLE_MEAN [Brightness, Contrast]")
print("  ]")
print(f"  Processed: {total_samples} samples")

# %% [markdown]
"""
## Results Summary

### Strategy Selection Guide

| Use Case | Recommended Strategy |
|----------|---------------------|
| Standard augmentation chain | `SEQUENTIAL` |
| Skip augmentation conditionally | `CONDITIONAL_SEQUENTIAL` |
| Multi-view generation | `PARALLEL` (merge='dict') |
| Averaged augmentation | `PARALLEL` (merge='mean') or `ENSEMBLE_MEAN` |
| Learnable augmentation policy | `WEIGHTED_PARALLEL` |
| Class-specific augmentation | `BRANCHING` |
| Test-time augmentation | `ENSEMBLE_MEAN` |

### JAX Compatibility Notes

| Pattern | Why Needed |
|---------|------------|
| Integer routing | `jax.lax.switch` requires int index |
| `jax.lax.cond` for conditions | Python `if` breaks tracing |
| Fixed output shapes | vmap requires consistent shapes |
| No dict key from traced values | Dict keys must be static |
"""

# %% [markdown]
"""
## Next Steps

- [DAG Fundamentals](../advanced/dag/01_dag_fundamentals_guide.ipynb) - Pipeline architecture
- [Sharding Guide](../advanced/distributed/02_sharding_guide.ipynb) - Distributed pipelines
- [Performance Guide](../advanced/performance/01_optimization_guide.ipynb) - Optimization tips
"""


# %%
def main():
    """Run the composition strategies tutorial."""
    print("=" * 60)
    print("Composition Strategies Tutorial")
    print("=" * 60)

    # Create data
    np.random.seed(42)
    data = {
        "image": np.random.rand(50, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (50,)).astype(np.int32),
    }
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    # Demo: Sequential
    print()
    print("1. SEQUENTIAL: Chain operators")
    seq = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[
                make_brightness_op(0.1, seed=1),
                make_contrast_op(1.2, seed=2),
            ],
        ),
        rngs=nnx.Rngs(0),
    )
    pipeline = from_source(source, batch_size=16).add(OperatorNode(seq))
    batch = next(iter(pipeline))
    print(f"   Output shape: {batch['image'].shape}")

    # Demo: Ensemble Mean
    print()
    print("2. ENSEMBLE_MEAN: Average augmentations")
    source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))
    ens = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[
                make_brightness_op(0.1, seed=3),
                make_brightness_op(-0.1, seed=4),
            ],
        ),
        rngs=nnx.Rngs(0),
    )
    pipeline = from_source(source2, batch_size=16).add(OperatorNode(ens))
    batch = next(iter(pipeline))
    print(f"   Output range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")

    # Demo: Branching
    print()
    print("3. BRANCHING: Route by label")
    source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))
    branch = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[
                make_brightness_op(0.2, seed=5),
                make_contrast_op(1.3, seed=6),
            ],
            router=label_router,
            default_branch=0,
        ),
        rngs=nnx.Rngs(0),
    )
    pipeline = from_source(source3, batch_size=16).add(OperatorNode(branch))
    batch = next(iter(pipeline))
    print(f"   Labels: {batch['label'][:5]}... → routed to different branches")

    print()
    print("=" * 60)
    print("Tutorial completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
