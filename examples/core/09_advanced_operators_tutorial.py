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
"""

# %% [markdown]
"""
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
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import (
    SelectorOperator,
    SelectorOperatorConfig,
)
from datarax.operators.modality.image import (
    BrightnessOperator,
    BrightnessOperatorConfig,
    ContrastOperator,
    ContrastOperatorConfig,
    NoiseOperator,
    NoiseOperatorConfig,
)
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)
from datarax.sources import MemorySource, MemorySourceConfig

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# %% [markdown]
"""
## Part 1: Setup - Create Sample Data

We'll create a simple image dataset for demonstrating the operators.
"""

# %%
# Create sample image data
np.random.seed(42)
num_samples = 100
data = {
    "image": np.random.rand(num_samples, 32, 32, 3).astype(np.float32),
    "label": np.random.randint(0, 10, (num_samples,)).astype(np.int32),
}

print(f"Dataset: {num_samples} samples")
print(
    f"  image: {data['image'].shape}, range [{data['image'].min():.2f}, {data['image'].max():.2f}]"
)
print(f"  label: {data['label'].shape}")

# %% [markdown]
"""
## Part 2: ProbabilisticOperator

`ProbabilisticOperator` wraps any operator and applies it with a configured
probability. This is essential for stochastic augmentation pipelines.

### Behavior by Probability

| Probability | Mode | Behavior |
|-------------|------|----------|
| `p = 0.0` | Deterministic | Never apply (passthrough) |
| `0 < p < 1` | Stochastic | Apply with probability p |
| `p = 1.0` | Deterministic | Always apply |

### JAX Compatibility

Uses `jax.lax.cond` for JIT-compatible conditional execution.
"""

# %%
# Create a child operator (brightness adjustment)
brightness_op = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(0.2, 0.2),  # Fixed +0.2 brightness
        stochastic=False,
    ),
    rngs=nnx.Rngs(0),
)

# Wrap with 50% probability
prob_brightness = ProbabilisticOperator(
    ProbabilisticOperatorConfig(
        operator=brightness_op,
        probability=0.5,  # Apply to ~50% of samples
    ),
    rngs=nnx.Rngs(augment=42),
)

print("ProbabilisticOperator created:")
print("  Wrapped: BrightnessOperator(+0.2)")
print("  Probability: 50%")
print(f"  Stochastic: {prob_brightness.config.stochastic}")

# %%
# Test the probabilistic operator
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = from_source(source, batch_size=32).add(OperatorNode(prob_brightness))

batch = next(iter(pipeline))
original_mean = data["image"][:32].mean()
transformed_mean = batch["image"].mean()

print()
print("Probabilistic application results:")
print(f"  Original mean: {original_mean:.4f}")
print(f"  Transformed mean: {transformed_mean:.4f}")
print("  (Some samples brightened, some unchanged)")

# %%
# Compare different probability values
print()
print("Effect of probability values:")

for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    # Create operator with specific probability
    prob_op = ProbabilisticOperator(
        ProbabilisticOperatorConfig(
            operator=brightness_op,
            probability=p,
        ),
        rngs=nnx.Rngs(augment=42),
    )

    # Apply to batch
    source_p = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    pipeline_p = from_source(source_p, batch_size=100).add(OperatorNode(prob_op))
    batch_p = next(iter(pipeline_p))

    # Compare means
    delta = batch_p["image"].mean() - data["image"].mean()
    mode = "deterministic" if p in [0.0, 1.0] else "stochastic"
    print(f"  p={p:.2f} ({mode:12s}): mean delta = {delta:+.4f}")

# %% [markdown]
"""
## Part 3: SelectorOperator

`SelectorOperator` randomly selects ONE operator from a list to apply per sample.
Uses weighted random selection with configurable weights.

### Key Features

- Wraps multiple operators with random selection
- Configurable weights (defaults to uniform)
- Uses `jax.lax.switch` for JIT compatibility
- Always stochastic (always makes a random choice)
"""

# %%
# Create multiple operators for selection
op_bright = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(0.15, 0.15),
        stochastic=False,
    ),
    rngs=nnx.Rngs(1),
)

op_contrast = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image",
        contrast_range=(1.3, 1.3),
        stochastic=False,
    ),
    rngs=nnx.Rngs(2),
)

op_noise = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.1,
        stochastic=True,
        stream_name="noise",
    ),
    rngs=nnx.Rngs(noise=3),
)

# Create selector with custom weights
selector = SelectorOperator(
    SelectorOperatorConfig(
        operators=[op_bright, op_contrast, op_noise],
        weights=[0.5, 0.3, 0.2],  # 50% brightness, 30% contrast, 20% noise
    ),
    rngs=nnx.Rngs(augment=100),
)

print("SelectorOperator created:")
print("  Operators: [Brightness, Contrast, Noise]")
print("  Weights: [50%, 30%, 20%]")
print(f"  Always stochastic: {selector.config.stochastic}")

# %%
# Apply selector and observe which operators were chosen
source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))
pipeline2 = from_source(source2, batch_size=50).add(OperatorNode(selector))

batch2 = next(iter(pipeline2))

# Analyze results by comparing to each operator's expected output
print()
print("Selector results (each sample gets ONE operator):")
print(f"  Batch shape: {batch2['image'].shape}")
print(f"  Output range: [{batch2['image'].min():.3f}, {batch2['image'].max():.3f}]")

# %%
# Create a uniform selector (equal weights)
uniform_selector = SelectorOperator(
    SelectorOperatorConfig(
        operators=[op_bright, op_contrast, op_noise],
        weights=None,  # Defaults to uniform [1/3, 1/3, 1/3]
    ),
    rngs=nnx.Rngs(augment=200),
)

print()
print("Uniform selector (equal probability):")
print(f"  Normalized weights: {uniform_selector.weights}")

# %% [markdown]
"""
## Part 4: PatchDropoutOperator

`PatchDropoutOperator` applies patch-based occlusion by dropping random
rectangular regions from images. Useful for occlusion robustness training.

### Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_patches` | Number of patches to drop | 4 |
| `patch_size` | Size as (height, width) | (8, 8) |
| `drop_value` | Fill value for dropped regions | 0.0 |
"""

# %%
# Create patch dropout operator
patch_dropout = PatchDropoutOperator(
    PatchDropoutOperatorConfig(
        field_key="image",
        num_patches=4,
        patch_size=(8, 8),
        drop_value=0.0,  # Black patches
        stochastic=True,  # Random positions per sample
        stream_name="patch",
    ),
    rngs=nnx.Rngs(patch=42),
)

print("PatchDropoutOperator created:")
print("  num_patches: 4")
print("  patch_size: (8, 8)")
print("  drop_value: 0.0 (black)")
print(f"  Stochastic: {patch_dropout.config.stochastic}")

# %%
# Apply patch dropout
source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))
pipeline3 = from_source(source3, batch_size=16).add(OperatorNode(patch_dropout))

batch3 = next(iter(pipeline3))

# Calculate how much of the image is covered by patches
# Each 8x8 patch covers 64 pixels out of 32x32=1024
coverage = (4 * 8 * 8) / (32 * 32) * 100

print()
print("Patch dropout results:")
print(f"  Output shape: {batch3['image'].shape}")
print(f"  Expected coverage: ~{coverage:.1f}% of image dropped")
print("  (Patches may overlap, actual coverage varies)")

# %%
# Deterministic patch dropout (fixed positions)
det_patch_dropout = PatchDropoutOperator(
    PatchDropoutOperatorConfig(
        field_key="image",
        num_patches=2,
        patch_size=(16, 16),
        drop_value=0.5,  # Gray patches
        stochastic=False,  # Same positions every time
    ),
    rngs=nnx.Rngs(0),
)

print()
print("Deterministic patch dropout:")
print("  Uses fixed random seed for reproducible positions")
print(f"  Stochastic: {det_patch_dropout.config.stochastic}")

# %% [markdown]
"""
## Part 5: Building AutoAugment-Style Pipelines

Combine these advanced operators to build sophisticated augmentation
pipelines similar to AutoAugment.
"""


# %%
def create_autoaugment_pipeline():
    """Create an AutoAugment-style augmentation pipeline.

    Pipeline structure:
    1. Probabilistically apply brightness (60%)
    2. Probabilistically apply contrast (60%)
    3. Randomly select one of: noise, patch dropout (weighted)
    """
    # Create base operators
    bright = BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.2, 0.2),
            stochastic=True,
            stream_name="bright",
        ),
        rngs=nnx.Rngs(bright=10),
    )

    contrast = ContrastOperator(
        ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.8, 1.2),
            stochastic=True,
            stream_name="contrast",
        ),
        rngs=nnx.Rngs(contrast=20),
    )

    noise = NoiseOperator(
        NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.05,
            stochastic=True,
            stream_name="noise",
        ),
        rngs=nnx.Rngs(noise=30),
    )

    patch = PatchDropoutOperator(
        PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(8, 8),
            drop_value=0.0,
            stochastic=True,
            stream_name="patch",
        ),
        rngs=nnx.Rngs(patch=40),
    )

    # Wrap with probabilistic application
    prob_bright = ProbabilisticOperator(
        ProbabilisticOperatorConfig(operator=bright, probability=0.6),
        rngs=nnx.Rngs(augment=100),
    )

    prob_contrast = ProbabilisticOperator(
        ProbabilisticOperatorConfig(operator=contrast, probability=0.6),
        rngs=nnx.Rngs(augment=200),
    )

    # Create selector for final augmentation
    final_selector = SelectorOperator(
        SelectorOperatorConfig(
            operators=[noise, patch],
            weights=[0.7, 0.3],  # 70% noise, 30% patch dropout
        ),
        rngs=nnx.Rngs(augment=300),
    )

    return prob_bright, prob_contrast, final_selector


# Build the pipeline
prob_bright, prob_contrast, final_selector = create_autoaugment_pipeline()

print("AutoAugment-style pipeline:")
print("  1. Brightness (60% probability)")
print("  2. Contrast (60% probability)")
print("  3. Selector: Noise (70%) or PatchDropout (30%)")

# %%
# Apply the AutoAugment-style pipeline
source4 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(3))

pipeline4 = (
    from_source(source4, batch_size=32)
    .add(OperatorNode(prob_bright))
    .add(OperatorNode(prob_contrast))
    .add(OperatorNode(final_selector))
)

# Process and collect statistics
stats = {"batches": 0, "samples": 0}
for batch in pipeline4:
    stats["batches"] += 1
    stats["samples"] += batch["image"].shape[0]

print()
print("Pipeline execution results:")
print(f"  Processed: {stats['samples']} samples in {stats['batches']} batches")

# %% [markdown]
"""
## Part 6: Understanding JAX Compatibility

All advanced operators use specific patterns for JAX compatibility:

### Key Patterns

| Pattern | Used In | Purpose |
|---------|---------|---------|
| `jax.lax.cond` | ProbabilisticOperator | Conditional execution |
| `jax.lax.switch` | SelectorOperator | Multi-way branching |
| `jax.lax.fori_loop` | PatchDropoutOperator | Loop over patches |
| Pre-generated random params | All | vmap compatibility |

### Why These Patterns?

1. **No Python if statements**: Traced JAX values can't be used in Python conditionals
2. **Pre-generated randoms**: Avoids RNG state mutations inside vmap
3. **Fixed output shapes**: vmap requires consistent shapes across branches
"""


# %%
# Demonstrate JIT compatibility
@jax.jit
def jit_apply(op, source):
    """JIT-compiled application of operator."""
    pipeline = from_source(source, batch_size=16).add(OperatorNode(op))
    batch = next(iter(pipeline))
    return batch["image"].mean()


# This works because operators use JAX-compatible patterns
source_jit = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(4))
result = jit_apply(prob_bright, source_jit)

print("JIT Compilation:")
print("  jit_apply() executed successfully")
print(f"  Result: {result:.4f}")

# %% [markdown]
"""
## Results Summary

### Advanced Operators

| Operator | Purpose | Key Config |
|----------|---------|------------|
| `ProbabilisticOperator` | Apply with probability | `probability` (0-1) |
| `SelectorOperator` | Random selection | `operators`, `weights` |
| `PatchDropoutOperator` | Spatial dropout | `num_patches`, `patch_size` |

### Use Cases

| Use Case | Operator(s) |
|----------|-------------|
| Stochastic augmentation | `ProbabilisticOperator` |
| AutoAugment policies | `SelectorOperator` + `ProbabilisticOperator` |
| Occlusion robustness | `PatchDropoutOperator` |
| Test-time augmentation | All, with various settings |
"""

# %% [markdown]
"""
## Next Steps

- [Composition Strategies](08_composition_strategies_tutorial.ipynb) - Combine operators
- [MixUp/CutMix Tutorial](../advanced/augmentation/01_mixup_cutmix_tutorial.ipynb) - Batch mixing
- [Performance Guide](../advanced/performance/01_optimization_guide.ipynb) - Optimization
"""


# %%
def main():
    """Run the advanced operators tutorial."""
    print("=" * 60)
    print("Advanced Operators Tutorial")
    print("=" * 60)

    # Create data
    np.random.seed(42)
    data = {
        "image": np.random.rand(50, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, (50,)).astype(np.int32),
    }

    # Demo 1: ProbabilisticOperator
    print()
    print("1. ProbabilisticOperator (p=0.5):")
    bright = BrightnessOperator(
        BrightnessOperatorConfig(field_key="image", brightness_range=(0.2, 0.2)),
        rngs=nnx.Rngs(0),
    )
    prob = ProbabilisticOperator(
        ProbabilisticOperatorConfig(operator=bright, probability=0.5),
        rngs=nnx.Rngs(augment=0),
    )
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
    pipeline = from_source(source, batch_size=50).add(OperatorNode(prob))
    batch = next(iter(pipeline))
    print(f"   Output mean: {batch['image'].mean():.4f}")

    # Demo 2: SelectorOperator
    print()
    print("2. SelectorOperator (3 operators):")
    op1 = BrightnessOperator(
        BrightnessOperatorConfig(field_key="image", brightness_range=(0.1, 0.1)),
        rngs=nnx.Rngs(1),
    )
    op2 = ContrastOperator(
        ContrastOperatorConfig(field_key="image", contrast_range=(1.2, 1.2)),
        rngs=nnx.Rngs(2),
    )
    selector = SelectorOperator(
        SelectorOperatorConfig(operators=[op1, op2]),
        rngs=nnx.Rngs(augment=10),
    )
    source2 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(1))
    pipeline2 = from_source(source2, batch_size=50).add(OperatorNode(selector))
    batch2 = next(iter(pipeline2))
    print(f"   Output mean: {batch2['image'].mean():.4f}")

    # Demo 3: PatchDropoutOperator
    print()
    print("3. PatchDropoutOperator (4 patches):")
    patch = PatchDropoutOperator(
        PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 8),
            stochastic=True,
            stream_name="patch",
        ),
        rngs=nnx.Rngs(patch=20),
    )
    source3 = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(2))
    pipeline3 = from_source(source3, batch_size=50).add(OperatorNode(patch))
    batch3 = next(iter(pipeline3))
    print(f"   Output mean: {batch3['image'].mean():.4f} (lower due to black patches)")

    print()
    print("=" * 60)
    print("Tutorial completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
