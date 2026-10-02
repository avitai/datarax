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
# MixUp and CutMix Augmentation Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~20 min |
| **Prerequisites** | Operators Tutorial, Augmentation Quick Reference |
| **Format** | Python + Jupyter |
| **Memory** | ~1 GB RAM |

## Overview

MixUp and CutMix are powerful batch-level augmentation techniques that mix
pairs of samples to create virtual training examples. Unlike element-level
augmentations, these require access to multiple samples simultaneously.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Understand MixUp and CutMix augmentation techniques
2. Use `BatchMixOperator` for both modes
3. Train on mixed records with the loss the MixUp and CutMix papers use, which reads each
   record's partner and the mixing ratio from the batch
4. Tune the alpha parameter for mixing strength
5. Visualize mixed samples and the mixing ratios
"""

# %% [markdown]
"""
## Setup

```bash
uv pip install "datarax[data]" matplotlib
# Prepare CIFAR-10 once as ArrayRecord, in a process of its own: preparing imports
# TensorFlow (the tfds extra); the example reads the prepared copy without it
uv pip install "datarax[tfds]"
python -c "import tensorflow_datasets as tfds; tfds.builder('cifar10', file_format='array_record').download_and_prepare()"
```
"""

# %%
# Core imports

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from substrax.artifacts import resolve_output_dir

from datarax.core.config import BatchMixOperatorConfig
from datarax.core.state_keys import MIX_LAMBDA, MIX_PARTNER
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.operators.batch_mix_operator import BatchMixOperator

# Datarax imports
from datarax.pipeline import Pipeline
from datarax.sources import TFDSEagerConfig, TFDSEagerSource


# %% [markdown]
"""
## Background: Batch-Level Augmentation

### Why MixUp and CutMix?

Standard augmentations (rotation, brightness, noise) operate on individual
samples. MixUp and CutMix operate on pairs of samples, creating "virtual"
training examples that don't exist in the original dataset.

### Key Differences

| Aspect | Element-Level | Batch-Level |
|--------|---------------|-------------|
| Scope | Single sample | Sample pairs |
| Labels | Unchanged | Unchanged; the loss mixes |
| Example | Rotation, Noise | MixUp, CutMix |
| Implementation | `vmap` per element | Full batch access |

### MixUp Formula

MixUp creates linear interpolations of each record with a partner record of the batch:

```
x_mixed = λ * x + (1 - λ) * x[partner]
loss    = λ * CE(pred, y) + (1 - λ) * CE(pred, y[partner])
```

Where λ ~ Beta(α, α).

### CutMix Formula

CutMix pastes a box of the partner's image into each image, and λ is the fraction of the
image kept:

```
x_mixed = mask * x + (1 - mask) * x[partner]
loss    = λ * CE(pred, y) + (1 - λ) * CE(pred, y[partner])
```

The labels themselves are never mixed. The operator records each record's partner row in
`states[MIX_PARTNER]` and λ in `batch_state[MIX_LAMBDA]`, and the loss reads them: this is the
loss of the published implementations (`mixup_criterion` in facebookresearch/mixup-cifar10
`train.py`, and the CutMix branch of clovaai/CutMix-PyTorch `train.py`). Integer class labels
stay integers, so nothing ever averages two class indices into a third.
"""

# %% [markdown]
"""
## Part 1: Load CIFAR-10 Data
"""

# %%
# CIFAR-10 constants
CIFAR10_MEAN = jnp.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = jnp.array([0.2470, 0.2435, 0.2616])
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

BATCH_SIZE = 32
NUM_CLASSES = 10


# %%
def preprocess_cifar10(element, key=None):  # noqa: ARG001
    """Normalize CIFAR-10 images; labels stay integer class indices."""
    del key
    image = element.data["image"]

    # Normalize to [0, 1] then standardize
    image = image.astype(jnp.float32) / 255.0
    image = (image - CIFAR10_MEAN) / CIFAR10_STD

    return element.update_data({"image": image})


preprocessor = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=preprocess_cifar10,
    rngs=nnx.Rngs(0),
)


# %%
def create_base_pipeline(num_samples=256):
    """Create CIFAR-10 pipeline with preprocessing."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="cifar10",
            split=f"train[:{num_samples}]",
        ),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_cifar10,
        rngs=nnx.Rngs(0),
    )

    return Pipeline(
        source=source, stages=[prep], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0), shuffle=True
    )


print("Base pipeline factory created")

# %% [markdown]
"""
## Part 2: MixUp Augmentation

MixUp creates linear interpolations between random pairs of samples.
"""

# %%
# Create MixUp operator
mixup_op = BatchMixOperator(
    BatchMixOperatorConfig(
        mode="mixup",
        alpha=0.4,  # Beta distribution parameter
        data_field="image",
        stochastic=True,
        stream_name="mixup",
    ),
    rngs=nnx.Rngs(mixup=100),
)

print("MixUp operator created:")
print("  mode: mixup")
print("  alpha: 0.4 (moderate mixing)")
print("  data_field: image (labels are left as they are)")


# %%
def create_mixup_pipeline(alpha=0.4, seed=42):
    """Create CIFAR-10 pipeline with MixUp."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="cifar10",
            split="train[:256]",
        ),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_cifar10,
        rngs=nnx.Rngs(0),
    )

    mixup = BatchMixOperator(
        BatchMixOperatorConfig(
            mode="mixup",
            alpha=alpha,
            data_field="image",
            stochastic=True,
            stream_name="mixup",
        ),
        rngs=nnx.Rngs(mixup=100 + seed),
    )

    return Pipeline(
        source=source, stages=[prep, mixup], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0), shuffle=True
    )


# Get MixUp batch
mixup_pipeline = create_mixup_pipeline(alpha=0.4)
mixup_batch = next(iter(mixup_pipeline))

print("\nMixUp batch:")
print(f"  Image shape: {mixup_batch['image'].shape}")
print(f"  Labels (unchanged integers): {mixup_batch['label'][:8]}")
print(f"  Partner of each record: {mixup_batch.states[MIX_PARTNER][:8]}")
print(f"  Mixing ratio λ: {float(mixup_batch.batch_state[MIX_LAMBDA]):.3f}")

# %% [markdown]
"""
### Visualize MixUp Results
"""

# %%
output_dir = resolve_output_dir("examples").path


def denormalize_cifar10(images):
    """Denormalize CIFAR-10 images for display."""
    images = np.array(images)
    images = images * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
    return np.clip(images, 0, 1)


def mixed_title(batch, i):
    """Record ``i``'s class and its partner's, each with its weight in the loss."""
    lam = float(batch.batch_state[MIX_LAMBDA])
    own = CIFAR10_CLASSES[int(batch["label"][i])][:3]
    other = CIFAR10_CLASSES[int(batch["label"][batch.states[MIX_PARTNER][i]])][:3]
    return f"{own}:{lam:.1f} {other}:{1 - lam:.1f}"


# Get original batch for comparison
base_pipeline = create_base_pipeline()
original_batch = next(iter(base_pipeline))

# Plot MixUp samples
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("MixUp: Original vs Mixed Samples", fontsize=14)

for i in range(8):
    # Original
    img_orig = denormalize_cifar10(original_batch["image"][i])
    axes[0, i].imshow(img_orig)
    axes[0, i].axis("off")
    axes[0, i].set_title(CIFAR10_CLASSES[int(original_batch["label"][i])], fontsize=8)

    # Mixed
    img_mixed = denormalize_cifar10(mixup_batch["image"][i])
    axes[1, i].imshow(img_mixed)
    axes[1, i].axis("off")

    # The record's own class with weight λ, its partner's with 1 - λ
    title = mixed_title(mixup_batch, i)
    axes[1, i].set_title(title, fontsize=8)

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("MixUp", fontsize=10)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-cifar-mixup-samples.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-cifar-mixup-samples.png'}")

# %% [markdown]
"""
## Part 3: CutMix Augmentation

CutMix cuts a rectangular patch from one image and pastes it onto another.
"""

# %%
# Create CutMix operator
cutmix_op = BatchMixOperator(
    BatchMixOperatorConfig(
        mode="cutmix",
        alpha=1.0,  # Uniform cut sizes
        data_field="image",
        stochastic=True,
        stream_name="cutmix",
    ),
    rngs=nnx.Rngs(cutmix=200),
)

print("CutMix operator created:")
print("  mode: cutmix")
print("  alpha: 1.0 (uniform cut sizes)")


# %%
def create_cutmix_pipeline(alpha=1.0, seed=42):
    """Create CIFAR-10 pipeline with CutMix."""
    source = TFDSEagerSource(
        TFDSEagerConfig(
            name="cifar10",
            split="train[:256]",
        ),
    )

    prep = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=preprocess_cifar10,
        rngs=nnx.Rngs(0),
    )

    cutmix = BatchMixOperator(
        BatchMixOperatorConfig(
            mode="cutmix",
            alpha=alpha,
            data_field="image",
            stochastic=True,
            stream_name="cutmix",
        ),
        rngs=nnx.Rngs(cutmix=200 + seed),
    )

    return Pipeline(
        source=source, stages=[prep, cutmix], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0), shuffle=True
    )


# Get CutMix batch
cutmix_pipeline = create_cutmix_pipeline(alpha=1.0)
cutmix_batch = next(iter(cutmix_pipeline))

print("\nCutMix batch:")
print(f"  Image shape: {cutmix_batch['image'].shape}")
print(f"  Fraction of each image kept, λ: {float(cutmix_batch.batch_state[MIX_LAMBDA]):.3f}")

# %%
# Plot CutMix samples
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("CutMix: Original vs Mixed Samples", fontsize=14)

for i in range(8):
    # Original
    img_orig = denormalize_cifar10(original_batch["image"][i])
    axes[0, i].imshow(img_orig)
    axes[0, i].axis("off")
    axes[0, i].set_title(CIFAR10_CLASSES[int(original_batch["label"][i])], fontsize=8)

    # CutMix
    img_cut = denormalize_cifar10(cutmix_batch["image"][i])
    axes[1, i].imshow(img_cut)
    axes[1, i].axis("off")

    # The record's own class with weight λ, its partner's with 1 - λ
    title = mixed_title(cutmix_batch, i)
    axes[1, i].set_title(title, fontsize=8)

axes[0, 0].set_ylabel("Original", fontsize=10)
axes[1, 0].set_ylabel("CutMix", fontsize=10)

plt.tight_layout()
plt.savefig(
    output_dir / "cv-cifar-cutmix-samples.png", dpi=150, bbox_inches="tight", facecolor="white"
)
plt.close()
print(f"Saved: {output_dir / 'cv-cifar-cutmix-samples.png'}")

# %% [markdown]
"""
## Part 4: Alpha Parameter Effect

The alpha parameter controls the Beta distribution for mixing ratio λ.

- α = 0.2: Strong bias toward original (λ near 0 or 1)
- α = 1.0: Uniform distribution (any λ equally likely)
- α = 4.0: Strong bias toward 50/50 mix (λ near 0.5)
"""

# %%
# Visualize effect of different alpha values
alphas = [0.2, 0.5, 1.0, 2.0]

fig, axes = plt.subplots(len(alphas), 8, figsize=(16, 8))
fig.suptitle("MixUp with Different Alpha Values", fontsize=14)

for row, alpha in enumerate(alphas):
    pipeline = create_mixup_pipeline(alpha=alpha, seed=row * 100)
    batch = next(iter(pipeline))

    for col in range(8):
        img = denormalize_cifar10(batch["image"][col])
        axes[row, col].imshow(img)
        axes[row, col].axis("off")

        if col == 0:
            axes[row, col].set_ylabel(
                f"α={alpha}", fontsize=10, rotation=0, ha="right", va="center"
            )

plt.tight_layout()
plt.savefig(output_dir / "cv-cifar-mix-alpha.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'cv-cifar-mix-alpha.png'}")

# %% [markdown]
"""
## Part 5: The Loss for Mixed Records

The loss reads what the operator wrote: each record's partner and λ. This is the loss the
papers train with (`mixup_criterion` in facebookresearch/mixup-cifar10, commit `eaff31a`,
`train.py:137-138`; clovaai/CutMix-PyTorch, commit `2d8eb68`, `train.py:237-240`). For integer
labels it equals cross-entropy against the soft label `λ * onehot(y) + (1 - λ) * onehot(y[partner])`,
which the cell below checks.
"""


# %%
def mixed_cross_entropy(logits, batch):
    """Cross-entropy of mixed records: λ CE(y) + (1 - λ) CE(y[partner]), per record."""
    lam = batch.batch_state[MIX_LAMBDA]
    partner = batch.states[MIX_PARTNER]
    labels = batch["label"]
    log_probs = jax.nn.log_softmax(logits)
    own = -jnp.take_along_axis(log_probs, labels[:, None], axis=1)[:, 0]
    other = -jnp.take_along_axis(log_probs, labels[partner][:, None], axis=1)[:, 0]
    return lam * own + (1 - lam) * other


logits = jax.random.normal(jax.random.key(0), (BATCH_SIZE, NUM_CLASSES))
for name, batch in (("MixUp", mixup_batch), ("CutMix", cutmix_batch)):
    lam = batch.batch_state[MIX_LAMBDA]
    soft = lam * jax.nn.one_hot(batch["label"], NUM_CLASSES) + (1 - lam) * jax.nn.one_hot(
        batch["label"][batch.states[MIX_PARTNER]], NUM_CLASSES
    )
    soft_ce = -jnp.sum(soft * jax.nn.log_softmax(logits), axis=1)
    difference = float(jnp.max(jnp.abs(mixed_cross_entropy(logits, batch) - soft_ce)))
    print(f"{name}: mixed loss vs soft-label cross-entropy, max difference {difference:.2e}")

# %% [markdown]
"""
### How strongly records are mixed

Each batch draws one λ. A mixed record's weight on its own label is λ, so the distribution of
λ over batches shows how "hard" the targets are: near 1, records are mostly themselves.
"""

# %%
# Collect λ over many batches
mixup_lambdas = []
cutmix_lambdas = []

for i in range(20):
    mixup_lambdas.append(
        float(next(iter(create_mixup_pipeline(alpha=0.4, seed=i))).batch_state[MIX_LAMBDA])
    )
    cutmix_lambdas.append(
        float(next(iter(create_cutmix_pipeline(alpha=1.0, seed=i + 100))).batch_state[MIX_LAMBDA])
    )

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(mixup_lambdas, bins=10, range=(0, 1), alpha=0.7, label="MixUp (α=0.4)", color="blue")
ax.hist(cutmix_lambdas, bins=10, range=(0, 1), alpha=0.7, label="CutMix (α=1.0)", color="orange")
ax.set_xlabel("λ, the weight of each record's own label")
ax.set_ylabel("Batches")
ax.set_title("Mixing ratio over 20 batches")
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "cv-cifar-mix-labels.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {output_dir / 'cv-cifar-mix-labels.png'}")

# %% [markdown]
"""
## Results Summary

### MixUp vs CutMix

| Aspect | MixUp | CutMix |
|--------|-------|--------|
| Operation | Linear blend | Patch paste |
| Visual effect | Ghostly overlap | Cut boundary visible |
| Information | Global features | Local features |
| Best for | General regularization | Localization tasks |

### Recommended Settings

| Use Case | Mode | Alpha |
|----------|------|-------|
| General training | MixUp | 0.2-0.4 |
| Strong regularization | MixUp | 1.0 |
| Object detection | CutMix | 1.0 |
| Combined (timm) | Both | 0.2 MixUp + 1.0 CutMix |

### Key Takeaways

1. **Mixed loss, not mixed labels**: the operator writes each record's partner and λ; the loss
   is λ·CE(y) + (1 − λ)·CE(y[partner]), as in the published implementations
2. **Alpha matters**: Lower α = more "pure" samples, higher α = more mixing
3. **Batch-level**: Uses `apply_batch()` instead of per-element `apply()`
4. **Order**: Apply after preprocessing, before model forward pass
5. **Training only**: Never use during evaluation/inference
"""

# %% [markdown]
"""
## Next Steps

- **Multi-source**: [Interleaved datasets](../multi_source/01_interleaved_tutorial.ipynb)
- **Full training**: [End-to-end CIFAR-10](../training/01_e2e_cifar10_guide.ipynb)
- **Performance**: [Optimization guide](../performance/01_optimization_guide.ipynb)
"""


# %%
def main():
    """Run the MixUp/CutMix tutorial."""
    print("MixUp and CutMix Augmentation Tutorial")
    print("=" * 50)

    # Test MixUp
    mixup_pipeline = create_mixup_pipeline(alpha=0.4)
    mixup_batch = next(iter(mixup_pipeline))
    print(f"MixUp batch shape: {mixup_batch['image'].shape}")
    print(f"MixUp λ = {float(mixup_batch.batch_state[MIX_LAMBDA]):.2f}")

    # Test CutMix
    cutmix_pipeline = create_cutmix_pipeline(alpha=1.0)
    cutmix_batch = next(iter(cutmix_pipeline))
    print(f"CutMix batch shape: {cutmix_batch['image'].shape}")
    print(f"CutMix λ = {float(cutmix_batch.batch_state[MIX_LAMBDA]):.2f}")

    print("\nTutorial completed successfully!")


if __name__ == "__main__":
    main()
