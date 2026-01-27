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
# IMDB Sentiment Analysis Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Datarax pipeline, HuggingFace datasets |
| **Format** | Python + Jupyter |
| **Memory** | ~500 MB RAM |

## Overview

This quick reference demonstrates loading the IMDB movie review dataset from
HuggingFace Hub for sentiment analysis. You'll learn to handle text data
in Datarax pipelines, which differs from image data handling.

## Learning Goals

By the end of this example, you will be able to:

1. Load IMDB dataset using `HFEagerSource` with streaming
2. Handle text data in Datarax pipelines
3. Apply text preprocessing transformations
4. Understand differences between text and image pipelines
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax with HuggingFace support
uv pip install "datarax[data]"
```

**Note**: First run may download dataset files from HuggingFace Hub.
"""

# %%
# Imports
import jax
import jax.numpy as jnp
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import HFEagerConfig, HFEagerSource

print(f"JAX devices: {jax.devices()}")

# %% [markdown]
"""
## IMDB Dataset Overview

The IMDB dataset contains 50,000 movie reviews labeled for sentiment analysis:

| Split | Samples | Labels |
|-------|---------|--------|
| train | 25,000 | 0 (negative), 1 (positive) |
| test | 25,000 | 0 (negative), 1 (positive) |

Each sample contains:
- `text`: The movie review text (string)
- `label`: Sentiment label (0 or 1)
"""

# %%
# Load IMDB in streaming mode
config = HFEagerConfig(
    name="stanfordnlp/imdb",  # Use full dataset path for reliability
    split="train",
    streaming=True,  # Stream to avoid downloading 84MB
    download_kwargs={"trust_remote_code": True},
)

source = HFEagerSource(config, rngs=nnx.Rngs(0))
print(f"Loaded HuggingFace dataset: {config.name}")
print(f"Split: {config.split}")
print("Mode: Streaming (no full download)")

# %% [markdown]
"""
## Step 1: Inspect Data Structure

Unlike image datasets, IMDB returns text strings. Let's examine the structure.
"""

# %%
# For text datasets, iterate element-by-element since strings can't be
# batched as JAX arrays (text needs tokenization first for batching)
print("Sample reviews from IMDB:")

for i, element in enumerate(source):
    if i >= 3:
        break

    print(f"\nExample {i + 1}:")
    print(f"  Keys: {list(element.keys())}")

    # Show label
    label = element.get("label")
    sentiment = "positive" if label == 1 else "negative"
    print(f"  Label: {label} ({sentiment})")

    # Show text preview
    text = element.get("text", "")
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    text_preview = str(text)[:100] + "..." if len(str(text)) > 100 else str(text)
    print(f"  Text preview: {text_preview}")

# Reset source for further use
source.reset()

# Expected output:
# Example 1:
#   Keys: ['text', 'label']
#   Label: 0 (negative)
#   Text preview: I rented I AM CURIOUS-YELLOW from my video store because of...

# %% [markdown]
"""
## Step 2: Text Preprocessing

For NLP tasks, you typically need to:
1. Tokenize text (convert to token IDs)
2. Truncate/pad to fixed length
3. Create attention masks

Here's a simple length-based transform for demonstration.
"""


# %%
def normalize_label(element, key=None):  # noqa: ARG001
    """Normalize sentiment label to JAX array."""
    del key  # Unused - deterministic operator

    # IMDB labels: 0=negative, 1=positive
    # Convert to proper JAX array for batching
    label = element.data.get("label", 0)
    return element.update_data({"label": jnp.array(label, dtype=jnp.int32)})


text_stats_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=normalize_label,
    rngs=nnx.Rngs(0),
)

print("Created label normalization operator")

# %% [markdown]
"""
## Step 3: Build Pipeline with Preprocessing

Chain the source with our preprocessing operator.
"""

# %%
# Create fresh source for the full pipeline
# Note: We exclude 'text' field because strings can't be batched as JAX arrays.
# For text processing, you would typically tokenize first or process element-by-element.
source2 = HFEagerSource(
    HFEagerConfig(
        name="stanfordnlp/imdb",
        split="train",
        streaming=True,
        download_kwargs={"trust_remote_code": True},
        exclude_keys={"text"},  # Exclude text field - can't batch strings
    ),
    rngs=nnx.Rngs(1),
)

# Build pipeline
pipeline = from_source(source2, batch_size=8).add(OperatorNode(text_stats_op))

print("Pipeline: HFEagerSource(IMDB) -> TextStats -> Output")

# %% [markdown]
"""
## Step 4: Process and Analyze

Collect statistics about review lengths and sentiment distribution.
"""

# %%
# Process batches and collect sentiment statistics
print("\nAnalyzing IMDB review sentiment:")

total_reviews = 0
total_positive = 0

num_batches = 20  # Process 20 batches for analysis

for i, batch in enumerate(pipeline):
    if i >= num_batches:
        break

    data = batch.get_data()

    batch_size = len(data["label"]) if hasattr(data["label"], "__len__") else 1
    total_reviews += batch_size

    # Count positives (label=1 is positive)
    labels = data["label"]
    if hasattr(labels, "__iter__"):
        total_positive += sum(1 for l in labels if l == 1)
    else:
        total_positive += 1 if labels == 1 else 0

    if i < 3:  # Show first 3 batches
        print(f"Batch {i}: {batch_size} samples, labels={labels[:5]}...")

print(f"\nSentiment Summary ({total_reviews} reviews analyzed):")
print(f"  Positive: {total_positive} ({100 * total_positive / total_reviews:.1f}%)")
total_negative = total_reviews - total_positive
print(f"  Negative: {total_negative} ({100 * total_negative / total_reviews:.1f}%)")

# Expected output:
# Sentiment Summary (160 reviews analyzed):
#   Positive: ~50%
#   Negative: ~50%

# %% [markdown]
"""
## Results Summary

| Component | Description |
|-----------|-------------|
| **Dataset** | IMDB (25k train reviews) |
| **Format** | Text + binary label |
| **Mode** | Streaming (no full download) |
| **Preprocessing** | Label normalization (text excluded for batching) |

### Text vs Image Pipelines

| Aspect | Image | Text |
|--------|-------|------|
| Data type | Arrays (H×W×C) | Strings (can't batch directly) |
| Batching | Stack arrays | Tokenize first, then batch |
| Normalization | Pixel scaling | Tokenization to IDs |
| Augmentation | Spatial transforms | Synonym replacement, etc. |

**Note:** Text strings cannot be batched as JAX arrays. For NLP tasks:
1. Use `exclude_keys` to skip text fields when batching numeric fields
2. Process text element-by-element, or
3. Tokenize text to numeric IDs before batching

### Integration Notes

For full NLP pipelines, you would typically:

1. Use a tokenizer (HuggingFace tokenizers, SentencePiece)
2. Convert tokens to fixed-length sequences
3. Add attention masks for padding
4. Store as JAX arrays for training
"""

# %% [markdown]
"""
## Next Steps

- **Full tutorial**: [HuggingFace Tutorial](02_hf_tutorial.ipynb) for advanced usage
- **Image datasets**: [CIFAR-10](../../core/04_cifar10_quickref.ipynb)
- **TFDS alternative**: [TFDS](../tfds/01_tfds_quickref.ipynb)
- **API Reference**: [HFEagerSource](https://datarax.readthedocs.io/sources/hf/)
"""


# %%
def main():
    """Run the IMDB quick reference example."""
    print("IMDB Sentiment Analysis Quick Reference")
    print("=" * 50)

    # Load dataset (exclude text field - strings can't be batched)
    config = HFEagerConfig(
        name="stanfordnlp/imdb",
        split="train",
        streaming=True,
        download_kwargs={"trust_remote_code": True},
        exclude_keys={"text"},
    )
    source = HFEagerSource(config, rngs=nnx.Rngs(0))

    # Create pipeline with label normalization
    pipeline = from_source(source, batch_size=8).add(OperatorNode(text_stats_op))

    # Process batches
    total_reviews = 0
    total_positive = 0

    for i, batch in enumerate(pipeline):
        if i >= 10:  # Process 10 batches
            break

        data = batch.get_data()
        batch_size = len(data["label"]) if hasattr(data["label"], "__len__") else 1
        total_reviews += batch_size

        labels = data["label"]
        if hasattr(labels, "__iter__"):
            total_positive += sum(1 for l in labels if l == 1)
        else:
            total_positive += 1 if labels == 1 else 0

    print(f"Processed {total_reviews} IMDB reviews")
    negative = total_reviews - total_positive
    print(f"Sentiment distribution: {total_positive} positive, {negative} negative")
    print("Example completed successfully!")


if __name__ == "__main__":
    main()
