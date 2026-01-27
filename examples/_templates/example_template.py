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
# Title: [Descriptive Example Name]

| Metadata | Value |
|----------|-------|
| **Level** | Beginner / Intermediate / Advanced |
| **Runtime** | ~X min (CPU) / ~Y min (GPU) |
| **Prerequisites** | [List required knowledge] |
| **Format** | Python + Jupyter |

## Overview

[Brief 1-2 paragraph description of what this example demonstrates and why it's useful.]

## Learning Goals

By the end of this example, you will be able to:

1. [First specific, measurable objective - use action verbs: Create, Train, Implement, Understand]
2. [Second objective]
3. [Third objective]
4. [Optional fourth objective]

"""

# %% [markdown]
"""
## Setup & Prerequisites

### Required Knowledge
- [Prerequisite 1](link) - Brief description
- [Prerequisite 2](link) - Brief description

### Installation

```bash
# Install datarax with data dependencies
uv pip install "datarax[data]"
```

**Estimated Time:** X-Y minutes
"""

# %%
# Imports
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

# %% [markdown]
"""
## Core Concepts

[For Tier 2 (Tutorials) and Tier 3 (Advanced) only - explain the theory/concepts.
For Tier 1 (Quick Reference), skip this section or keep minimal.]

### Key Concept 1

[Explanation with optional LaTeX: $x = \\frac{a}{b}$]

### Key Concept 2

[Explanation with intuition and examples]
"""

# %% [markdown]
"""
## Implementation

### Step 1: [First Step Title]

[Brief explanation of what we're doing and why]
"""


# %%
# Step 1: Create sample data
def create_sample_data(num_samples: int = 100) -> dict:
    """Create sample data for the example.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        Dictionary with 'image' and 'label' keys.
    """
    # Use numpy for initial data creation
    image = np.random.randint(0, 255, (num_samples, 28, 28, 1)).astype(np.float32)
    label = np.random.randint(0, 10, (num_samples,)).astype(np.int32)
    return {"image": image, "label": label}


data = create_sample_data()
print(f"Data shape: image={data['image'].shape}, label={data['label'].shape}")
# Expected output:
# Data shape: image=(100, 28, 28, 1), label=(100,)

# %% [markdown]
"""
### Step 2: [Second Step Title]

[Brief explanation]
"""

# %%
# Step 2: Create data source
source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))
print(f"Source length: {len(source)}")
# Expected output:
# Source length: 100

# %% [markdown]
"""
### Step 3: [Third Step Title]

[Brief explanation]
"""

# %%
# Step 3: Create pipeline
pipeline = from_source(source, batch_size=32)

# Iterate and process
for i, batch in enumerate(pipeline):
    if i >= 2:  # Only show first 2 batches
        break
    print(f"Batch {i}: image={batch['image'].shape}, label={batch['label'].shape}")

# Expected output:
# Batch 0: image=(32, 28, 28, 1), label=(32,)
# Batch 1: image=(32, 28, 28, 1), label=(32,)

# %% [markdown]
"""
## Results & Evaluation

### What We Achieved

[Summary of what was demonstrated]

### Key Metrics

| Metric | Value |
|--------|-------|
| [Metric 1] | [Value] |
| [Metric 2] | [Value] |

### Interpretation

[What the results mean, realistic expectations, comparison to baselines if relevant]
"""

# %%
# Optional: Visualization or final metrics
print("Example completed successfully!")

# %% [markdown]
"""
## Next Steps & Resources

### Try These Experiments

1. [Specific, achievable experiment 1]
2. [Experiment 2]
3. [Experiment 3]

### Related Examples

- [Example Name](../path/to/example.py) - Brief description
- [Another Example](../path/to/another.py) - Brief description

### API Reference

- [MemorySource](../../docs/sources/memory_source.md)
- [ElementOperator](../../docs/operators/element_operator.md)

### Further Reading

- [External resource 1](url)
- [External resource 2](url)
"""


# %%
def main():
    """Main entry point for command-line execution."""
    print("Running example...")

    # Re-run the example steps
    data = create_sample_data()
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))
    pipeline = from_source(source, batch_size=32)

    total_samples = 0
    for batch in pipeline:
        total_samples += batch["image"].shape[0]

    print(f"Processed {total_samples} samples")
    print("Done!")


if __name__ == "__main__":
    main()
