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
# Calibration Eval — Pipeline.scan for inference and metrics

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~1 min |
| **Prerequisites** | Pipeline Quickstart |
| **Format** | Python + Jupyter |

## Overview

Demonstrates that `Pipeline.scan` works equally well for inference and
calibration sweeps — no gradient updates, metrics accumulate via
`init_carry`, and the per-batch outputs are returned alongside the
final aggregate.

## Setup

```bash
uv pip install datarax
```

Activate the project virtualenv:

```bash
source activate.sh
```

## Learning Goals

By the end of this example, you will be able to:

1. Use `Pipeline.scan` for evaluation only (no gradient updates).
2. Accumulate metrics through the `carry` argument.
3. Return both per-batch outputs and final aggregate metrics.
"""

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# %% [markdown]
"""
## 1. Model — pretend this comes from a checkpoint
"""


# %%
class _Classifier(nnx.Module):
    def __init__(self, *, num_classes: int, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(3, 8, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.head = nnx.Linear(8 * 32 * 32, num_classes, rngs=rngs)

    def __call__(self, image: jax.Array) -> jax.Array:
        h = nnx.relu(self.conv(image))
        return self.head(h.reshape(h.shape[0], -1))


# %% [markdown]
"""
## 2. Step function — eval only, no gradients, accumulates metrics in carry
"""


# %%
def eval_step(carry, model, batch):
    """Per-batch eval: compute logits, accuracy, top-1 confidence.

    ``carry`` is ``(total_correct, total_count, total_max_softmax)``.
    Per-step output is the batch-mean confidence so callers can plot
    confidence over batches.
    """
    total_correct, total_count, total_conf = carry

    logits = model(batch["image"])
    probs = nnx.softmax(logits, axis=-1)
    pred = jnp.argmax(logits, axis=-1)

    correct = jnp.sum((pred == batch["label"]).astype(jnp.int32))
    confidence = jnp.mean(jnp.max(probs, axis=-1))

    new_carry = (
        total_correct + correct,
        total_count + jnp.int32(batch["label"].shape[0]),
        total_conf + confidence,
    )
    return new_carry, confidence


# %% [markdown]
"""
## 3. Driver
"""


# %%
def main() -> None:
    """Run main."""
    print("=" * 70)
    print("Calibration eval: forward-only sweep with metric accumulation")
    print("=" * 70)

    rng = np.random.default_rng(0)
    images = rng.uniform(0, 1, size=(256, 32, 32, 3)).astype(np.float32)
    labels = rng.integers(0, 10, size=(256,)).astype(np.int32)
    data = {"image": jnp.asarray(images), "label": jnp.asarray(labels)}

    pipeline = Pipeline(
        source=MemorySource(MemorySourceConfig(shuffle=False), data),
        stages=[],
        batch_size=32,
        rngs=nnx.Rngs(0),
    )

    model = _Classifier(num_classes=10, rngs=nnx.Rngs(42))

    init_carry = (jnp.int32(0), jnp.int32(0), jnp.float32(0.0))
    final_carry, per_batch_conf = pipeline.scan(
        eval_step,
        modules=(model,),
        length=8,
        init_carry=init_carry,
    )

    total_correct, total_count, total_conf = final_carry
    n_steps = per_batch_conf.shape[0]
    accuracy = float(total_correct) / float(total_count)
    avg_confidence = float(total_conf) / float(n_steps)

    print(f"  Total samples:    {int(total_count)}")
    print(f"  Total correct:    {int(total_correct)}")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  Mean confidence:  {avg_confidence:.4f}")
    print(f"  Per-batch confidence trace: {[round(float(c), 4) for c in per_batch_conf]}")


if __name__ == "__main__":
    main()
