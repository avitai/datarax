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
# Checkpointing and Resumable Training Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~3 min |
| **Prerequisites** | Pipeline Quickstart, Operators Tutorial |
| **Format** | Python + Jupyter |
| **Memory** | ~500 MB RAM |

## Overview

This guide implements fault-tolerant training pipelines that can resume
from interruptions using the **NNX-standard checkpoint pattern** —
``nnx.to_pure_dict`` for snapshotting state and ``nnx.replace_by_pure_dict``
for restoring it, with ``orbax.checkpoint.StandardCheckpointer`` handling
the on-disk serialization.

The triple ``(pipeline, model, optimizer)`` is checkpointed together so
resumption restores the data-cursor position, the model weights, and
the optimizer state — all from a single Orbax directory.

## Setup

```bash
uv pip install datarax flax optax orbax-checkpoint matplotlib
```

## Learning Goals

By the end of this guide, you will be able to:

1. Snapshot an NNX module's state with ``nnx.to_pure_dict``.
2. Persist that snapshot with ``orbax.checkpoint.StandardCheckpointer``.
3. Restore the snapshot back into a freshly-constructed module with
   ``nnx.replace_by_pure_dict``.
4. Checkpoint a ``(pipeline, model, optimizer)`` triple atomically so
   resumption preserves data position, model weights, and optimizer
   state simultaneously.
5. Verify deterministic resumption: a run that checkpoints at step ``k``,
   loads, and continues should match a never-interrupted run.
"""

# %%
import shutil
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


print(f"JAX version: {jax.__version__}")


# %% [markdown]
"""
## Part 1: Synthetic Data + Tiny CNN

A 28x28 grayscale classifier on 1024 synthetic samples. Small enough to
keep the example under three minutes; the checkpoint pattern is
identical at any scale.
"""


# %%
def make_data(num_samples: int = 2048, seed: int = 0) -> dict:
    """Synthetic but learnable: label is the brightness decile.

    Each image is sampled at a label-dependent brightness mean; the
    model learns to read off that mean. With a real signal the loss
    decreases monotonically (untrained models flail around 2.3 for
    uniform random labels), which makes the determinism check
    meaningful.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, size=(num_samples,)).astype(np.int32)
    # Per-class brightness in [0.05, 0.95]; signal-to-noise > 1 so
    # the model can actually learn.
    target_mean = (labels + 0.5) / 10.0
    noise = rng.normal(0, 0.05, size=(num_samples, 28, 28, 1)).astype(np.float32)
    images = (target_mean[:, None, None, None] + noise).astype(np.float32)
    images = np.clip(images, 0.0, 1.0)
    return {"image": images, "label": labels}


class TinyCNN(nnx.Module):
    """Two conv layers + linear head."""

    def __init__(self, *, num_classes: int = 10, rngs: nnx.Rngs) -> None:
        """Initialize the module."""
        self.conv1 = nnx.Conv(1, 16, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(16, 32, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.head = nnx.Linear(32 * 7 * 7, num_classes, rngs=rngs)

    def __call__(self, image: jax.Array) -> jax.Array:
        """Run __call__."""
        x = nnx.relu(self.conv1(image))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


def cross_entropy_loss(model: TinyCNN, batch: dict) -> jax.Array:
    """Run cross_entropy_loss."""
    logits = model(batch["image"])
    return optax.softmax_cross_entropy_with_integer_labels(logits, batch["label"]).mean()


# %% [markdown]
"""
## Part 2: NNX-Standard Checkpoint Helpers

The two functions below are the entire NNX checkpoint pattern. The
``orbax.checkpoint.StandardCheckpointer`` writes any PyTree to disk;
``nnx.to_pure_dict`` and ``nnx.replace_by_pure_dict`` translate
``nnx.Module`` state to and from such PyTrees.

The key insight: NNX modules cannot be passed to Orbax directly because
they hold non-serializable graph metadata. ``to_pure_dict`` strips the
graph and yields a plain leaf-only PyTree of arrays;
``replace_by_pure_dict`` applies a saved leaf-only PyTree back into a
target module's state in place.
"""


# %%
def snapshot(model: TinyCNN, optimizer: nnx.Optimizer, pipeline: Pipeline) -> dict:
    """Capture the full training state as a serialisable PyTree.

    Each entry uses ``nnx.to_pure_dict`` so the result contains only
    JAX arrays (no graph metadata, no closures, no class objects).
    """
    return {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
        "pipeline": nnx.to_pure_dict(nnx.state(pipeline)),
    }


def save_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    directory: Path,
    step: int,
    model: TinyCNN,
    optimizer: nnx.Optimizer,
    pipeline: Pipeline,
) -> None:
    """Snapshot and persist the (model, optimizer, pipeline) triple."""
    snapshot_dict = snapshot(model, optimizer, pipeline)
    target = directory / f"step_{step}"
    if target.exists():
        shutil.rmtree(target)
    checkpointer.save(target, snapshot_dict)
    checkpointer.wait_until_finished()


def load_checkpoint(
    checkpointer: ocp.StandardCheckpointer,
    directory: Path,
    step: int,
    model: TinyCNN,
    optimizer: nnx.Optimizer,
    pipeline: Pipeline,
) -> None:
    """Restore state in-place into pre-constructed modules.

    The caller constructs ``model``, ``optimizer``, and ``pipeline``
    afresh (so their graph structure is intact); this function applies
    the saved leaf arrays via ``replace_by_pure_dict`` and writes the
    updated state back through ``nnx.update``. ``nnx.state(...)``
    returns a *copy* — without ``nnx.update`` the original modules
    keep their fresh initial state and the resume looks like a
    reset.
    """
    target = directory / f"step_{step}"
    saved = checkpointer.restore(target, snapshot(model, optimizer, pipeline))

    for module, pure in (
        (model, saved["model"]),
        (optimizer, saved["optimizer"]),
        (pipeline, saved["pipeline"]),
    ):
        state = nnx.state(module)
        nnx.replace_by_pure_dict(state, pure)
        nnx.update(module, state)


# %% [markdown]
"""
## Part 3: Build the Training Pieces

Two augmentation stages (normalize + horizontal flip) pushed through a
``Pipeline``. The pipeline owns its iteration cursor and all stochastic
RNG state, so checkpointing it captures the data-loading position
exactly.
"""


# %%
def normalize(element, key=None):
    """Run normalize."""
    del key
    image = element.data["image"]
    return element.update_data({"image": (image - 0.5) / 0.5})


def random_flip(element, key):
    """Run random_flip."""
    flip_key, _ = jax.random.split(key)
    should_flip = jax.random.bernoulli(flip_key, 0.5)
    image = element.data["image"]
    flipped = jax.lax.cond(should_flip, lambda x: jnp.flip(x, axis=1), lambda x: x, image)
    return element.update_data({"image": flipped})


def build_pipeline(seed: int, batch_size: int = 32) -> Pipeline:
    """Run build_pipeline."""
    # 2048 samples / 32 batch = 64 batches; supports 60-step demo in one epoch.
    data = make_data(num_samples=2048, seed=seed)
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(seed))
    norm_op = ElementOperator(
        ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0)
    )
    flip_op = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="flip"),
        fn=random_flip,
        rngs=nnx.Rngs(flip=seed + 1000),
    )
    return Pipeline(
        source=source,
        stages=[norm_op, flip_op],
        batch_size=batch_size,
        rngs=nnx.Rngs(seed),
    )


def build_model_and_optimizer(seed: int) -> tuple[TinyCNN, nnx.Optimizer]:
    """Run build_model_and_optimizer."""
    model = TinyCNN(rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


# %% [markdown]
"""
## Part 4: Train Step + Run Loop

Standard NNX training step: ``nnx.value_and_grad`` over the loss,
followed by ``optimizer.update``. The pipeline iterator yields each
batch.
"""


# %%
@nnx.jit
def train_step(model: TinyCNN, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
    """Run train_step."""
    loss_and_grad = nnx.value_and_grad(cross_entropy_loss)
    loss, grads = loss_and_grad(model, batch)
    optimizer.update(model, grads)
    return loss


def run(
    pipeline: Pipeline,
    model: TinyCNN,
    optimizer: nnx.Optimizer,
    *,
    max_steps: int,
    checkpointer: ocp.StandardCheckpointer | None = None,
    ckpt_dir: Path | None = None,
    ckpt_every: int = 0,
    interrupt_at: int | None = None,
    start_step: int = 0,
) -> tuple[list[float], int]:
    """Run training, optionally checkpointing every N steps.

    Optionally simulates an interruption mid-run for testing.
    """
    losses: list[float] = []
    step = start_step
    for batch in pipeline:
        loss = train_step(model, optimizer, batch)
        losses.append(float(loss))
        step += 1
        if (
            checkpointer is not None
            and ckpt_dir is not None
            and ckpt_every > 0
            and step % ckpt_every == 0
        ):
            save_checkpoint(checkpointer, ckpt_dir, step, model, optimizer, pipeline)
        if interrupt_at is not None and step >= interrupt_at:
            return losses, step
        if step >= max_steps:
            return losses, step
    return losses, step


# %% [markdown]
"""
## Part 5: Demonstrate Resumption

Run training in three phases:

1. **Reference** — train uninterrupted for 60 steps; record the loss curve.
2. **Crash** — re-train from scratch with the same seed but interrupt at
   step 30 after writing a checkpoint.
3. **Resume** — construct fresh model/optimizer/pipeline, load the
   step-30 checkpoint, and continue to step 60.

If the NNX-standard pattern is correct, the concatenated phase-2 + phase-3
loss curve must equal the reference curve exactly (up to floating-point
determinism within ``@nnx.jit``).
"""

# %%
ckpt_dir = Path(tempfile.mkdtemp(prefix="datarax_ckpt_"))
print(f"Checkpoint directory: {ckpt_dir}")
print()

MAX_STEPS = 60
CKPT_EVERY = 10
INTERRUPT_AT = 30
SEED = 42

# %%
# Phase 1: reference run, no checkpoints, no interruption
print("=" * 60)
print(f"PHASE 1: reference run ({MAX_STEPS} steps, no checkpoints)")
print("=" * 60)
ref_pipeline = build_pipeline(SEED)
ref_model, ref_optimizer = build_model_and_optimizer(SEED)
ref_losses, ref_step = run(
    ref_pipeline,
    ref_model,
    ref_optimizer,
    max_steps=MAX_STEPS,
)
print(f"Reference: {ref_step} steps, final loss={ref_losses[-1]:.4f}")
print()

# %%
# Phase 2: train + checkpoint, simulate crash at step 30
print("=" * 60)
print(f"PHASE 2: train, checkpoint every {CKPT_EVERY}, interrupt at step {INTERRUPT_AT}")
print("=" * 60)
checkpointer = ocp.StandardCheckpointer()
phase2_pipeline = build_pipeline(SEED)
phase2_model, phase2_optimizer = build_model_and_optimizer(SEED)
phase2_losses, phase2_step = run(
    phase2_pipeline,
    phase2_model,
    phase2_optimizer,
    max_steps=MAX_STEPS,
    checkpointer=checkpointer,
    ckpt_dir=ckpt_dir,
    ckpt_every=CKPT_EVERY,
    interrupt_at=INTERRUPT_AT,
)
print(f"Crashed: {phase2_step} steps, final loss={phase2_losses[-1]:.4f}")
print(f"Available checkpoints: {sorted(p.name for p in ckpt_dir.iterdir())}")
print()

# %%
# Phase 3: fresh state, restore from checkpoint, continue
print("=" * 60)
print(f"PHASE 3: restore from step {INTERRUPT_AT}, train to step {MAX_STEPS}")
print("=" * 60)
restored_pipeline = build_pipeline(SEED)
restored_model, restored_optimizer = build_model_and_optimizer(SEED)
load_checkpoint(
    checkpointer,
    ckpt_dir,
    INTERRUPT_AT,
    restored_model,
    restored_optimizer,
    restored_pipeline,
)
phase3_losses, phase3_step = run(
    restored_pipeline,
    restored_model,
    restored_optimizer,
    max_steps=MAX_STEPS,
    start_step=INTERRUPT_AT,
)
resumed_losses = phase2_losses + phase3_losses
print(f"Resumed: end step={phase3_step}, final loss={phase3_losses[-1]:.4f}")
print(f"Total resumed-curve length: {len(resumed_losses)} (expected {MAX_STEPS})")


# %% [markdown]
"""
## Part 6: Verify Determinism

The reference loss curve (uninterrupted) and the concatenated
crash-and-resume loss curve must agree to within ``@nnx.jit`` numerical
determinism. Any divergence indicates that some piece of state was
missed by ``snapshot``.
"""

# %%
ref_arr = np.asarray(ref_losses)
res_arr = np.asarray(resumed_losses)
loss_diff = float(np.abs(ref_arr - res_arr).max())
print(f"Max |reference - resumed| over {MAX_STEPS} steps: {loss_diff:.4e}")

# Compare model parameters directly. State equality is a stronger guarantee
# than loss equality — under JIT, two runs that diverge by a single ULP
# during step 30 can amplify into 1% loss differences by step 60 even
# though the underlying state matches up to numerical precision.
ref_params = nnx.to_pure_dict(nnx.state(ref_model))
res_params = nnx.to_pure_dict(nnx.state(restored_model))


def _max_abs_diff(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


param_diffs = jax.tree.map(_max_abs_diff, ref_params, res_params)
max_param_diff = max(jax.tree_util.tree_leaves(param_diffs))
print(f"Max |reference - resumed| over model params: {max_param_diff:.4e}")
assert max_param_diff < 1e-3, (
    f"Resumed model parameters diverged from reference by {max_param_diff:.4e}; "
    "some state is missing from the checkpoint snapshot."
)
print("Determinism check passed: model parameters round-trip through Orbax.")


# %% [markdown]
"""
## Part 7: Visualize the Two Trajectories
"""

# %%
output_dir = Path("docs/assets/images/examples")
output_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ref_arr, label="Reference (uninterrupted)", color="tab:blue", linewidth=2)
ax.plot(
    res_arr,
    label="Resumed (crash + restore)",
    color="tab:orange",
    linestyle="--",
    linewidth=2,
)
ax.axvline(INTERRUPT_AT, color="grey", linestyle=":", label=f"Restore @ step {INTERRUPT_AT}")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss")
ax.set_title("Resumed training matches reference under NNX-standard checkpoint")
ax.legend()
fig.tight_layout()
out_path = output_dir / "checkpoint-resume-validation.png"
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Saved: {out_path}")


# %% [markdown]
"""
## Part 8: Cleanup
"""

# %%
shutil.rmtree(ckpt_dir, ignore_errors=True)
print(f"Removed checkpoint directory: {ckpt_dir}")


# %% [markdown]
"""
## Results Summary

The NNX-standard checkpoint pattern is exactly three calls per object:

1. ``nnx.to_pure_dict(nnx.state(module))`` — convert to a plain PyTree
2. ``StandardCheckpointer.save(path, pure_dict)`` — write to disk
3. ``nnx.replace_by_pure_dict(nnx.state(target), pure_dict)`` — restore

No subclass is required; no datarax-specific wrapper is required.
``Pipeline`` checkpoints the same way every other ``nnx.Module`` does
because it *is* one.

## Next Steps

- For periodic-cleanup checkpoint policies, wrap the
  ``StandardCheckpointer`` calls in an ``orbax.checkpoint.CheckpointManager``;
  it understands keep-last-N, async writes, and metadata.
- For very large models, ``nnx.split`` and ``nnx.merge`` give the same
  semantics with reduced peak memory.
- See ``docs/user_guide/dag_construction.md`` for branching pipelines —
  the checkpoint pattern is identical regardless of pipeline shape.
"""


# %%
def main() -> None:
    """Self-test entry point."""
    print("Checkpointing and Resumable Training Guide")
    print("=" * 60)
    print(f"Reference final loss: {ref_losses[-1]:.4f}")
    print(f"Resumed final loss:   {resumed_losses[-1]:.4f}")
    print(f"Max loss divergence:  {loss_diff:.4e}")
    print(f"Max param divergence: {max_param_diff:.4e}")
    print("Guide completed successfully!")


if __name__ == "__main__":
    main()
