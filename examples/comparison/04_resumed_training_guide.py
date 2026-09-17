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
# Grain and Datarax: Resumed Training Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~1 min (CPU) |
| **Prerequisites** | [Iteration and Checkpoint State](01_grain_datarax_quickref.py), [Process and Device Sharding](03_sharding_guide.py) |
| **Format** | Python + Jupyter |

## Overview

A training run that can be interrupted needs a checkpoint that holds three things: the
model, the optimizer, and the position of the data loader, including the randomness it was
about to use. Restore all three and the run continues as if it had never stopped; restore
two and the model trains on records it has already seen, or on different noise.

This guide trains the same model from a Grain loader and from a Datarax pipeline, saves
the three parts at the same step through one `substrax.checkpoint.OrbaxCheckpointStore`,
restores them into fresh objects, and shows that both resumed runs reproduce the
uninterrupted run loss for loss. The loss is `calibrax.metrics.functional.mse`. The two
libraries differ only in what the loader contributes to the checkpoint and how it is put
back.

## Learning Goals

By the end of this guide, you will be able to:

1. Save model, optimizer and loader state together with `OrbaxCheckpointStore`, and read
   back the step and loss it records
2. Restore a Grain loader from its JSON state and a Datarax pipeline from its iterator state
3. Show that a resumed run reproduces the uninterrupted run exactly, in both libraries
4. Say what each loader's state holds and why the randomness resumes with it
"""

# %% [markdown]
"""
## Setup

Grain, `substrax` and `calibrax` are installed as Datarax dependencies. The records are 64
rows of three features with a target, and a stochastic operator adds noise to each record,
so a resumed run only matches the uninterrupted one if the loader's randomness resumes too.

```bash
uv pip install datarax
```
"""

# %%
import json
import shutil
import tempfile
from pathlib import Path

import grain
import jax
import numpy as np
import optax
from calibrax.metrics.functional import mse
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline, PipelineIterator
from datarax.sources import MemorySource, MemorySourceConfig


NUM_RECORDS = 64
BATCH_SIZE = 8
STEPS_PER_EPOCH = NUM_RECORDS // BATCH_SIZE
NUM_EPOCHS = 5
TOTAL_STEPS = NUM_EPOCHS * STEPS_PER_EPOCH
CHECKPOINT_STEP = 20
NOISE_SCALE = 0.1
LEARNING_RATE = 0.1
TRUE_WEIGHTS = np.array([[2.0], [-1.0], [0.5]], dtype=np.float32)

features = np.random.default_rng(0).normal(size=(NUM_RECORDS, 3)).astype(np.float32)
targets = features @ TRUE_WEIGHTS
checkpoint_root = Path(tempfile.mkdtemp(prefix="comparison_checkpoints_"))
print(
    f"x={features.shape}, y={targets.shape}; {TOTAL_STEPS} steps, checkpoint at {CHECKPOINT_STEP}"
)
# Expected output:
# x=(64, 3), y=(64, 1); 40 steps, checkpoint at 20

# %% [markdown]
"""
## Core Concepts

### What a checkpoint holds

| Part | Grain | Datarax |
|---|---|---|
| Model and optimizer | `nnx.to_pure_dict(nnx.state(...))`: arrays only | The same |
| Loader state | `iterator.get_state()`: JSON bytes with the last index each worker served, the sampler and the source description | `iterator.get_state()`: `position`, `epoch`, one count per random stream, `version` |
| Randomness | Resumes because the draw index resumes | Resumes because the epoch and record index resume |
| Put back with | `iterator.set_state(bytes)` on a loader built the same way | `iterator.set_state(dict)` on a fresh iterator of a pipeline built the same way |

`OrbaxCheckpointStore.save(payload, step, loss)` writes any pytree of arrays and plain
leaves under a step number, with the step, a timestamp and the loss in a JSON sidecar;
`restore(template, step)` fills a template of the same structure. Grain's bytes travel as a
string leaf; Datarax's state is a dict of ints and lists.

### The same training step

The model is a linear regression, the optimizer plain SGD, and the loss
`calibrax.metrics.functional.mse`. One `nnx.jit` step serves both loaders, which yield
`{"x", "y"}` batches: NumPy arrays from Grain, JAX arrays from Datarax.
"""


# %%
class LinearRegression(nnx.Module):
    """One linear layer from three features to one target."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Initialize the layer."""
        self.linear = nnx.Linear(3, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Predict the target."""
        return self.linear(x)


def build_model() -> tuple[LinearRegression, nnx.Optimizer]:
    """A fresh model and optimizer from the same seed every time."""
    model = LinearRegression(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=LEARNING_RATE), wrt=nnx.Param)
    return model, optimizer


@nnx.jit
def train_step(model: LinearRegression, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
    """One SGD step on the batch, returning its loss."""

    def loss_fn(module: LinearRegression) -> jax.Array:
        return mse(module(batch["x"]), batch["y"])

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def training_state(model: LinearRegression, optimizer: nnx.Optimizer) -> dict:
    """The model and optimizer as pytrees of arrays, which a checkpoint store writes."""
    return {
        "model": nnx.to_pure_dict(nnx.state(model)),
        "optimizer": nnx.to_pure_dict(nnx.state(optimizer)),
    }


def load_training_state(model: LinearRegression, optimizer: nnx.Optimizer, saved: dict) -> None:
    """Write the saved arrays back into the live model and optimizer."""
    for module, pure in ((model, saved["model"]), (optimizer, saved["optimizer"])):
        state = nnx.state(module)
        nnx.replace_by_pure_dict(state, pure)
        nnx.update(module, state)


# %% [markdown]
"""
## Part 1: The Uninterrupted Runs

Each library trains for five epochs with per-record noise, and its loss curve is the
reference the resumed run must reproduce. Grain's sampler runs `num_epochs=5` on one
iterator; a Datarax pipeline serves one epoch per iterator, and `reset()` starts the next.
"""


# %%
class Records(grain.sources.RandomAccessDataSource):
    """``{"x", "y"}`` records read by index."""

    def __len__(self) -> int:
        """Return the number of records."""
        return NUM_RECORDS

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        """Return the record at ``i``."""
        return {"x": features[i], "y": targets[i]}

    def __repr__(self) -> str:
        """Describe the records, which Grain compares when it restores a checkpoint."""
        return f"Records(num_records={NUM_RECORDS})"


class AddNoise(grain.transforms.RandomMap):
    """Add Gaussian noise drawn from the generator Grain passes with each record."""

    def random_map(self, element: dict, rng: np.random.Generator) -> dict:
        """Return the record with noise added to ``x``."""
        noise = rng.normal(scale=NOISE_SCALE, size=element["x"].shape).astype(np.float32)
        return {**element, "x": element["x"] + noise}


def build_grain_iterator():
    """A shuffled, noisy, batched loader over five epochs."""
    sampler = grain.samplers.IndexSampler(
        num_records=NUM_RECORDS, shuffle=True, num_epochs=NUM_EPOCHS, seed=0
    )
    loader = grain.DataLoader(
        data_source=Records(),
        sampler=sampler,
        operations=[AddNoise(), grain.transforms.Batch(batch_size=BATCH_SIZE)],
    )
    return iter(loader)


def add_noise(element, key):
    """Add Gaussian noise drawn from this record's own key."""
    x = element.data["x"]
    return element.update_data({"x": x + NOISE_SCALE * jax.random.normal(key, x.shape)})


def build_datarax_pipeline() -> Pipeline:
    """A shuffled, noisy, batched pipeline over the records."""
    source = MemorySource(
        MemorySourceConfig(shuffle=True), data={"x": features, "y": targets}, rngs=nnx.Rngs(0)
    )
    noise = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="noise"),
        fn=add_noise,
        rngs=nnx.Rngs(noise=0),
    )
    return Pipeline(source=source, stages=[noise], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0))


def datarax_iterators(pipeline: Pipeline, first: PipelineIterator | None = None):
    """Yield one iterator per epoch, starting from ``first`` when a run resumes."""
    iterator = first if first is not None else iter(pipeline)
    while True:
        if not isinstance(iterator, PipelineIterator):
            raise TypeError("a MemorySource pipeline iterates through a PipelineIterator")
        yield iterator
        if int(iterator.get_state()["epoch"]) + 1 >= NUM_EPOCHS:
            return
        pipeline.reset()
        iterator = iter(pipeline)


def train_grain(
    iterator,
    model: LinearRegression,
    optimizer: nnx.Optimizer,
    steps: int,
    save: tuple[int, OrbaxCheckpointStore] | None = None,
) -> tuple[list[float], dict | None]:
    """Train over ``steps`` batches; ``save`` names the step to checkpoint at and the store.

    Returns the losses and the payload saved, or ``None`` when nothing was saved.
    """
    losses: list[float] = []
    payload = None
    for _ in range(steps):
        batch = next(iterator)
        losses.append(float(train_step(model, optimizer, {"x": batch["x"], "y": batch["y"]})))
        if save is not None and len(losses) == save[0]:
            payload = {**training_state(model, optimizer), "loader": iterator.get_state().decode()}
            save[1].save(payload, step=save[0], loss=losses[-1])
    return losses, payload


def train_datarax(
    pipeline: Pipeline,
    model: LinearRegression,
    optimizer: nnx.Optimizer,
    steps: int,
    first: PipelineIterator | None = None,
    save: tuple[int, OrbaxCheckpointStore] | None = None,
) -> tuple[list[float], dict | None]:
    """Train over ``steps`` batches; ``save`` names the step to checkpoint at and the store.

    Returns the losses and the payload saved, or ``None`` when nothing was saved.
    """
    losses: list[float] = []
    payload = None
    for iterator in datarax_iterators(pipeline, first):
        for batch in iterator:
            losses.append(float(train_step(model, optimizer, batch)))
            if save is not None and len(losses) == save[0]:
                payload = {**training_state(model, optimizer), "loader": iterator.get_state()}
                save[1].save(payload, step=save[0], loss=losses[-1])
            if len(losses) == steps:
                return losses, payload
    return losses, payload


def saved_payload(payload: dict | None) -> dict:
    """The payload a training run saved, which a run asked to checkpoint always has."""
    if payload is None:
        raise RuntimeError("the run did not reach its checkpoint step")
    return payload


def restored_payload(store: OrbaxCheckpointStore, template: dict, step: int) -> tuple[dict, dict]:
    """Restore ``template``'s structure from ``store`` at ``step``, with the step's metadata."""
    restored, metadata = store.restore(template, step=step)
    if not isinstance(restored, dict):
        raise TypeError(f"expected the saved payload dict, got {type(restored).__name__}")
    return restored, metadata


grain_model, grain_optimizer = build_model()
grain_reference, _ = train_grain(build_grain_iterator(), grain_model, grain_optimizer, TOTAL_STEPS)
datarax_model, datarax_optimizer = build_model()
datarax_reference, _ = train_datarax(
    build_datarax_pipeline(), datarax_model, datarax_optimizer, TOTAL_STEPS
)
print(
    f"Grain reference:   {len(grain_reference)} steps, loss first {grain_reference[0]:.4f}, last {grain_reference[-1]:.4f}"
)
print(
    f"Datarax reference: {len(datarax_reference)} steps, loss first {datarax_reference[0]:.4f}, last {datarax_reference[-1]:.4f}"
)
print(f"Grain weights:   {np.round(np.asarray(grain_model.linear.kernel[...]).ravel(), 2)}")
print(
    f"Datarax weights: {np.round(np.asarray(datarax_model.linear.kernel[...]).ravel(), 2)} (true {TRUE_WEIGHTS.ravel()})"
)
# Expected output:
# Grain reference:   40 steps, loss first 4.2264, last 0.0126
# Datarax reference: 40 steps, loss first 1.9849, last 0.0402
# Grain weights:   [ 1.95 -1.    0.48]
# Datarax weights: [ 1.99 -0.99  0.51] (true [ 2.  -1.   0.5])

# %% [markdown]
"""
## Part 2: Train, Checkpoint at Step 20, Stop

Each run starts again from the same seeds, trains 20 steps, and saves model, optimizer and
loader state under step 20 in its own store. Step 20 is halfway through the third epoch,
so the loader state carries a mid-epoch position: Grain's bytes name the last index each
worker served, Datarax's dict names position 32 of epoch 2.
"""

# %%
grain_store = OrbaxCheckpointStore(checkpoint_root / "grain")
model, optimizer = build_model()
grain_before, grain_payload = train_grain(
    build_grain_iterator(), model, optimizer, CHECKPOINT_STEP, save=(CHECKPOINT_STEP, grain_store)
)
datarax_store = OrbaxCheckpointStore(checkpoint_root / "datarax")
model, optimizer = build_model()
datarax_before, datarax_payload = train_datarax(
    build_datarax_pipeline(),
    model,
    optimizer,
    CHECKPOINT_STEP,
    save=(CHECKPOINT_STEP, datarax_store),
)
print(f"Latest step in each store: {grain_store.latest_step()}, {datarax_store.latest_step()}")
print(f"Grain loader state keys: {sorted(json.loads(saved_payload(grain_payload)['loader']))}")
print(f"Datarax loader state: {saved_payload(datarax_payload)['loader']}")
# Expected output:
# Latest step in each store: 20, 20
# Grain loader state keys: ['data_source', 'last_seen_indices', 'last_worker_index', 'sampler', 'version', 'worker_count']
# Datarax loader state: {'position': 32, 'epoch': 2, 'rng_counts': [0, 1, 0], 'version': 2, 'fingerprint': {'batch_size': 8, 'length': 64, 'drop_last': False, 'num_epochs': 1, 'shuffled': True}}  # noqa: E501

# %% [markdown]
"""
## Part 3: Restore into Fresh Objects and Continue

A new model and optimizer are built from a different seed, so nothing but the checkpoint
can make them match. `restore(template, step)` fills a template of the same structure and
returns the metadata the store recorded with the step. The model and optimizer arrays go
back through `nnx.replace_by_pure_dict`, Grain's bytes go to `set_state` on a loader built
the same way, and Datarax's dict goes to `set_state` on a fresh iterator of a pipeline
built the same way. Each run then trains the remaining 20 steps.
"""


# %%
def build_fresh_model() -> tuple[LinearRegression, nnx.Optimizer]:
    """A model and optimizer from another seed, so only the checkpoint can align them."""
    model = LinearRegression(rngs=nnx.Rngs(1))
    optimizer = nnx.Optimizer(model, optax.sgd(learning_rate=LEARNING_RATE), wrt=nnx.Param)
    return model, optimizer


model, optimizer = build_fresh_model()
template = {**training_state(model, optimizer), "loader": ""}
restored, grain_metadata = restored_payload(grain_store, template, CHECKPOINT_STEP)
load_training_state(model, optimizer, restored)
grain_iterator = build_grain_iterator()
grain_iterator.set_state(restored["loader"].encode())
grain_after, _ = train_grain(grain_iterator, model, optimizer, TOTAL_STEPS - CHECKPOINT_STEP)
grain_store.close()

model, optimizer = build_fresh_model()
template = {
    **training_state(model, optimizer),
    "loader": {
        "position": 0,
        "epoch": 0,
        "rng_counts": [0, 0, 0],
        "version": 2,
        # The state names the configuration it is valid for; the template only fixes the shape.
        "fingerprint": {
            "batch_size": 0,
            "length": 0,
            "drop_last": False,
            "num_epochs": 0,
            "shuffled": False,
        },
    },
}
restored, datarax_metadata = restored_payload(datarax_store, template, CHECKPOINT_STEP)
load_training_state(model, optimizer, restored)
pipeline = build_datarax_pipeline()
datarax_iterator = iter(pipeline)
if not isinstance(datarax_iterator, PipelineIterator):
    raise TypeError("a MemorySource pipeline iterates through a PipelineIterator")
datarax_iterator.set_state(restored["loader"])
datarax_after, _ = train_datarax(
    pipeline, model, optimizer, TOTAL_STEPS - CHECKPOINT_STEP, first=datarax_iterator
)
datarax_store.close()

grain_resumed = grain_before + grain_after
datarax_resumed = datarax_before + datarax_after
grain_matches = np.array_equal(grain_resumed, grain_reference)
datarax_matches = np.array_equal(datarax_resumed, datarax_reference)
print(f"Restored Grain step {grain_metadata['step']} (loss {grain_metadata['loss']:.4f})")
print(f"Restored Datarax step {datarax_metadata['step']} (loss {datarax_metadata['loss']:.4f})")
print(f"Grain: resumed run reproduces the reference loss for loss: {grain_matches}")
print(f"Datarax: resumed run reproduces the reference loss for loss: {datarax_matches}")
print(
    f"Steps 20-22 reference {np.round(datarax_reference[20:23], 4)} resumed {np.round(datarax_resumed[20:23], 4)}"
)
# Expected output:
# Restored Grain step 20 (loss 0.0837)
# Restored Datarax step 20 (loss 0.0310)
# Grain: resumed run reproduces the reference loss for loss: True
# Datarax: resumed run reproduces the reference loss for loss: True
# Steps 20-22 reference [0.067  0.0206 0.044 ] resumed [0.067  0.0206 0.044 ]

# %% [markdown]
"""
## Results Summary

| | Grain | Datarax |
|---|---|---|
| Loader state | JSON bytes: last index per worker, sampler and source description | `position`, `epoch`, one count per random stream, `version` |
| Restored into | `set_state(bytes)` on a loader built the same way | `set_state(dict)` on a fresh iterator of a pipeline built the same way |
| Randomness after resume | The draw index continues, so the same generators follow | The epoch and record index continue, so the same keys follow |
| Model and optimizer | `nnx.to_pure_dict` in, `nnx.replace_by_pure_dict` out | The same |
| Store | One `OrbaxCheckpointStore.save(payload, step, loss)` | The same |

Both resumed runs reproduce their reference loss for loss. The difference between the
libraries is what the loader contributes to the payload: bytes that describe a Python
loader, or a small dict that names a position in a compiled one.

## Next Steps

1. [Resumable Training Guide](../advanced/checkpointing/02_resumable_training_guide.py):
   the Datarax pattern with the pipeline's module state and Orbax's `StandardCheckpointer`
2. [Checkpoint Quick Reference](../advanced/checkpointing/01_checkpoint_quickref.py): iterator
   state for every Datarax source
3. [Iteration and Checkpoint State](01_grain_datarax_quickref.py): the quick reference this
   guide builds on

## Cleanup

Both stores are closed, so the temporary checkpoint directory can go.
"""

# %%
shutil.rmtree(checkpoint_root, ignore_errors=True)
print(f"Checkpoint directory removed: {not checkpoint_root.exists()}")
# Expected output:
# Checkpoint directory removed: True


# %%
def main() -> None:
    """Check that both resumed runs reproduce their references."""
    if not (grain_matches and datarax_matches):
        raise SystemExit("a resumed run diverged from the uninterrupted one")
    print("Both resumed runs reproduce the uninterrupted runs.")


if __name__ == "__main__":
    main()
