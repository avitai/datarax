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
# Grain and Datarax: Randomness and Learnable Operators Tutorial

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~1 min (CPU) |
| **Prerequisites** | [Iteration and Checkpoint State](01_grain_datarax_quickref.py) |
| **Format** | Python + Jupyter |

## Overview

A random transform needs a source of randomness for each record, and a transform with
parameters needs a gradient to reach them. This tutorial runs both in Grain and in
Datarax on the same images and follows what each library does: where a record's
randomness comes from, what changes it, and whether `jax.grad` can see the transform.

A Grain transform is a Python object the loader calls on each record, with a NumPy
generator the sampler hands it. A Datarax operator is an `nnx.Module` inside the
pipeline module: it receives a JAX key derived from the record itself, its variables are
pipeline state, and its computation is JAX code.

## Learning Goals

By the end of this tutorial, you will be able to:

1. Say what a record's randomness depends on in each library, and reproduce it
2. Fit the parameters of a Datarax stage by differentiating an epoch of `Pipeline.scan`
3. Learn the mixture weights of a `WEIGHTED_PARALLEL` composite of image operators
"""

# %% [markdown]
"""
## Setup

Grain is installed as a Datarax dependency.

```bash
uv pip install datarax
```

The records are 64 small random images, each carrying its own index so a batch can say
which records it holds after shuffling.
"""

# %%
import grain
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

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
)
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


NUM_RECORDS = 64
BATCH_SIZE = 8
NUM_BATCHES = NUM_RECORDS // BATCH_SIZE
IMAGE_SHAPE = (8, 8, 3)
NOISE_SCALE = 0.1

images = np.random.default_rng(0).uniform(0.2, 0.8, size=(NUM_RECORDS, *IMAGE_SHAPE))
images = images.astype(np.float32)
index = np.arange(NUM_RECORDS, dtype=np.int32)
print(f"images={images.shape}, index={index.shape}")
# Expected output:
# images=(64, 8, 8, 3), index=(64,)

# %% [markdown]
"""
## Core Concepts

### A generator per draw, or a key per record

| | Grain | Datarax |
|---|---|---|
| Randomness for a record | `np.random.Generator(np.random.Philox(key=seed + draw_index))`, made by the sampler for each draw | `fold_in(fold_in(base_key, epoch), record_index)`, derived by iteration from the operator's stable base key |
| Belongs to | The draw: the position in the sampled sequence | The record: its stable index in the source |
| Changes it | The shuffle order, the worker split | The epoch |
| Reproduce it | Rebuild the generator for that draw index | `datarax.core.prng.per_record_keys(base_key, indices, epoch)` |

### A transform is code, or a module

A Grain transform is a Python object; the loader calls it on each record, outside any JAX
trace, so `jax.grad` cannot reach a parameter it holds. A Datarax stage is an `nnx.Module`
inside the `Pipeline` module: its `nnx.Param` variables are pipeline state, `Pipeline.scan`
runs an epoch as one compiled program, and `nnx.value_and_grad` differentiates through it.
"""

# %% [markdown]
"""
## Part 1: Where a Record's Randomness Comes From

Both loaders add Gaussian noise to each image. The helper below runs a loader over one
epoch and returns the noise each record received, keyed by the record's index, so two runs
can be compared record by record whatever order they served the records in.
"""


# %%
def noise_by_record(batches) -> np.ndarray:
    """Return the noise each record received, indexed by record."""
    noise = np.zeros_like(images)
    for batch in batches:
        batch_index = np.asarray(batch["index"])
        noise[batch_index] = np.asarray(batch["image"]) - images[batch_index]
    return noise


# %% [markdown]
"""
### Grain: a generator per draw

An `IndexSampler` with a seed gives every draw a generator,
`np.random.Generator(np.random.Philox(key=seed + draw_index))`, where the draw index counts
the records served so far and keeps counting across epochs. A `RandomMap` transform receives
that generator with the record. The generator belongs to the draw, not to the record: the
same record served at a different position, such as under another shuffle order, gets
different noise.
"""


# %%
class ImageRecords(grain.sources.RandomAccessDataSource):
    """``{"image": images[i], "index": i}`` records."""

    def __len__(self) -> int:
        """Return the number of records."""
        return NUM_RECORDS

    def __getitem__(self, i: int) -> dict[str, np.ndarray]:
        """Return the record at ``i``."""
        return {"image": images[i], "index": index[i]}


class AddNoise(grain.transforms.RandomMap):
    """Add Gaussian noise drawn from the generator Grain passes with each record."""

    def random_map(self, element: dict, rng: np.random.Generator) -> dict:
        """Return the record with noise added to ``image``."""
        noise = rng.normal(scale=NOISE_SCALE, size=IMAGE_SHAPE).astype(np.float32)
        return {**element, "image": element["image"] + noise}


def build_grain_loader(shuffle_seed: int) -> grain.DataLoader:
    """A shuffled, noisy, batched loader over one epoch of the images."""
    sampler = grain.samplers.IndexSampler(
        num_records=NUM_RECORDS, shuffle=True, num_epochs=1, seed=shuffle_seed
    )
    return grain.DataLoader(
        data_source=ImageRecords(),
        sampler=sampler,
        operations=[AddNoise(), grain.transforms.Batch(batch_size=BATCH_SIZE)],
    )


grain_noise = noise_by_record(build_grain_loader(shuffle_seed=1))
grain_noise_reshuffled = noise_by_record(build_grain_loader(shuffle_seed=2))
grain_same_under_reshuffle = np.array_equal(grain_noise, grain_noise_reshuffled)
print(f"Grain: same noise per record under another shuffle order: {grain_same_under_reshuffle}")

# The first draw of the seed-1 loader used Philox(key=1 + 0). Rebuilding that generator and
# applying the transform's arithmetic reproduces the served image of whichever record came first.
first_batch = next(iter(build_grain_loader(shuffle_seed=1)))
first_record = int(np.asarray(first_batch["index"])[0])
draw_rng = np.random.Generator(np.random.Philox(key=1 + 0))
rebuilt_noise = draw_rng.normal(scale=NOISE_SCALE, size=IMAGE_SHAPE).astype(np.float32)
grain_rebuilt = np.array_equal(images[first_record] + rebuilt_noise, first_batch["image"][0])
print(
    f"Grain: record {first_record} was drawn first; Philox(seed + 0) rebuilds it: {grain_rebuilt}"
)
# Expected output:
# Grain: same noise per record under another shuffle order: False
# Grain: record 35 was drawn first; Philox(seed + 0) rebuilds it: True

# %% [markdown]
"""
### Datarax: a key per record

A stochastic Datarax operator holds one stable base key, and iteration gives each record
the key `fold_in(fold_in(base_key, epoch), record_index)` (`datarax.core.prng.per_record_keys`).
The key belongs to the record: within an epoch its noise is the same under any shuffle order,
batch size, worker split or resume point. The epoch is folded in, so every epoch draws fresh
noise, as Grain's draw index continuing across epochs does.
"""


# %%
def add_noise(element, key):
    """Add Gaussian noise drawn from this record's own key."""
    image = element.data["image"]
    return element.update_data({"image": image + NOISE_SCALE * jax.random.normal(key, image.shape)})


def build_datarax_pipeline(shuffle_seed: int, batch_size: int = BATCH_SIZE) -> Pipeline:
    """A shuffled, noisy, batched pipeline over the images."""
    source = MemorySource(
        MemorySourceConfig(shuffle=True),
        data={"image": images, "index": index},
        rngs=nnx.Rngs(shuffle_seed),
    )
    noise = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="noise"),
        fn=add_noise,
        rngs=nnx.Rngs(noise=0),
    )
    return Pipeline(source=source, stages=[noise], batch_size=batch_size, rngs=nnx.Rngs(0))


datarax_noise = noise_by_record(build_datarax_pipeline(shuffle_seed=1))
datarax_same_under_reshuffle = np.array_equal(
    datarax_noise, noise_by_record(build_datarax_pipeline(shuffle_seed=2))
)
datarax_same_under_rebatch = np.array_equal(
    datarax_noise, noise_by_record(build_datarax_pipeline(shuffle_seed=1, batch_size=16))
)
print(f"Datarax: same noise per record under another shuffle order: {datarax_same_under_reshuffle}")
print(f"Datarax: same noise per record under batch size 16: {datarax_same_under_rebatch}")
# Expected output:
# Datarax: same noise per record under another shuffle order: True
# Datarax: same noise per record under batch size 16: True

# %% [markdown]
"""
## Part 2: Gradients Through a Learnable Stage

A stage with an `nnx.Param` is a trainable part of the pipeline. The loss below runs a
whole epoch through `Pipeline.scan`, so `nnx.value_and_grad` differentiates through the
stage, and `nnx.jit` compiles the step. `reset()` rewinds the pipeline before each epoch.

A Grain transform runs as Python and NumPy code in the loader, outside any JAX trace, so
`jax.grad` cannot reach a parameter inside it. With Grain, a learnable preprocessing step
belongs in the model.

The stage scales each channel; the targets are the images scaled by `TRUE_SCALE`.
"""


# %%
TRUE_SCALE = np.array([2.0, -1.0, 0.5], dtype=np.float32)
targets = images * TRUE_SCALE


class LearnableScale(nnx.Module):
    """Multiply ``image`` by one learnable scale per channel."""

    def __init__(self) -> None:
        """Start every scale at one."""
        self.scale = nnx.Param(jnp.ones(3))

    def __call__(self, batch: dict) -> dict:
        """Return the batch with ``image`` rescaled."""
        return {**batch, "image": batch["image"] * self.scale[...]}


def build_training_pipeline(stage: nnx.Module, target: np.ndarray) -> Pipeline:
    """Batch the images and their targets in order through ``stage``."""
    source = MemorySource(MemorySourceConfig(), data={"image": images, "target": target})
    return Pipeline(source=source, stages=[stage], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0))


def epoch_loss(pipeline: Pipeline) -> jax.Array:
    """Mean squared error between the stage output and the targets over one epoch."""

    def accumulate(total: jax.Array, batch: dict) -> tuple[jax.Array, None]:
        return total + jnp.mean((batch["image"] - batch["target"]) ** 2), None

    total, _ = pipeline.scan(accumulate, length=NUM_BATCHES, init_carry=jnp.zeros(()))
    return total / NUM_BATCHES


@nnx.jit
def gradient_step(pipeline: Pipeline, learning_rate: jax.Array) -> jax.Array:
    """One gradient-descent step on every ``nnx.Param`` in the pipeline.

    The learning rate is a traced array, so the compiled step is reused for any value.
    """
    loss, grads = nnx.value_and_grad(epoch_loss, argnums=nnx.DiffState(0, nnx.Param))(pipeline)
    params = nnx.state(pipeline, nnx.Param)
    nnx.update(pipeline, jax.tree.map(lambda p, g: p - learning_rate * g, params, grads))
    return loss


def train(pipeline: Pipeline, learning_rate: float, epochs: int) -> list[float]:
    """Run ``epochs`` gradient steps, each over one epoch, and return the losses."""
    rate = jnp.float32(learning_rate)
    losses = []
    for _ in range(epochs):
        pipeline.reset()
        losses.append(float(gradient_step(pipeline, rate)))
    return losses


scale_stage = LearnableScale()
scale_losses = train(build_training_pipeline(scale_stage, targets), learning_rate=3.0, epochs=30)
learned_scale = np.asarray(scale_stage.scale[...])
print(f"Loss: first {scale_losses[0]:.4f}, last {scale_losses[-1]:.2e}")
print(f"Learned scale: {np.round(learned_scale, 3)} (true {TRUE_SCALE})")
# Expected output:
# Loss: first 0.4902, last 0.00e+00
# Learned scale: [ 2.  -1.   0.5] (true [ 2.  -1.   0.5])

# %% [markdown]
"""
## Part 3: Learning the Mixture of Image Operators

A `CompositeOperatorModule` with the `WEIGHTED_PARALLEL` strategy applies every child
operator to the same record and mixes their outputs. With `learnable_weights=True` the
mixture is `softmax(logits / temperature)` over an `nnx.Param`, so the same gradient step
as Part 2 learns which operators the data calls for, the relaxation DARTS and Faster
AutoAugment use for augmentation search.

Here the targets are a fixed mixture, one quarter brightened and three quarters
contrast-adjusted, produced by the same two operators with static weights. The learnable
composite starts from equal weights and recovers the mixture.
"""


# %%
def image_operators() -> list:
    """A brightness operator and a contrast operator, both deterministic."""
    brightness = BrightnessOperator(
        BrightnessOperatorConfig(field_key="image", brightness_delta=0.2, stochastic=False),
        rngs=nnx.Rngs(0),
    )
    contrast = ContrastOperator(
        ContrastOperatorConfig(field_key="image", contrast_factor=1.5, stochastic=False),
        rngs=nnx.Rngs(0),
    )
    return [brightness, contrast]


def mixture(weights: list[float], learnable: bool) -> CompositeOperatorModule:
    """Mix the two image operators' outputs with ``weights``."""
    config = CompositeOperatorConfig(
        strategy=CompositionStrategy.WEIGHTED_PARALLEL,
        operators=image_operators(),
        weights=weights,
        learnable_weights=learnable,
    )
    return CompositeOperatorModule(config, rngs=nnx.Rngs(0))


target_mixture = mixture([0.25, 0.75], learnable=False)
target_pipeline = Pipeline(
    source=MemorySource(MemorySourceConfig(), data={"image": images}),
    stages=[target_mixture],
    batch_size=BATCH_SIZE,
    rngs=nnx.Rngs(0),
)
mixed_targets = np.concatenate([np.asarray(batch["image"]) for batch in target_pipeline])

learned_mixture = mixture([0.5, 0.5], learnable=True)
print(f"Mixed field: {learned_mixture.config.mix_fields}")
print(f"Initial mixture weights: {np.round(np.asarray(learned_mixture.mixture_weights()), 3)}")
mixture_losses = train(
    build_training_pipeline(learned_mixture, mixed_targets), learning_rate=5.0, epochs=200
)
learned_weights = np.asarray(learned_mixture.mixture_weights())
print(f"Loss: first {mixture_losses[0]:.5f}, last {mixture_losses[-1]:.2e}")
print(f"Learned mixture weights: {np.round(learned_weights, 3)} (target [0.25 0.75])")
# Expected output:
# Mixed field: ('image',)
# Initial mixture weights: [0.5 0.5]
# Loss: first 0.00296, last 8.50e-10
# Learned mixture weights: [0.25 0.75] (target [0.25 0.75])

# %% [markdown]
"""
## Results Summary

| | Grain | Datarax |
|---|---|---|
| Randomness for a record | `Philox(seed + draw_index)` from the sampler: belongs to the draw | `fold_in(fold_in(base_key, epoch), record_index)`: belongs to the record |
| Changes it | Shuffle order, worker split | The epoch |
| Learnable transform | Not reachable: Python code outside JAX; put it in the model | Any `nnx.Param` in a stage, through `Pipeline.scan` |
| Mixture of operators | A `Map` that applies fixed weights | `WEIGHTED_PARALLEL` with `learnable_weights=True` |

The Datarax stage state travels with the pipeline wherever its module state goes: the
learnable stage recovers the true scales `[2, -1, 0.5]`, and the composite recovers the
mixture `[0.25, 0.75]`, by gradient descent through the pipeline itself.

## Next Steps

1. [Sharding Guide](03_sharding_guide.py): place batches from both libraries on a device
   mesh
2. [DADA Learned Augmentation](../advanced/differentiable/01_dada_learned_augmentation_guide.py):
   a learned augmentation policy trained through a Datarax pipeline
3. [Resumable Training Guide](../advanced/checkpointing/02_resumable_training_guide.py):
   checkpointing pipeline, model and optimizer state together
"""


# %%
def main() -> None:
    """Check the randomness claims and that both learnable stages fit their targets."""
    if grain_same_under_reshuffle or not grain_rebuilt:
        raise SystemExit("Grain noise did not follow Philox(seed + draw_index)")
    if not (datarax_same_under_reshuffle and datarax_same_under_rebatch):
        raise SystemExit("Datarax noise changed with the shuffle order or the batch size")
    if not np.allclose(learned_scale, TRUE_SCALE, atol=1e-2):
        raise SystemExit(f"learned scale {learned_scale} did not reach {TRUE_SCALE}")
    if not np.allclose(learned_weights, [0.25, 0.75], atol=1e-2):
        raise SystemExit(f"learned mixture {learned_weights} did not reach [0.25, 0.75]")
    print("Randomness claims hold and both learnable stages fit their targets.")


if __name__ == "__main__":
    main()
