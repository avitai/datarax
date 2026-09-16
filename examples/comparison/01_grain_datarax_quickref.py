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
# Grain and Datarax: Iteration and Checkpoint State Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~1 min (CPU) |
| **Prerequisites** | [Pipeline Tutorial](../core/02_pipeline_tutorial.py) |
| **Format** | Python + Jupyter |

## Overview

Grain and Datarax both read records, apply a random transform to each record, batch
them, and resume an interrupted epoch from saved iterator state. This quick reference
runs one job with each library, so the two APIs sit side by side: what a checkpoint
holds, where the per-record randomness comes from, and where the transform runs.

Measured throughput and memory for both libraries are on the
[framework comparison page](../../docs/benchmarks/comparison.md); this example makes
no speed claim.

## Learning Goals

By the end of this example, you will be able to:

1. Build a Grain `DataLoader` and a Datarax `Pipeline` over the same records
2. Save iterator state mid-epoch in both libraries and resume from it exactly
3. Say what each library's checkpoint holds and where its randomness comes from
"""

# %% [markdown]
"""
## Setup

Grain is installed as a Datarax dependency.

```bash
uv pip install datarax
```
"""

# %%
import json

import grain
import jax
import numpy as np
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline, PipelineIterator
from datarax.sources import MemorySource, MemorySourceConfig


NUM_RECORDS = 64
BATCH_SIZE = 8
SEED = 0
NOISE_SCALE = 0.1

features = np.arange(NUM_RECORDS * 3, dtype=np.float32).reshape(NUM_RECORDS, 3)
print(f"Records: {features.shape}")
# Expected output:
# Records: (64, 3)

# %% [markdown]
"""
## Step 1: A Grain Loader

Grain reads records by index from any object with `__len__` and `__getitem__`. The
`IndexSampler` fixes the order (shuffled here) and gives every draw a NumPy generator,
which a `RandomMap` transform receives. The `DataLoader` runs the transforms record by
record in Python, in worker processes when `worker_count` is above zero.
"""


# %%
class FeatureRecords(grain.sources.RandomAccessDataSource):
    """One ``{"x": row}`` record per row of ``features``."""

    def __init__(self, values: np.ndarray) -> None:
        """Keep the rows the records are read from."""
        self._values = values

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._values)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return the record at ``index``."""
        return {"x": self._values[index]}

    def __repr__(self) -> str:
        """Describe the records, which Grain compares when it restores a checkpoint."""
        return f"FeatureRecords(shape={self._values.shape}, dtype={self._values.dtype})"


class AddNoise(grain.transforms.RandomMap):
    """Add Gaussian noise drawn from the generator Grain passes with each record."""

    def random_map(self, element: dict, rng: np.random.Generator) -> dict:
        """Return the record with noise added to ``x``."""
        noise = rng.normal(scale=NOISE_SCALE, size=element["x"].shape).astype(np.float32)
        return {"x": element["x"] + noise}


def build_grain_loader() -> grain.DataLoader:
    """A shuffled, noisy, batched loader over one epoch of ``features``."""
    sampler = grain.samplers.IndexSampler(
        num_records=NUM_RECORDS, shuffle=True, num_epochs=1, seed=SEED
    )
    return grain.DataLoader(
        data_source=FeatureRecords(features),
        sampler=sampler,
        operations=[AddNoise(), grain.transforms.Batch(batch_size=BATCH_SIZE)],
    )


grain_iterator = iter(build_grain_loader())
first_grain_batch = next(grain_iterator)
print(f"Grain batch: x={first_grain_batch['x'].shape} {type(first_grain_batch['x']).__name__}")
# Expected output:
# Grain batch: x=(8, 3) ndarray

# %% [markdown]
"""
## Step 2: Resume the Grain Loader

The iterator's `get_state()` returns JSON bytes. They record the last index each worker
served, and the `repr` of the sampler and data source the loader was built with.
`set_state()` on a loader built the same way continues from there; it refuses a
checkpoint whose sampler or data source `repr` differs, which is why `FeatureRecords`
defines `__repr__` from its data rather than inheriting the object address.
"""

# %%
grain_state = grain_iterator.get_state()
print(f"Grain checkpoint keys: {sorted(json.loads(grain_state))}")
expected_grain = [next(grain_iterator)["x"] for _ in range(2)]

resumed_grain = iter(build_grain_loader())
resumed_grain.set_state(grain_state)
resumed_grain_batches = [next(resumed_grain)["x"] for _ in range(2)]
grain_matches = all(
    np.array_equal(got, want)
    for got, want in zip(resumed_grain_batches, expected_grain, strict=True)
)
print(f"Grain resumes exactly: {grain_matches}")
# Expected output:
# Grain checkpoint keys: ['data_source', 'last_seen_indices', 'last_worker_index', 'sampler', 'version', 'worker_count']
# Grain resumes exactly: True

# %% [markdown]
"""
## Step 3: A Datarax Pipeline

A Datarax `Pipeline` is an `nnx.Module`. The source shuffles, and the stochastic
operator draws each record's key from its own stable base key, as
`fold_in(fold_in(base_key, epoch), record_index)`. The operator is a JAX function of one
record, and iteration runs source, stages and batching as one compiled step.
"""


# %%
def add_noise(element, key):
    """Add Gaussian noise drawn from this record's own key."""
    x = element.data["x"]
    return element.update_data({"x": x + NOISE_SCALE * jax.random.normal(key, x.shape)})


def build_datarax_pipeline() -> Pipeline:
    """A shuffled, noisy, batched pipeline over ``features``."""
    source = MemorySource(
        MemorySourceConfig(shuffle=True), data={"x": features}, rngs=nnx.Rngs(SEED)
    )
    noise = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="noise"),
        fn=add_noise,
        rngs=nnx.Rngs(noise=SEED),
    )
    return Pipeline(source=source, stages=[noise], batch_size=BATCH_SIZE, rngs=nnx.Rngs(SEED))


# A pipeline over a random-access source iterates through a checkpointable PipelineIterator.
datarax_iterator = iter(build_datarax_pipeline())
if not isinstance(datarax_iterator, PipelineIterator):
    raise TypeError("a MemorySource pipeline iterates through a PipelineIterator")
first_datarax_batch = next(datarax_iterator)
print(
    f"Datarax batch: x={first_datarax_batch['x'].shape} {type(first_datarax_batch['x']).__name__}"
)
# Expected output:
# Datarax batch: x=(8, 3) ArrayImpl

# %% [markdown]
"""
## Step 4: Resume the Datarax Pipeline

The iterator's `get_state()` returns the records consumed, the epoch, one count per
random stream, and the `version` those counts are in. A stochastic operator contributes
one count, which stays 0: iteration keys each record on the operator's stable base key
and never draws from the operator's own stream. The counts that move belong to the
pipeline and the source. Position and epoch decide the batches, so a pipeline built with
the same seeds continues with the same ones.
"""

# %%
datarax_state = datarax_iterator.get_state()
print(f"Datarax checkpoint: {datarax_state}")
expected_datarax = [np.asarray(next(datarax_iterator)["x"]) for _ in range(2)]

resumed_datarax = iter(build_datarax_pipeline())
if not isinstance(resumed_datarax, PipelineIterator):
    raise TypeError("a MemorySource pipeline iterates through a PipelineIterator")
resumed_datarax.set_state(datarax_state)
resumed_datarax_batches = [np.asarray(next(resumed_datarax)["x"]) for _ in range(2)]
datarax_matches = all(
    np.array_equal(got, want)
    for got, want in zip(resumed_datarax_batches, expected_datarax, strict=True)
)
print(f"Datarax resumes exactly: {datarax_matches}")
# Expected output:
# Datarax checkpoint: {'position': np.int64(8), 'epoch': 0, 'rng_counts': [0, 1, 0], 'version': 1}
# Datarax resumes exactly: True

# %% [markdown]
"""
## Results Summary

| | Grain | Datarax |
|---|---|---|
| Loop object | `DataLoader` iterator | `Pipeline` (an `nnx.Module`) iterator |
| Checkpoint | JSON bytes: last index per worker, sampler and source description | `position`, `epoch`, one count per random stream, and their `version` |
| Per-record randomness | `np.random.Generator(Philox(key=seed + draw index))` from the sampler | `fold_in(fold_in(base_key, epoch), record_index)` from the operator's stable base key |
| Where transforms run | Python, per record, optionally in worker processes | Inside one `jax.jit` step with batching |

Both loops resume mid-epoch exactly. The difference is where the work runs: Grain keeps
transforms in Python around the data source, while Datarax traces them into the same
compiled program as batching, which is also what lets a later example differentiate
through a pipeline.

## Next Steps

1. [Stateful Operators Tutorial](02_stateful_operators_tutorial.py): randomness, running
   statistics and gradients through operators in both libraries
2. [Checkpoint Quick Reference](../advanced/checkpointing/01_checkpoint_quickref.py):
   saving iterator state with Orbax
3. [Framework comparison](../../docs/benchmarks/comparison.md): measured throughput and
   memory
"""


# %%
def main() -> None:
    """Run both loaders and check that each resumes exactly."""
    print(f"Grain resumes exactly: {grain_matches}")
    print(f"Datarax resumes exactly: {datarax_matches}")
    if not (grain_matches and datarax_matches):
        raise SystemExit("a resumed loader diverged from the uninterrupted one")


if __name__ == "__main__":
    main()
