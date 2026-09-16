# Grain and Datarax: Iteration and Checkpoint State Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~1 min (CPU) |
| **Prerequisites** | [Pipeline Tutorial](../core/pipeline-tutorial.md) |
| **Format** | Python + Jupyter |
| **Source** | `examples/comparison/01_grain_datarax_quickref.py` |

## Overview

Grain and Datarax both read records, apply a random transform to each record, batch them,
and resume an interrupted epoch from saved iterator state. This quick reference runs one job
with each library, so the two APIs sit side by side: what a checkpoint holds, where the
per-record randomness comes from, and where the transform runs.

Grain is installed as a Datarax dependency, so the example needs nothing beyond `datarax`.
Measured throughput and memory for both libraries are on the
[framework comparison page](../../benchmarks/comparison.md); this example makes no speed claim.

## What You'll Learn

1. Build a Grain `DataLoader` and a Datarax `Pipeline` over the same records
2. Save iterator state mid-epoch in both libraries and resume from it exactly
3. Say what each library's checkpoint holds and where its randomness comes from

## Coming from Grain?

| Grain | Datarax |
|-------|---------|
| `RandomAccessDataSource` with `__len__` and `__getitem__` | `MemorySource` over arrays, reading records by index |
| `IndexSampler(shuffle=True, seed=...)` fixes the order | `MemorySourceConfig(shuffle=True)` with the source's `nnx.Rngs` |
| `RandomMap.random_map(element, rng)` receives a NumPy generator per record | `ElementOperator` receives each record's JAX key |
| `transforms.Batch(batch_size=...)` as an operation | `Pipeline(..., batch_size=...)` |
| `iterator.get_state()` returns JSON bytes | `iterator.get_state()` returns `position`, `epoch`, `rng_counts` and `version` |
| Transforms run in Python, per record, optionally in worker processes | Source, stages and batching run as one compiled step |

## What Each Checkpoint Holds

A Grain checkpoint records the last index each worker served and the `repr` of the sampler
and data source the loader was built with. `set_state()` refuses a checkpoint whose sampler
or data-source `repr` differs, so a data source defines `__repr__` from its data rather than
inheriting the object address.

A Datarax checkpoint records the records consumed, the epoch, one count per random stream,
and the `version` those counts are in. A stochastic operator contributes one count, which
stays 0: iteration keys each record as `fold_in(fold_in(base_key, epoch), record_index)`
from the operator's stable base key, and never draws from the operator's own stream. The
counts that move belong to the pipeline and the source.

Both loops resume mid-epoch exactly. The difference is where the work runs: Grain keeps
transforms in Python around the data source, while Datarax traces them into the same
compiled program as batching.

## Next Steps

1. [Checkpoint Quick Reference](../advanced/checkpointing/checkpoint-quickref.md): saving
   iterator state with Orbax
2. [Framework comparison](../../benchmarks/comparison.md): measured throughput and memory
