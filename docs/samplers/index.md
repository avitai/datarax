# Samplers

Data sampling strategies for pipeline iteration. Samplers control the
order in which data elements are accessed, enabling shuffling,
sequential access, and epoch-aware iteration. Sources consume a
sampler internally to translate iteration into indexed reads.

## Available Samplers

| Sampler | Order | Use Case |
|---------|-------|----------|
| **SequentialSampler** | 0, 1, 2, ... | Validation, inference |
| **ShuffleSampler** | Random | Training |
| **RangeSampler** | Subset range | Debugging, partial sweeps |
| **EpochAwareSampler** | Tracks epochs | Training loops |

`★ Insight ─────────────────────────────────────`

- Samplers generate **indices**, not data
- `ShuffleSampler` reshuffles each epoch for better training
- `EpochAwareSampler` tracks the epoch counter for callers that need it
- All samplers are `nnx.Module` subclasses and integrate with `Pipeline` checkpointing

`─────────────────────────────────────────────────`

## Quick Start

```python
from flax import nnx

from datarax.samplers import (
    SequentialSamplerConfig,
    SequentialSamplerModule,
    ShuffleSampler,
    ShuffleSamplerConfig,
)

# Shuffled sampling for training
train_sampler = ShuffleSampler(
    ShuffleSamplerConfig(dataset_size=10_000, seed=42),
    rngs=nnx.Rngs(0),
)

# Sequential for validation
val_sampler = SequentialSamplerModule(
    SequentialSamplerConfig(num_records=1_000, num_epochs=1),
    rngs=nnx.Rngs(0),
)

# Each sampler emits dataset indices.
for idx in val_sampler:
    sample = dataset[int(idx)]  # noqa: F821 (illustrative)
```

## Modules

- [sequential_sampler](sequential_sampler.md) - Sequential index iteration (0, 1, 2, ...)
- [shuffle_sampler](shuffle_sampler.md) - Random shuffled sampling
- [range_sampler](range_sampler.md) - Sample from index range
- [epoch_aware_sampler](epoch_aware_sampler.md) - Automatic epoch tracking

## Integration with Pipeline

Sources own their own shuffling — set ``shuffle=True`` on the source
config and the source's iterator emits indices in shuffled order via
the built-in Feistel-cipher index shuffle. Pipeline then orchestrates
batched access.

```python
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(
    MemorySourceConfig(shuffle=True),
    data=data,
    rngs=nnx.Rngs(seed=42),
)

pipeline = Pipeline(
    source=source,
    stages=[normalize, augment],
    batch_size=32,
    rngs=nnx.Rngs(0),
)

for batch in pipeline:
    train_step(batch)
```

Use the standalone sampler modules above (``ShuffleSampler``,
``SequentialSamplerModule``, etc.) when you need the index sequence
for a custom integration outside Pipeline.

## See Also

- [Sources](../sources/index.md) - Data sources
- [Pipeline](../user_guide/dag_construction.md) - Pipeline construction
