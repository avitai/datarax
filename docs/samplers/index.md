# Samplers

Data sampling strategies for pipeline iteration. Samplers control the order in which data elements are accessed, enabling shuffling, sequential access, and epoch-aware iteration.

## Available Samplers

| Sampler | Order | Use Case |
|---------|-------|----------|
| **SequentialSampler** | 0, 1, 2, ... | Validation, inference |
| **ShuffleSampler** | Random | Training |
| **RangeSampler** | Subset range | Debugging, sampling |
| **EpochAwareSampler** | Tracks epochs | Training loops |

`★ Insight ─────────────────────────────────────`

- Samplers generate **indices**, not data
- Use `EpochAwareSampler` for automatic epoch tracking
- `ShuffleSampler` reshuffles each epoch for better training
- Combine with `DataLoader` for complete data loading

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.samplers import ShuffleSampler, SequentialSampler

# Shuffled sampling for training
train_sampler = ShuffleSampler(dataset_size=10000, seed=42)

# Sequential for validation
val_sampler = SequentialSampler(dataset_size=1000)

# Get indices
for idx in train_sampler:
    sample = dataset[idx]
```

## Modules

- [sequential_sampler](sequential_sampler.md) - Sequential index iteration (0, 1, 2, ...)
- [shuffle_sampler](shuffle_sampler.md) - Random shuffled sampling
- [range_sampler](range_sampler.md) - Sample from index range
- [epoch_aware_sampler](epoch_aware_sampler.md) - Automatic epoch tracking

## With DataLoader

```python
from datarax.dag.nodes import DataLoader

loader = DataLoader(
    source=my_source,
    batch_size=32,
    sampler=ShuffleSampler(len(source), seed=42),
)

for batch in loader:
    train_step(batch)
```

## See Also

- [Sources](../sources/index.md) - Data sources
- [DAG Loaders](../dag/loaders.md) - DataLoader node
