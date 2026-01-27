# Control

Pipeline control flow and execution management utilities. These modules handle asynchronous operations and data prefetching for optimal performance.

## Components

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **Prefetcher** | Async data loading | Hide I/O latency |

`★ Insight ─────────────────────────────────────`

- Prefetching loads next batch while GPU processes current
- Overlaps I/O and compute for better throughput
- Most useful when I/O is the bottleneck
- Works automatically with DAGExecutor

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.control import Prefetcher

# Wrap iterator with prefetching
prefetcher = Prefetcher(
    iterator=pipeline,
    prefetch_count=2,  # Keep 2 batches ready
)

for batch in prefetcher:
    # Next batch loads while this one processes
    train_step(batch)
```

## Modules

- [prefetcher](prefetcher.md) - Asynchronous data prefetching for pipeline optimization

## How Prefetching Works

```
Without prefetching:
[Load B1] [Process B1] [Load B2] [Process B2] ...
          ^-- GPU idle during load

With prefetching:
[Load B1] [Load B2   ] [Load B3   ] ...
          [Process B1] [Process B2] ...
          ^-- GPU always busy
```

## Integration with DAG

```python
from datarax.dag import from_source

# Prefetching is built into DAGExecutor
pipeline = from_source(
    source,
    batch_size=32,
    prefetch=2,  # Prefetch 2 batches
)
```

## See Also

- [DAG Executor](../dag/dag_executor.md) - Pipeline execution
- [Performance](../performance/index.md) - Optimization tools
- [Benchmarking](../benchmarking/index.md) - Measure improvements
