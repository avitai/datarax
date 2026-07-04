# Control

Pipeline control flow and execution management utilities. These modules
handle asynchronous operations and data prefetching for optimal performance.

## Components

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **Prefetcher** | Async data loading | Hide I/O latency |
| **DevicePrefetcher** | Overlap host→device transfer | Keep the accelerator fed |
| **create_prefetch_stream** | Backend-selectable prefetch | Grain / Flax / thread modes |

!!! note "Key points"

    - Prefetching loads the next batch while the accelerator processes the current one
    - Overlaps I/O and compute for better throughput
    - Most useful when I/O is the bottleneck
    - Prefetching is applied by wrapping a pipeline's output stream — it is
      not enabled automatically

## Quick Start

Wrap an existing iterator (for example, `iter(pipeline)`) to prefetch
batches ahead of consumption:

```python
from datarax.control import Prefetcher

# Buffer up to 2 batches ahead of the consumer
stream = Prefetcher(buffer_size=2).prefetch(iter(pipeline))

for batch in stream:
    # The next batch is prepared while this one is processed
    train_step(batch)
```

## Modules

- [prefetcher](prefetcher.md) - Asynchronous data prefetching for pipeline optimization

## How Prefetching Works

```
Without prefetching:
[Load B1] [Process B1] [Load B2] [Process B2] ...
          ^-- accelerator idle during load

With prefetching:
[Load B1] [Load B2   ] [Load B3   ] ...
          [Process B1] [Process B2] ...
          ^-- accelerator always busy
```

## Backend-Selectable Prefetch

`create_prefetch_stream` selects the prefetch backend by mode — `"none"`
(pass-through), `"grain"` (Grain's `device_put` double-buffer), `"flax"`
(`flax.jax_utils.prefetch_to_device`), or `"thread"` (Datarax's closeable
threaded wrapper):

```python
from datarax.control import create_prefetch_stream

stream = create_prefetch_stream(iter(pipeline), mode="thread", size=2)
for batch in stream:
    train_step(batch)
```

`DevicePrefetcher` provides an explicit host→device double buffer for the
same purpose.

## See Also

- [Pipeline](../dag/index.md) - Pipeline construction and execution
- [Performance](../performance/index.md) - Optimization tools
- [Benchmarking](../benchmarking/index.md) - Measure improvements
