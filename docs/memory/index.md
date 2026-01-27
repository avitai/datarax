# Memory

Memory management utilities for efficient data handling. These tools help manage shared memory for multi-process data loading.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **SharedMemoryManager** | Cross-process sharing | Multi-worker dataloading |

`★ Insight ─────────────────────────────────────`

- Shared memory avoids copying data between processes
- Essential for multi-worker data loading
- Managed automatically by DataLoader
- Use for custom multi-process pipelines

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.memory import SharedMemoryManager

# Create shared memory region
manager = SharedMemoryManager(size_bytes=1024 * 1024 * 100)  # 100MB

# Write data (from worker process)
manager.write("batch_0", batch_data)

# Read data (from main process)
batch = manager.read("batch_0")

# Cleanup
manager.close()
```

## Modules

- [shared_memory_manager](shared_memory_manager.md) - Shared memory for multi-process data sharing

## Multi-Worker Pattern

```python
from datarax.memory import SharedMemoryManager
from multiprocessing import Process

def worker(manager, worker_id):
    for i in range(100):
        batch = load_batch(i)
        manager.write(f"worker_{worker_id}_batch_{i}", batch)

# Create shared memory
manager = SharedMemoryManager(size_bytes=1e9)

# Spawn workers
workers = [Process(target=worker, args=(manager, i)) for i in range(4)]
for w in workers:
    w.start()

# Read from main process
for batch_key in manager.keys():
    batch = manager.read(batch_key)
    process(batch)
```

## See Also

- [DAG Loaders](../dag/loaders.md) - DataLoader with workers
- [Control](../control/index.md) - Prefetching
- [Performance](../performance/index.md) - Optimization
