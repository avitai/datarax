# Memory

Memory management utilities for efficient data handling. These tools help
manage shared memory for multi-process data loading.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **SharedMemoryManager** | Cross-process sharing | Multi-worker dataloading |

!!! note "Key points"

    - Shared memory avoids copying data between processes
    - Arrays of at least 1 MiB are automatically placed in shared memory
    - Used by `datarax.workers` for cross-process transforms
    - Use for custom multi-process pipelines

## Quick Start

```python
import numpy as np
from datarax.memory import SharedMemoryManager

# The manager sizes segments automatically
manager = SharedMemoryManager()

# Publish an array to shared memory (arrays >= 1 MiB are shared)
shared = manager.make_shared("batch_0", np.zeros((1024, 1024), dtype=np.float32))

# Retrieve it from another reference
batch = manager.get_shared("batch_0")

# Release all segments
manager.cleanup()
```

## Modules

- [shared_memory_manager](shared_memory_manager.md) - Shared memory for multi-process data sharing

## Multi-Worker Pattern

Use the context manager so segments are always released:

```python
import numpy as np
from datarax.memory import SharedMemoryManager

with SharedMemoryManager() as manager:
    for i in range(100):
        batch = load_batch(i)
        manager.make_shared(f"batch_{i}", np.asarray(batch))

    # Retrieve by name
    batch = manager.get_shared("batch_0")
    process(batch)
# Segments are cleaned up on exit
```

## See Also

- [Workers](../root/index.md) - Cross-process transform workers
- [Control](../control/index.md) - Prefetching
- [Performance](../performance/index.md) - Optimization
