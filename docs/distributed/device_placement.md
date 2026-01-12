# Device Placement

::: datarax.distributed.device_placement

## Overview

The `device_placement` module provides utilities for explicit device placement of JAX arrays and PyTrees, enabling efficient data distribution across accelerators. It includes hardware-aware batch size recommendations based on JAX performance guidelines.

## Key Components

### HardwareType

Enumeration of supported hardware types for batch size recommendations:

```python
from datarax.distributed.device_placement import HardwareType

# Available hardware types:
# - TPU_V5E, TPU_V5P, TPU_V4
# - H100, A100, V100 (GPUs)
# - CPU
# - UNKNOWN (conservative defaults)
```

### BatchSizeRecommendation

Dataclass containing hardware-specific batch size recommendations:

```python
from datarax.distributed.device_placement import BatchSizeRecommendation

# Fields:
# - min_batch_size: Minimum for reasonable efficiency
# - optimal_batch_size: For peak throughput
# - critical_batch_size: For reaching roofline performance
# - max_memory_batch_size: Before OOM (estimate)
# - notes: Additional guidance
```

### DevicePlacement Class

Main utility class for device placement operations.

## Usage Examples

### Basic Device Placement

```python
import jax
import jax.numpy as jnp
from datarax.distributed.device_placement import DevicePlacement

# Create placement utility
placement = DevicePlacement()

# Place data on specific device
data = jnp.ones((256, 224, 224, 3))
placed = placement.place_on_device(data, jax.devices()[0])

# Check detected hardware
print(f"Hardware type: {placement.hardware_type}")
print(f"Number of devices: {placement.num_devices}")
```

### Getting Batch Size Recommendations

```python
from datarax.distributed.device_placement import (
    DevicePlacement,
    get_batch_size_recommendation,
    HardwareType
)

# Auto-detect hardware and get recommendations
placement = DevicePlacement()
rec = placement.get_batch_size_recommendation()

print(f"Minimum batch size: {rec.min_batch_size}")
print(f"Optimal batch size: {rec.optimal_batch_size}")
print(f"Critical batch size: {rec.critical_batch_size}")
print(f"Notes: {rec.notes}")

# Or use the convenience function
rec = get_batch_size_recommendation()

# Get for specific hardware
h100_rec = get_batch_size_recommendation(HardwareType.H100)
print(f"H100 optimal batch: {h100_rec.optimal_batch_size}")  # 320
```

### Validating Batch Size

```python
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

# Validate a batch size
is_valid, message = placement.validate_batch_size(64)
print(message)

# Validate without suboptimal warnings
is_valid, message = placement.validate_batch_size(64, warn_suboptimal=False)
```

### Distributing Data Across Devices

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

# Create a device mesh
devices = np.array(jax.devices()).reshape(-1)
mesh = Mesh(devices, axis_names=("data",))

# Create sharding specification
sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))

# Distribute batch across devices
data = jnp.ones((8, 28, 28, 3))
distributed = placement.distribute_batch(data, sharding)
```

### Sharding Along Batch Dimension

```python
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

# Create mesh
devices = np.array(jax.devices()).reshape(-1)
mesh = Mesh(devices, axis_names=("data",))

# Shard PyTree along batch dimension
data = {
    "images": jnp.ones((8, 28, 28, 3)),
    "labels": jnp.ones((8,), dtype=jnp.int32)
}

sharded = placement.shard_batch_dim(data, mesh, batch_axis=0, mesh_axis="data")
```

### Replicating Data Across Devices

```python
import jax
import jax.numpy as jnp
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

# Replicate model weights across all devices
weights = {"w": jnp.ones((128, 64)), "b": jnp.zeros(64)}
replicated = placement.replicate_across_devices(weights)
```

### Prefetching to Device

```python
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()

def data_generator():
    for i in range(100):
        yield {"batch": i}

# Create prefetching iterator
prefetched = placement.prefetch_to_device(
    data_generator(),
    buffer_size=2  # Prefetch 2 batches ahead
)

for batch in prefetched:
    # Process batch (already on device)
    pass
```

### Getting Device Information

```python
from datarax.distributed.device_placement import DevicePlacement

placement = DevicePlacement()
info = placement.get_device_info()

print(f"Number of devices: {info['num_devices']}")
print(f"Hardware type: {info['hardware_type']}")
print(f"Platforms: {info['platforms']}")
```

## Hardware-Specific Recommendations

Based on JAX performance guidelines:

| Hardware | Min Batch | Optimal | Critical | Notes |
|----------|-----------|---------|----------|-------|
| TPU v5e  | 64        | 256     | 240      | Critical for roofline |
| TPU v5p  | 128       | 512     | 480      | Higher throughput variant |
| TPU v4   | 64        | 256     | 192      | Similar to v5e |
| H100     | 64        | 320     | 298      | Critical for roofline |
| A100     | 32        | 256     | 240      | 80GB variant |
| V100     | 16        | 128     | 96       | Memory-limited |
| CPU      | 1         | 32      | 16       | Bandwidth-bound |

## Integration with Datarax Pipelines

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.distributed.device_placement import DevicePlacement, get_batch_size_recommendation
import jax.numpy as jnp

# Get recommended batch size
rec = get_batch_size_recommendation()
batch_size = rec.optimal_batch_size

# Create pipeline with optimal batch size
data = [{"image": jnp.ones((28, 28, 3))} for _ in range(1000)]
source = MemorySource(MemorySourceConfig(), data)
pipeline = from_source(source, batch_size=batch_size)

# Use device placement for explicit placement
placement = DevicePlacement()

for batch in pipeline:
    # Place batch on device explicitly
    placed_batch = placement.place_on_device(batch)
    # Process placed batch...
    break
```

## Best Practices

1. **Use critical batch size**: For maximum throughput, aim for at least the critical batch size for your hardware.

2. **Validate early**: Use `validate_batch_size()` during pipeline setup to catch suboptimal configurations.

3. **Explicit placement**: Use `place_on_device()` or `distribute_batch()` for explicit device placement of pipeline outputs.

4. **Prefetch for overlap**: Use `prefetch_to_device()` to overlap data transfer with computation.

5. **Check hardware detection**: Use `get_device_info()` to verify correct hardware detection.

## Convenience Functions

The module also provides standalone functions:

```python
from datarax.distributed.device_placement import (
    place_on_device,        # Place data on device
    distribute_batch,       # Distribute with sharding
    get_batch_size_recommendation  # Get recommendations
)
```
