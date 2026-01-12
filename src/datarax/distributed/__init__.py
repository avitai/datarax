"""Datarax distributed training components.

This module provides utilities and components for distributed training across
multiple devices and hosts, including:

- DeviceMeshManager: Create and manage JAX device meshes
- DevicePlacement: Explicit device placement utilities
- DataParallel: Data-parallel training utilities
- DistributedMetrics: Metrics aggregation across devices

Device Placement Guidelines (per JAX performance guide):
    - Always use explicit device placement for data pipeline outputs
    - Use prefetching to overlap data transfer with compute
    - TPU v5e: Critical batch size >= 240
    - H100 GPU: Critical batch size >= 298
"""

# Re-export specific modules
from datarax.distributed.data_parallel import DataParallel
from datarax.distributed.device_mesh import DeviceMeshManager
from datarax.distributed.device_placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    HardwareType,
    distribute_batch,
    get_batch_size_recommendation,
    place_on_device,
)
from datarax.distributed.metrics import DistributedMetrics


__all__ = [
    "DataParallel",
    "DeviceMeshManager",
    "DevicePlacement",
    "DistributedMetrics",
    "HardwareType",
    "BatchSizeRecommendation",
    "place_on_device",
    "distribute_batch",
    "get_batch_size_recommendation",
]
