"""Datarax distributed training components.

This module provides utilities and components for distributed training across
multiple devices and hosts, including:

- DeviceMeshManager: Create and manage JAX device meshes
- DevicePlacement: Explicit device placement utilities
- Data parallel functions: Sharding, SPMD/pmap training steps, gradient reduction
- Metrics functions: Cross-device metric aggregation (SPMD and collective)
- Sharding utilities: MeshRules, factory functions for NNX SPMD patterns

Device Placement Guidelines (per JAX performance guide):

    - Always use explicit device placement for data pipeline outputs
    - Use prefetching to overlap data transfer with compute
    - TPU v5e: Critical batch size >= 240
    - H100 GPU: Critical batch size >= 298
"""

from datarax.distributed.data_parallel import (
    create_data_parallel_sharding,
    data_parallel_train_step,
    place_batch_on_shards,
    place_model_state_on_shards,
    place_nnx_state_on_shards,
    reduce_gradient_tree,
    reduce_gradients_across_devices,
    spmd_train_step,
)
from datarax.distributed.device_mesh import DeviceMeshManager
from datarax.distributed.device_placement import (
    BatchSizeRecommendation,
    DevicePlacement,
    distribute_batch,
    get_batch_size_recommendation,
    HardwareType,
    place_on_device,
    prefetch_to_device,
)
from datarax.distributed.metrics import (
    all_gather,
    collect_from_devices,
    reduce_custom,
    reduce_max,
    reduce_mean,
    reduce_mean_collective,
    reduce_min,
    reduce_sum,
    reduce_sum_collective,
)
from datarax.distributed.sharding import (
    create_named_sharding,
    data_parallel_rules,
    fsdp_rules,
    MeshRules,
    partition_spec_for_names,
)


__all__ = [
    # Mesh management
    "DeviceMeshManager",
    # Device placement
    "DevicePlacement",
    "HardwareType",
    "BatchSizeRecommendation",
    "place_on_device",
    "distribute_batch",
    "get_batch_size_recommendation",
    "prefetch_to_device",
    # Data parallel functions
    "create_data_parallel_sharding",
    "place_batch_on_shards",
    "spmd_train_step",
    "data_parallel_train_step",
    "place_model_state_on_shards",
    "place_nnx_state_on_shards",
    "reduce_gradients_across_devices",
    "reduce_gradient_tree",
    # SPMD metrics (jnp.* on global arrays)
    "reduce_mean",
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_custom",
    # Collective metrics (lax.p* for pmap/shard_map)
    "reduce_mean_collective",
    "reduce_sum_collective",
    "all_gather",
    "collect_from_devices",
    # NNX SPMD sharding
    "MeshRules",
    "data_parallel_rules",
    "fsdp_rules",
    "create_named_sharding",
    "partition_spec_for_names",
]
