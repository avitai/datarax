"""Device placement utilities for JAX distributed training.

This module provides utilities for explicit device placement of JAX arrays and
PyTrees, enabling efficient data distribution across accelerators.

Key Features:
    - Explicit device placement with jax.device_put
    - Hardware-aware batch size recommendations
    - PyTree-aware device placement
    - Prefetching utilities for overlapping compute and data transfer

Performance Guidelines (per JAX guide):
    - TPU v5e: Critical batch size >= 240 for optimal throughput
    - H100 GPU: Critical batch size >= 298 for optimal throughput
    - Always use explicit device placement for data pipeline outputs
    - Prefetch to device memory to overlap data transfer with compute

Example:
    >>> from datarax.distributed.device_placement import DevicePlacement
    >>> placement = DevicePlacement()
    >>>
    >>> # Place data on specific device
    >>> data = jnp.ones((256, 224, 224, 3))
    >>> placed = placement.place_on_device(data, jax.devices()[0])
    >>>
    >>> # Distribute batch across devices
    >>> mesh = Mesh(np.array(jax.devices()), axis_names=("data",))
    >>> sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))
    >>> distributed = placement.distribute_batch(data, sharding)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding, SingleDeviceSharding

# Note: jax.Device type is used with string annotation to avoid import issues

# PyTree is a conceptual type for JAX tree structures
# Using Any for type hints as JAX PyTree can be any nested structure
PyTree = Any


class HardwareType(Enum):
    """Enumeration of supported hardware types."""

    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"
    TPU_V4 = "tpu_v4"
    H100 = "h100"
    A100 = "a100"
    V100 = "v100"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BatchSizeRecommendation:
    """Hardware-specific batch size recommendations.

    Attributes:
        min_batch_size: Minimum batch size for reasonable efficiency.
        optimal_batch_size: Optimal batch size for peak throughput.
        critical_batch_size: Critical batch size for reaching roofline (per JAX guide).
        max_memory_batch_size: Maximum batch size before OOM (estimate).
        notes: Additional notes about the recommendation.
    """

    min_batch_size: int
    optimal_batch_size: int
    critical_batch_size: int
    max_memory_batch_size: int | None = None
    notes: str = ""


# Hardware-specific batch size recommendations per JAX performance guide
_BATCH_SIZE_RECOMMENDATIONS: dict[HardwareType, BatchSizeRecommendation] = {
    HardwareType.TPU_V5E: BatchSizeRecommendation(
        min_batch_size=64,
        optimal_batch_size=256,
        critical_batch_size=240,
        notes="Critical batch size for reaching roofline on TPU v5e.",
    ),
    HardwareType.TPU_V5P: BatchSizeRecommendation(
        min_batch_size=128,
        optimal_batch_size=512,
        critical_batch_size=480,
        notes="Higher throughput variant, benefits from larger batches.",
    ),
    HardwareType.TPU_V4: BatchSizeRecommendation(
        min_batch_size=64,
        optimal_batch_size=256,
        critical_batch_size=192,
        notes="Similar characteristics to TPU v5e but slightly lower critical batch.",
    ),
    HardwareType.H100: BatchSizeRecommendation(
        min_batch_size=64,
        optimal_batch_size=320,
        critical_batch_size=298,
        notes="Critical batch size for reaching roofline on H100.",
    ),
    HardwareType.A100: BatchSizeRecommendation(
        min_batch_size=32,
        optimal_batch_size=256,
        critical_batch_size=240,
        notes="A100 80GB variant. 40GB variant may need smaller batches.",
    ),
    HardwareType.V100: BatchSizeRecommendation(
        min_batch_size=16,
        optimal_batch_size=128,
        critical_batch_size=96,
        notes="Older generation, memory-limited on 16GB variant.",
    ),
    HardwareType.CPU: BatchSizeRecommendation(
        min_batch_size=1,
        optimal_batch_size=32,
        critical_batch_size=16,
        notes="CPU is memory-bandwidth bound, smaller batches often sufficient.",
    ),
    HardwareType.UNKNOWN: BatchSizeRecommendation(
        min_batch_size=32,
        optimal_batch_size=128,
        critical_batch_size=64,
        notes="Conservative defaults for unknown hardware.",
    ),
}


class DevicePlacement:
    """Utility class for explicit device placement of JAX arrays.

    This class provides methods for placing arrays on specific devices,
    distributing batches across devices using sharding, and providing
    hardware-aware batch size recommendations.

    Example:
        >>> placement = DevicePlacement()
        >>> data = jnp.ones((4, 8))
        >>>
        >>> # Place on first GPU
        >>> gpu_data = placement.place_on_device(data, jax.devices("gpu")[0])
        >>>
        >>> # Get batch size recommendation
        >>> rec = placement.get_batch_size_recommendation()
        >>> print(f"Optimal batch: {rec.optimal_batch_size}")
    """

    def __init__(self, default_device: jax.Device | None = None):  # type: ignore[name-defined]
        """Initialize DevicePlacement.

        Args:
            default_device: Default device to use when none is specified.
                If None, uses jax.devices()[0].
        """
        self.default_device = default_device or jax.devices()[0]
        self._hardware_type = self._detect_hardware_type()

    def _detect_hardware_type(self) -> HardwareType:
        """Detect the hardware type from available devices.

        Returns:
            HardwareType enum value.
        """
        devices = jax.devices()
        if not devices:
            return HardwareType.UNKNOWN

        device = devices[0]
        platform = device.platform.lower()
        device_kind = str(device.device_kind).lower() if device.device_kind else ""

        if platform == "tpu":
            if "v5e" in device_kind or "v5 lite" in device_kind:
                return HardwareType.TPU_V5E
            elif "v5p" in device_kind:
                return HardwareType.TPU_V5P
            elif "v4" in device_kind:
                return HardwareType.TPU_V4
            return HardwareType.TPU_V4  # Default TPU

        elif platform == "gpu" or platform == "cuda":
            if "h100" in device_kind:
                return HardwareType.H100
            elif "a100" in device_kind:
                return HardwareType.A100
            elif "v100" in device_kind:
                return HardwareType.V100
            return HardwareType.A100  # Default modern GPU

        elif platform == "cpu":
            return HardwareType.CPU

        return HardwareType.UNKNOWN

    def place_on_device(
        self,
        data: PyTree,
        device: jax.Device | None = None,  # type: ignore[name-defined]
    ) -> PyTree:
        """Place data on a specific device.

        This uses jax.device_put for explicit device placement, ensuring
        data is transferred to the target device.

        Args:
            data: PyTree of JAX arrays to place on device.
            device: Target device. If None, uses the default device.

        Returns:
            PyTree with arrays placed on the specified device.

        Example:
            >>> data = {"images": jnp.ones((4, 28, 28, 3))}
            >>> gpu_data = placement.place_on_device(data, jax.devices("gpu")[0])
        """
        device = device or self.default_device
        sharding = SingleDeviceSharding(device)
        return jax.device_put(data, sharding)

    def distribute_batch(
        self,
        data: PyTree,
        sharding: Sharding,
    ) -> PyTree:
        """Distribute data across devices using the specified sharding.

        This applies explicit device placement using jax.device_put with
        a Sharding object, distributing the data across multiple devices.

        Args:
            data: PyTree of JAX arrays to distribute.
            sharding: JAX Sharding specification.

        Returns:
            PyTree with arrays distributed according to the sharding.

        Example:
            >>> mesh = Mesh(np.array(jax.devices()), ("data",))
            >>> sharding = NamedSharding(mesh, PartitionSpec("data", None))
            >>> distributed = placement.distribute_batch(data, sharding)
        """
        return jax.device_put(data, sharding)

    def replicate_across_devices(
        self,
        data: PyTree,
        devices: list[jax.Device] | None = None,  # type: ignore[name-defined]
    ) -> PyTree:
        """Replicate data across all specified devices.

        Creates a copy of the data on each device, useful for broadcasting
        model weights or constants.

        Args:
            data: PyTree of JAX arrays to replicate.
            devices: List of devices to replicate to. If None, uses all devices.

        Returns:
            PyTree with arrays replicated across devices.
        """
        if devices is None:
            devices = jax.devices()

        # Create a mesh with a single replicated dimension
        import numpy as np

        mesh = Mesh(np.array(devices), axis_names=("replica",))
        sharding = NamedSharding(mesh, PartitionSpec(None))  # Replicated
        return jax.device_put(data, sharding)

    def shard_batch_dim(
        self,
        data: PyTree,
        mesh: Mesh,
        batch_axis: int = 0,
        mesh_axis: str = "data",
    ) -> PyTree:
        """Shard data along the batch dimension.

        This is the most common sharding pattern for data-parallel training,
        where each device processes a slice of the batch.

        Args:
            data: PyTree of JAX arrays to shard.
            mesh: Device mesh to shard across.
            batch_axis: The axis index representing the batch dimension.
            mesh_axis: The mesh axis name to shard along.

        Returns:
            PyTree with arrays sharded along the batch dimension.
        """

        def create_pspec(leaf: Any) -> PartitionSpec | None:
            """Create appropriate PartitionSpec for a leaf."""
            if not isinstance(leaf, jax.Array):
                return None
            ndim = len(leaf.shape)
            if ndim == 0:
                return PartitionSpec()  # Scalar, replicate
            # Create spec with mesh_axis at batch_axis position
            axes: list[str | None] = [None] * ndim
            if batch_axis < ndim:
                axes[batch_axis] = mesh_axis
            return PartitionSpec(*axes)

        # Get leaf structure and create shardings
        leaves, treedef = jax.tree.flatten(data)
        pspecs = [create_pspec(leaf) for leaf in leaves]
        shardings = [NamedSharding(mesh, pspec) if pspec is not None else None for pspec in pspecs]

        # Apply sharding to each leaf
        sharded_leaves = [
            jax.device_put(leaf, sharding) if sharding is not None else leaf
            for leaf, sharding in zip(leaves, shardings, strict=True)
        ]

        return jax.tree.unflatten(treedef, sharded_leaves)

    def prefetch_to_device(
        self,
        data_iterator: Any,
        device: jax.Device | None = None,  # type: ignore[name-defined]
        buffer_size: int = 2,
    ) -> Any:
        """Create a prefetching wrapper that places data on device asynchronously.

        This enables overlapping data transfer with computation for improved
        throughput in training loops.

        Args:
            data_iterator: Iterator yielding PyTrees of data.
            device: Target device for prefetching.
            buffer_size: Number of batches to prefetch.

        Returns:
            Iterator that yields device-placed data.

        Note:
            This is a simple implementation. For production use, consider
            using jax.experimental.multihost_utils or custom async prefetching.
        """
        device = device or self.default_device

        # Create a simple prefetch buffer
        def prefetch_gen():
            # Prefetch buffer_size items ahead
            buffer = []
            for item in data_iterator:
                placed = self.place_on_device(item, device)
                buffer.append(placed)
                if len(buffer) > buffer_size:
                    yield buffer.pop(0)
            # Drain remaining buffer
            yield from buffer

        return prefetch_gen()

    def get_batch_size_recommendation(
        self,
        hardware_type: HardwareType | None = None,
    ) -> BatchSizeRecommendation:
        """Get batch size recommendation for the current hardware.

        Args:
            hardware_type: Override hardware type. If None, uses detected type.

        Returns:
            BatchSizeRecommendation with hardware-specific values.
        """
        hw_type = hardware_type or self._hardware_type
        return _BATCH_SIZE_RECOMMENDATIONS.get(
            hw_type, _BATCH_SIZE_RECOMMENDATIONS[HardwareType.UNKNOWN]
        )

    def validate_batch_size(
        self,
        batch_size: int,
        warn_suboptimal: bool = True,
    ) -> tuple[bool, str]:
        """Validate batch size against hardware recommendations.

        Args:
            batch_size: The batch size to validate.
            warn_suboptimal: Whether to warn for suboptimal (but valid) sizes.

        Returns:
            Tuple of (is_valid, message).
        """
        rec = self.get_batch_size_recommendation()

        if batch_size < rec.min_batch_size:
            return (
                False,
                f"Batch size {batch_size} is below minimum recommended {rec.min_batch_size} "
                f"for {self._hardware_type.value}. Performance will be significantly degraded.",
            )

        if batch_size < rec.critical_batch_size:
            if warn_suboptimal:
                return (
                    True,
                    f"Batch size {batch_size} is below critical size {rec.critical_batch_size} "
                    f"for {self._hardware_type.value}. Consider increasing for optimal throughput.",
                )

        if batch_size >= rec.optimal_batch_size:
            return True, f"Batch size {batch_size} is optimal for {self._hardware_type.value}."

        return True, f"Batch size {batch_size} is acceptable for {self._hardware_type.value}."

    @property
    def hardware_type(self) -> HardwareType:
        """Get the detected hardware type."""
        return self._hardware_type

    @property
    def num_devices(self) -> int:
        """Get the number of available devices."""
        return len(jax.devices())

    def get_device_info(self) -> dict[str, Any]:
        """Get information about available devices.

        Returns:
            Dictionary containing device information.
        """
        devices = jax.devices()
        return {
            "num_devices": len(devices),
            "hardware_type": self._hardware_type.value,
            "platforms": list({d.platform for d in devices}),
            "device_kinds": list({str(d.device_kind) for d in devices if d.device_kind}),
            "devices": [
                {
                    "id": d.id,
                    "platform": d.platform,
                    "device_kind": str(d.device_kind) if d.device_kind else None,
                }
                for d in devices
            ],
        }


def place_on_device(data: PyTree, device: jax.Device | None = None) -> PyTree:  # type: ignore[name-defined]
    """Convenience function for placing data on a device.

    Args:
        data: PyTree of JAX arrays.
        device: Target device. If None, uses first available device.

    Returns:
        PyTree with arrays on the specified device.
    """
    placement = DevicePlacement(device)
    return placement.place_on_device(data, device)


def distribute_batch(data: PyTree, sharding: Sharding) -> PyTree:
    """Convenience function for distributing data across devices.

    Args:
        data: PyTree of JAX arrays.
        sharding: JAX Sharding specification.

    Returns:
        PyTree with arrays distributed according to sharding.
    """
    return jax.device_put(data, sharding)


def get_batch_size_recommendation(
    hardware_type: HardwareType | None = None,
) -> BatchSizeRecommendation:
    """Get batch size recommendation for current or specified hardware.

    Args:
        hardware_type: Hardware type to get recommendation for.

    Returns:
        BatchSizeRecommendation with hardware-specific values.
    """
    placement = DevicePlacement()
    return placement.get_batch_size_recommendation(hardware_type)
