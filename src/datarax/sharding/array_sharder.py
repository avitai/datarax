"""Array sharder implementation for Datarax.

This module provides a unified implementation of the SharderModule that handles
sharding of JAX arrays across devices. Supports both static method usage
and NNX module instantiation.
"""

from typing import Any, Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, Sharding

from datarax.core.sharder import (
    LogicalAxisSpec,
    SharderModule,
    SharderModuleConfig,
)
from datarax.typing import Batch


class ArraySharder(SharderModule):
    """Unified array sharding implementation for Datarax.

    This class provides methods for sharding JAX arrays across devices,
    with support for both static method usage and NNX module instantiation.
    Supports logical axis naming and advanced sharding operations.
    """

    def __init__(
        self,
        config: SharderModuleConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize the ArraySharder.

        Args:
            config: Optional configuration with sharding_rules mapping from
                logical axis names to physical device mesh axis names.
            rngs: Optional Rngs for random operations (typically not needed for
                sharders).
        """
        super().__init__(config, rngs=rngs)

    def shard(self, batch: Batch, sharding: Sharding) -> Batch:
        """Distribute a batch of data across JAX devices.

        Args:
            batch: A batch of data elements (PyTree).
            sharding: A JAX Sharding object specifying how to distribute arrays.

        Returns:
            A sharded batch (PyTree of jax.Array objects with the specified sharding).
        """
        return jax.tree.map(
            lambda x: self._shard_array(x, sharding),
            batch,
            is_leaf=lambda x: isinstance(x, jax.Array | jax.Array),
        )

    @staticmethod
    def shard_static(batch: Batch, sharding: Sharding) -> Batch:
        """Static method to distribute a batch of data across JAX devices.

        Args:
            batch: A batch of data elements (PyTree).
            sharding: A JAX Sharding object specifying how to distribute arrays.

        Returns:
            A sharded batch (PyTree of jax.Array objects with the specified sharding).
        """
        return jax.tree.map(
            lambda x: ArraySharder._shard_array_static(x, sharding),
            batch,
            is_leaf=lambda x: isinstance(x, jax.Array | jax.Array),
        )

    def _shard_array(self, array: Any, sharding: Sharding) -> jax.Array:
        """Shard a single array.

        Args:
            array: The array to shard.
            sharding: A JAX Sharding object specifying how to distribute the array.

        Returns:
            A sharded jax.Array.
        """
        return self._shard_array_static(array, sharding)

    @staticmethod
    def _shard_array_static(array: Any, sharding: Sharding) -> jax.Array:
        """Static method to shard a single array.

        Args:
            array: The array to shard.
            sharding: A JAX Sharding object specifying how to distribute the array.

        Returns:
            A sharded jax.Array.
        """
        # If the array is already a jax.Array with the correct sharding, return it
        if hasattr(array, "sharding") and array.sharding == sharding:
            return array

        # Convert to a JAX array if it's not already
        if not isinstance(array, jax.Array):
            array = jnp.asarray(array)

        # Use device_put to move the array to the correct devices with the specified sharding
        return jax.device_put(array, sharding)

    def shard_with_info(
        self, batch: Batch, sharding: Sharding, info: dict[str, Any] | None = None
    ) -> Batch:
        """Return sharded batch with debug information.

        Args:
            batch: A batch of data elements (PyTree).
            sharding: A JAX Sharding object specifying how to distribute arrays.
            info: Optional dictionary to update with sharding info.

        Returns:
            A sharded batch (PyTree of jax.Array objects with the specified sharding).
        """
        return self.shard_with_info_static(batch, sharding, info)

    @staticmethod
    def shard_with_info_static(
        batch: Batch, sharding: Sharding, info: dict[str, Any] | None = None
    ) -> Batch:
        """Static method to return sharded batch with debug information.

        Args:
            batch: A batch of data elements (PyTree).
            sharding: A JAX Sharding object specifying how to distribute arrays.
            info: Optional dictionary to update with sharding info.

        Returns:
            A sharded batch (PyTree of jax.Array objects with the specified sharding).
        """
        # Initialize info dictionary if not provided
        if info is None:
            info = {}

        # Collect information about the batch before sharding
        info["batch_pre_sharding"] = {
            "structure": jax.tree_util.tree_structure(batch),
            "shapes": jax.tree.map(
                lambda x: getattr(x, "shape", None)
                if isinstance(x, jax.Array | jax.Array)
                else None,
                batch,
            ),
        }

        # Shard the batch
        sharded_batch = ArraySharder.shard_static(batch, sharding)

        # Collect information about the sharded batch
        info["batch_post_sharding"] = {
            "structure": jax.tree_util.tree_structure(sharded_batch),
            "shapes": jax.tree.map(
                lambda x: getattr(x, "shape", None)
                if isinstance(x, jax.Array | jax.Array)
                else None,
                sharded_batch,
            ),
            "shardings": jax.tree.map(
                lambda x: getattr(x, "sharding", None) if isinstance(x, jax.Array) else None,
                sharded_batch,
            ),
        }

        return sharded_batch

    def shard_with_logical_names(
        self,
        batch: Batch,
        mesh: Mesh,
        logical_spec: LogicalAxisSpec | PartitionSpec,
    ) -> Batch:
        """Shard a batch using logical axis names.

        This method allows using more descriptive logical axis names instead
        of device mesh axis names directly.

        Args:
            batch: A batch of data elements (PyTree).
            mesh: The device mesh to use for sharding.
            logical_spec: A tuple of logical axis names or a PartitionSpec.

        Returns:
            A sharded batch (PyTree of jax.Array objects).
        """
        named_sharding = self.get_named_sharding(mesh, logical_spec)
        return self.shard(batch, named_sharding)

    def apply_parallel_transform(
        self,
        batch: Batch,
        transform_fn: Callable[[Batch], Batch],
        mesh: Mesh,
        in_spec: LogicalAxisSpec | PartitionSpec,
        out_spec: LogicalAxisSpec | PartitionSpec | None = None,
    ) -> Batch:
        """Apply a transformation to the batch in parallel across devices.

        This is particularly useful for operations that can be performed
        independently on each shard of the data.

        Args:
            batch: The batch to transform.
            transform_fn: The function to apply to the batch.
            mesh: The device mesh to use.
            in_spec: The input sharding specification.
            out_spec: Optional output sharding specification.

        Returns:
            The transformed batch.
        """
        return self.parallel_transform(batch, transform_fn, mesh, in_spec, out_spec)

    def create_sharded_param(
        self,
        init_fn: Callable,
        shape: tuple[int, ...],
        logical_spec: LogicalAxisSpec | PartitionSpec,
    ) -> nnx.Param:
        """Create a parameter with explicit sharding annotation.

        This utility method makes it easier to create model parameters with
        appropriate sharding annotations for distributed training.

        Args:
            init_fn: The initialization function for the parameter.
            shape: The shape of the parameter.
            logical_spec: The logical sharding specification.

        Returns:
            An initialized parameter with sharding annotation.
        """
        # Convert logical spec to physical if needed
        physical_spec = self.get_partition_spec(logical_spec)

        # Convert PartitionSpec to tuple if needed and ensure correct type
        if isinstance(physical_spec, PartitionSpec):
            physical_tuple = tuple(physical_spec)
        else:
            physical_tuple = physical_spec

        if self.rngs is None:
            raise ValueError("rngs must be provided to create a sharded param.")

        rngs_params = self.rngs.params()
        return nnx.Param(
            nnx.with_partitioning(init_fn, physical_tuple)(rngs_params, shape),
        )
