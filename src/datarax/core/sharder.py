"""Base implementation for sharder modules in Datarax.

This module provides the SharderModule base class, which is the foundation for
all NNX-based sharder implementations in Datarax. Sharders are responsible for
distributing batches of data across JAX devices.
"""

from dataclasses import dataclass
from typing import Any, Callable, Union

import flax.nnx as nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding

from datarax.core.config import DataraxModuleConfig
from datarax.core.module import DataraxModule
from datarax.typing import Batch


# Type aliases for sharding specifications
AxisName = str
LogicalAxisSpec = tuple[AxisName | None, ...]
ShardingRules = list[tuple[AxisName, AxisName | None]]


@dataclass
class SharderModuleConfig(DataraxModuleConfig):
    """Configuration for SharderModule.

    Attributes:
        sharding_rules: Optional mapping from logical axis names to physical
            device mesh axis names. If provided, logical axis names can be
            used in sharding specifications.
    """

    sharding_rules: ShardingRules | None = None


class SharderModule(DataraxModule):
    """Base class for NNX-based sharder modules in Datarax.

    SharderModule provides a foundation for implementing data sharders that can
    be integrated with NNX-based components. It handles the distribution of
    data batches across JAX devices based on a specified sharding configuration.
    """

    def __init__(
        self,
        config: SharderModuleConfig | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the SharderModule.

        Args:
            config: Optional configuration for the sharder.
            rngs: Optional Rngs for random operations (typically not needed for
                sharders).
            name: Optional name for the module.
        """
        if config is None:
            config = SharderModuleConfig()
        super().__init__(config, rngs=rngs, name=name)

    def get_partition_spec(self, logical_spec: LogicalAxisSpec | PartitionSpec) -> PartitionSpec:
        """Convert a logical axis specification to a physical PartitionSpec.

        Args:
            logical_spec: A tuple of logical axis names or a PartitionSpec.

        Returns:
            A PartitionSpec using physical device mesh axis names.
        """
        sharding_rules = self.config.sharding_rules
        if not sharding_rules or isinstance(logical_spec, PartitionSpec):
            return (
                logical_spec
                if isinstance(logical_spec, PartitionSpec)
                else PartitionSpec(*logical_spec)
            )

        # Map logical axis names to physical device mesh axis names
        physical_spec: list[AxisName | None] = []
        for dim in logical_spec:
            if dim is None:
                physical_spec.append(None)
            else:
                mapped = next((phys for log, phys in sharding_rules if log == dim), dim)
                physical_spec.append(mapped)

        return PartitionSpec(*physical_spec)

    def get_named_sharding(
        self, mesh: Mesh, logical_spec: LogicalAxisSpec | PartitionSpec
    ) -> NamedSharding:
        """Create a NamedSharding from a logical axis specification and mesh.

        Args:
            mesh: The device mesh to use for sharding.
            logical_spec: A tuple of logical axis names or a PartitionSpec.

        Returns:
            A NamedSharding using the provided mesh and converted partition spec.
        """
        pspec = self.get_partition_spec(logical_spec)
        return NamedSharding(mesh, pspec)

    def shard(self, batch: Batch, sharding: Sharding) -> Batch:
        """Distribute a batch of data across JAX devices.

        This method should be implemented by subclasses to perform the specific
        sharding logic for different types of data.

        Args:
            batch: A batch of data elements (PyTree).
            sharding: A JAX Sharding object specifying how to distribute
                arrays.

        Returns:
            A sharded batch (PyTree of jax.Array objects with the specified
            sharding).
        """
        raise NotImplementedError("Subclasses must implement this method")

    def parallel_transform(
        self,
        batch: Batch,
        transform_fn: Callable[[Batch], Batch],
        mesh: Mesh,
        in_spec: Union[LogicalAxisSpec, PartitionSpec],
        out_spec: LogicalAxisSpec | PartitionSpec | None = None,
    ) -> Batch:
        """Apply a transformation to each shard of the batch in parallel.

        This uses nnx.shard_map to efficiently process data in parallel
        across multiple devices.

        Args:
            batch: The batch of data to transform.
            transform_fn: A function that takes a batch as input and returns
                a transformed batch.
            mesh: The device mesh to use for sharding.
            in_spec: The input sharding specification.
            out_spec: Optional output sharding specification. If not provided,
                the input spec is used.

        Returns:
            The transformed batch with the specified sharding.
        """
        in_pspec = self.get_partition_spec(in_spec)
        out_pspec = self.get_partition_spec(out_spec) if out_spec else in_pspec

        with mesh:
            return nnx.shard_map(
                transform_fn, mesh=mesh, in_specs=(in_pspec,), out_specs=out_pspec
            )(batch)

    def get_state(self) -> dict[str, Any]:
        """Get the current state of the SharderModule for checkpointing.

        Returns:
            A dictionary containing the internal state of the SharderModule.
        """
        state = super().get_state()
        # Add sharding rules to the state if present
        if self.config.sharding_rules:
            state["sharding_rules"] = self.config.sharding_rules
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint.

        Args:
            state: A dictionary containing the internal state to restore.
        """
        # Extract sharding rules if present (note: we update the config)
        if "sharding_rules" in state:
            # Use object.__setattr__ since config is a frozen dataclass
            object.__setattr__(self.config, "sharding_rules", state.pop("sharding_rules"))

        # Restore the rest of the state
        super().set_state(state)
