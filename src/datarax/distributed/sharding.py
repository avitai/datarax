"""NNX SPMD sharding utilities for Datarax.

This module provides utilities for configuring sharding strategies
in JAX SPMD training, following upstream Flax NNX patterns.

The core abstraction is MeshRules, which maps logical dimension names
(data, embed, mlp, heads) to physical mesh axis names. Factory functions
create standard configurations for data-parallel and FSDP training.

Example:
    mesh = jax.make_mesh((4, 2), ("data", "model"))
    rules = fsdp_rules(data_axis="data", model_axis="model")
    spec = apply_sharding_rules(rules, "data", "embed")
    # => PartitionSpec("data", "model")
"""

from dataclasses import dataclass

from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class MeshRules:
    """Logical-to-physical mesh axis mapping for SPMD sharding.

    Maps logical dimension names to physical mesh axis names.
    Following the pattern from Flax NNX FSDP examples.

    Attributes:
        data: Mesh axis for the batch/data dimension.
        embed: Mesh axis for embedding dimensions.
        mlp: Mesh axis for MLP hidden dimensions.
        heads: Mesh axis for attention head dimensions.
    """

    data: str | None = None
    embed: str | None = None
    mlp: str | None = None
    heads: str | None = None

    def __call__(self, *keys: str) -> tuple[str | None, ...]:
        """Look up mesh axes for the given logical dimension names.

        Args:
            *keys: Logical dimension names (must be attributes of MeshRules).

        Returns:
            Tuple of mesh axis names (or None for unmapped dimensions).
        """
        return tuple(getattr(self, key) for key in keys)


def data_parallel_rules(data_axis: str = "data") -> MeshRules:
    """Create MeshRules for pure data-parallel training.

    Shards only the batch dimension across devices.

    Args:
        data_axis: Name of the mesh axis for data parallelism.

    Returns:
        MeshRules with only the data axis mapped.
    """
    return MeshRules(data=data_axis)


def fsdp_rules(
    data_axis: str = "data",
    model_axis: str = "model",
) -> MeshRules:
    """Create MeshRules for Fully Sharded Data Parallel (FSDP) training.

    Shards the batch dimension across the data axis and model weight
    dimensions across the model axis.

    Args:
        data_axis: Name of the mesh axis for data parallelism.
        model_axis: Name of the mesh axis for model parallelism.

    Returns:
        MeshRules with data and model axes mapped.
    """
    return MeshRules(
        data=data_axis,
        embed=model_axis,
        mlp=model_axis,
        heads=model_axis,
    )


def create_named_sharding(
    mesh: Mesh,
    *axis_names: str | None,
) -> NamedSharding:
    """Create a NamedSharding from a mesh and axis names.

    Args:
        mesh: The device mesh.
        *axis_names: Axis names for each dimension. Use None for replicated
            dimensions. If no axes are provided, creates a fully replicated
            sharding.

    Returns:
        A JAX NamedSharding object.
    """
    return NamedSharding(mesh, PartitionSpec(*axis_names))


def apply_sharding_rules(
    rules: MeshRules,
    *logical_names: str,
) -> PartitionSpec:
    """Apply MeshRules to create a PartitionSpec for a tensor.

    Maps logical dimension names through the rules to produce physical
    mesh axis assignments.

    Args:
        rules: The MeshRules mapping to apply.
        *logical_names: Logical dimension names for each tensor axis.

    Returns:
        A PartitionSpec with physical mesh axis assignments.
    """
    axes = rules(*logical_names)
    return PartitionSpec(*axes)
