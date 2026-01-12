"""Data parallelism utilities for Datarax.

This module provides a unified implementation for data-parallel training
in JAX models, with support for multi-device and multi-host configurations.
Supports both static method usage and NNX module instantiation.
"""

from typing import Any, Callable

import flax.nnx as nnx
import jax
import optax
from jax import lax
from jax.sharding import Mesh, PartitionSpec, Sharding

from datarax.typing import Batch


class DataParallel(nnx.Module):
    """Unified data-parallel training utilities for Datarax.

    This class provides methods for implementing data parallelism in JAX,
    including functions for sharding data and model parameters across devices,
    and for aggregating gradients during optimization.

    Supports both static method usage (stateless) and instance method usage (stateful).

    Usage:
        # Static method usage (stateless)
        sharding = DataParallel.create_data_parallel_sharding(mesh)

        # Instance method usage (stateful)
        dp = DataParallel()
        sharding = dp.create_data_parallel_sharding(mesh)
    """

    def __init__(self):
        """Initialize DataParallel module."""
        super().__init__()

    def create_data_parallel_sharding(self, mesh: Mesh, data_axis: str = "data") -> Sharding:
        """Create a Sharding object for data-parallel training.

        Args:
            mesh: The device mesh to use for sharding.
            data_axis: The name of the mesh axis to use for data parallelism.

        Returns:
            A JAX Sharding object for data-parallel training.
        """
        return jax.sharding.NamedSharding(mesh, PartitionSpec(data_axis))

    @staticmethod
    def create_data_parallel_sharding_static(mesh: Mesh, data_axis: str = "data") -> Sharding:
        """Static version of create_data_parallel_sharding."""
        return jax.sharding.NamedSharding(mesh, PartitionSpec(data_axis))

    def shard_batch(self, batch: Batch, sharding: Sharding) -> Batch:
        """Shard a batch of data across devices.

        Args:
            batch: The batch to shard.
            sharding: The sharding specification to use.

        Returns:
            The sharded batch.
        """

        def maybe_shard(x):
            if isinstance(x, jax.Array | jax.Array):
                return jax.device_put(x, sharding)
            return x

        return jax.tree.map(maybe_shard, batch)

    @staticmethod
    def shard_batch_static(batch: Batch, sharding: Sharding) -> Batch:
        """Static version of shard_batch."""
        return jax.tree.map(
            lambda x: jax.device_put(x, sharding) if isinstance(x, jax.Array | jax.Array) else x,
            batch,
        )

    def data_parallel_train_step(
        self,
        state_fn: Callable,
        loss_fn: Callable,
        optimizer: Any,
        batch: Batch,
        state: Any,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Execute a data-parallel training step.

        Args:
            state_fn: Function that returns the training state.
            loss_fn: Function that computes the loss value.
            optimizer: The optimizer to use.
            batch: The batch to process.
            state: The current training state.
            rngs: Optional random number generator keys.

        Returns:
            A tuple of (new_state, metrics).
        """

        def _train_step(state, batch):
            def loss_fn_wrapper(params):
                variables = {"params": params, **state.model_state}
                loss, grads = jax.value_and_grad(loss_fn)(variables, batch, rngs=rngs)
                return loss, grads

            # Compute loss and gradients
            loss, grads = loss_fn_wrapper(state.params)

            # Average gradients across devices
            grads = lax.pmean(grads, axis_name="batch")

            # Apply updates
            updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)

            # Create new state
            new_state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
            )

            metrics = {
                "loss": loss,
                "learning_rate": optimizer.learning_rate(state.step),
            }

            return new_state, metrics

        # Use pmap to run in parallel
        parallel_train_step = jax.pmap(_train_step, axis_name="batch")
        return parallel_train_step(state, batch)

    @staticmethod
    def data_parallel_train_step_static(
        state_fn: Callable,
        loss_fn: Callable,
        optimizer: Any,
        batch: Batch,
        state: Any,
        rngs: nnx.Rngs | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Static version of data_parallel_train_step."""

        def _train_step(state, batch):
            def loss_fn_wrapper(params):
                variables = {"params": params, **state.model_state}
                loss, grads = jax.value_and_grad(loss_fn)(variables, batch, rngs=rngs)
                return loss, grads

            # Compute loss and gradients
            loss, grads = loss_fn_wrapper(state.params)

            # Average gradients across devices
            grads = lax.pmean(grads, axis_name="batch")

            # Apply updates
            updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)

            # Create new state
            new_state = state.replace(
                step=state.step + 1,
                params=new_params,
                opt_state=new_opt_state,
            )

            metrics = {
                "loss": loss,
                "learning_rate": optimizer.learning_rate(state.step),
            }

            return new_state, metrics

        # Use pmap to run in parallel
        parallel_train_step = jax.pmap(_train_step, axis_name="batch")
        return parallel_train_step(state, batch)

    def shard_model_state(
        self,
        state: Any,
        mesh: Mesh,
        param_sharding: str | dict[str, PartitionSpec] | None = None,
    ) -> Any:
        """Shard a model's state across devices.

        Args:
            state: The model state to shard.
            mesh: The device mesh to shard across.
            param_sharding: Optional parameter sharding specifications.

        Returns:
            The sharded model state.
        """
        # If param_sharding is a string, it's a replication mode
        if isinstance(param_sharding, str) and param_sharding == "replicate":
            sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
            return jax.device_put(state, sharding)

        # If param_sharding is a dict, apply specific sharding per parameter
        if isinstance(param_sharding, dict):
            # Create a function to recursively apply sharding specs
            def apply_sharding(path, value):
                if path in param_sharding:
                    spec = param_sharding[path]
                    sharding = jax.sharding.NamedSharding(mesh, spec)
                    return jax.device_put(value, sharding)
                return value

            # Use flax FlattenState to apply sharding to nested params
            from flax.traverse_util import flatten_dict, unflatten_dict

            flat_params = flatten_dict(state)
            sharded_flat_params = {
                k: apply_sharding("/".join(k), v) for k, v in flat_params.items()
            }
            return unflatten_dict(sharded_flat_params)

        # Default: replicate all parameters
        sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
        return jax.device_put(state, sharding)

    @staticmethod
    def shard_model_state_static(
        state: Any,
        mesh: Mesh,
        param_sharding: str | dict[str, PartitionSpec] | None = None,
    ) -> Any:
        """Static version of shard_model_state."""
        # If param_sharding is a string, it's a replication mode
        if isinstance(param_sharding, str) and param_sharding == "replicate":
            sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
            return jax.device_put(state, sharding)

        # If param_sharding is a dict, apply specific sharding per parameter
        if isinstance(param_sharding, dict):
            # Create a function to recursively apply sharding specs
            def apply_sharding(path, value):
                if path in param_sharding:
                    sharding = jax.sharding.NamedSharding(mesh, param_sharding[path])
                    return jax.device_put(value, sharding)
                return value

            # Use flax FlattenState to apply sharding to nested params
            from flax.traverse_util import flatten_dict, unflatten_dict

            flat_params = flatten_dict(state)
            sharded_flat_params = {
                k: apply_sharding("/".join(k), v) for k, v in flat_params.items()
            }
            return unflatten_dict(sharded_flat_params)

        # Default: replicate all parameters
        sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
        return jax.device_put(state, sharding)

    def all_reduce_gradients(self, gradients: Any, reduce_type: str = "mean") -> Any:
        """All-reduce gradients across devices.

        Args:
            gradients: The gradients to reduce.
            reduce_type: The type of reduction to apply ("mean" or "sum").

        Returns:
            The reduced gradients.
        """
        if reduce_type.lower() == "mean":
            return lax.pmean(gradients, axis_name="batch")
        elif reduce_type.lower() == "sum":
            return lax.psum(gradients, axis_name="batch")
        else:
            raise ValueError(f"Unsupported reduce_type: {reduce_type}")

    @staticmethod
    def all_reduce_gradients_static(gradients: Any, reduce_type: str = "mean") -> Any:
        """Static version of all_reduce_gradients."""
        if reduce_type.lower() == "mean":
            return lax.pmean(gradients, axis_name="batch")
        elif reduce_type.lower() == "sum":
            return lax.psum(gradients, axis_name="batch")
        else:
            raise ValueError(f"Unsupported reduce_type: {reduce_type}")
