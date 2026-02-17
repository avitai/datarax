"""Data parallelism utilities for Datarax.

This module provides functions for data-parallel training in JAX models,
supporting both modern SPMD (via nnx.jit + mesh) and legacy pmap patterns.
"""

from typing import Any
from collections.abc import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.sharding import Mesh, PartitionSpec, Sharding

from datarax.typing import Batch


def create_data_parallel_sharding(mesh: Mesh, data_axis: str = "data") -> Sharding:
    """Create a Sharding object for data-parallel training.

    Args:
        mesh: The device mesh to use for sharding.
        data_axis: The name of the mesh axis to use for data parallelism.

    Returns:
        A JAX Sharding object for data-parallel training.
    """
    return jax.sharding.NamedSharding(mesh, PartitionSpec(data_axis))


def shard_batch(batch: Batch, sharding: Sharding) -> Batch:
    """Shard a batch of data across devices.

    Args:
        batch: The batch to shard.
        sharding: The sharding specification to use.

    Returns:
        The sharded batch.
    """

    def maybe_shard(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return jax.device_put(x, sharding)
        return x

    return jax.tree.map(maybe_shard, batch)


def spmd_train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    loss_fn: Callable[[nnx.Module, Batch], jax.Array],
    batch: Batch,
) -> jax.Array:
    """Execute a data-parallel training step using SPMD.

    Uses nnx.value_and_grad for automatic differentiation. The XLA compiler
    handles gradient AllReduce automatically when model parameters are
    sharded across devices via jax.set_mesh or explicit NamedSharding.

    This function should be called inside an @nnx.jit decorated function
    with a mesh context active.

    Args:
        model: The NNX model to train.
        optimizer: The NNX optimizer wrapping the model.
        loss_fn: Function (model, batch) -> loss scalar.
        batch: The training batch (should be pre-sharded).

    Returns:
        The loss value.

    Example:
        mesh = jax.make_mesh((4,), ("data",))
        rules = data_parallel_rules()

        @nnx.jit
        def train_step(model, optimizer, batch):
            return spmd_train_step(model, optimizer, my_loss_fn, batch)

        with jax.set_mesh(mesh):
            loss = train_step(model, optimizer, batch)
    """
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss


def data_parallel_train_step(
    loss_fn: Callable,
    optimizer: Any,
    batch: Batch,
    state: Any,
    rngs: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Execute a data-parallel training step using pmap.

    Uses jax.pmap with lax.pmean for gradient averaging.
    For modern SPMD training, prefer spmd_train_step instead.

    Args:
        loss_fn: Function that computes the loss value.
        optimizer: The optimizer to use.
        batch: The batch to process.
        state: The current training state.
        rngs: Optional random number generator keys.

    Returns:
        A tuple of (new_state, metrics).
    """

    def _train_step(state: Any, batch: Batch) -> tuple[Any, dict[str, Any]]:
        def loss_fn_wrapper(params: Any) -> tuple[Any, Any]:
            variables = {"params": params, **state.model_state}
            loss, grads = jax.value_and_grad(loss_fn)(variables, batch, rngs=rngs)
            return loss, grads

        loss, grads = loss_fn_wrapper(state.params)
        grads = lax.pmean(grads, axis_name="batch")

        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)

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

    parallel_train_step = jax.pmap(_train_step, axis_name="batch")
    return parallel_train_step(state, batch)


def shard_model_state(
    state: Any,
    mesh: Mesh,
    param_sharding: str | dict[str, PartitionSpec] | None = None,
) -> Any:
    """Shard a model's state across devices.

    Args:
        state: The model state to shard.
        mesh: The device mesh to shard across.
        param_sharding: Optional parameter sharding specifications.
            Use "replicate" to replicate all parameters, or a dict mapping
            parameter paths to PartitionSpec for per-parameter sharding.

    Returns:
        The sharded model state.
    """
    if isinstance(param_sharding, str) and param_sharding == "replicate":
        sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
        return jax.device_put(state, sharding)

    if isinstance(param_sharding, dict):
        from flax.traverse_util import flatten_dict, unflatten_dict

        def apply_sharding(path: str, value: Any) -> Any:
            if path in param_sharding:
                named_sharding = jax.sharding.NamedSharding(mesh, param_sharding[path])
                return jax.device_put(value, named_sharding)
            return value

        flat_params = flatten_dict(state)
        sharded_flat_params = {k: apply_sharding("/".join(k), v) for k, v in flat_params.items()}
        return unflatten_dict(sharded_flat_params)

    # Default: replicate all parameters
    sharding = jax.sharding.NamedSharding(mesh, PartitionSpec())
    return jax.device_put(state, sharding)


def all_reduce_gradients(
    gradients: Any,
    reduce_type: str = "mean",
    axis_name: str = "batch",
) -> Any:
    """All-reduce gradients across devices using collective operations.

    Only valid inside a pmap or shard_map context.
    For SPMD training with nnx.jit, gradient reduction is handled
    automatically by the compiler.

    Args:
        gradients: The gradients to reduce.
        reduce_type: The type of reduction ("mean" or "sum").
        axis_name: The axis name for the collective operation.

    Returns:
        The reduced gradients.

    Raises:
        ValueError: If reduce_type is not "mean" or "sum".
    """
    if reduce_type.lower() == "mean":
        return lax.pmean(gradients, axis_name=axis_name)
    if reduce_type.lower() == "sum":
        return lax.psum(gradients, axis_name=axis_name)
    raise ValueError(f"Unsupported reduce_type: {reduce_type}")


def reduce_gradients(gradients: Any, reduce_type: str = "mean") -> Any:
    """Reduce gradients using standard JAX operations on global arrays.

    Works in SPMD contexts (inside nnx.jit with mesh). The XLA compiler
    handles cross-device communication automatically.

    Args:
        gradients: The gradients to reduce (global sharded arrays).
        reduce_type: The type of reduction ("mean" or "sum").

    Returns:
        The reduced gradients.

    Raises:
        ValueError: If reduce_type is not "mean" or "sum".
    """
    if reduce_type.lower() == "mean":
        return jax.tree.map(lambda g: jnp.mean(g), gradients)
    if reduce_type.lower() == "sum":
        return jax.tree.map(lambda g: jnp.sum(g), gradients)
    raise ValueError(f"Unsupported reduce_type: {reduce_type}")
