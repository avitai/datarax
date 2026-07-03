"""Differentiable in-DAG rebatching node.

Rebatching regroups the batch (leading) axis of a pipeline batch. This module
provides the JAX-native, differentiable, *within-step* form: it reshapes each
array leaf from ``(N, *rest)`` to ``(N // group_size, group_size, *rest)`` with a
pure reshape, so gradients flow and every shape stays static — safe inside the
jitted ``Pipeline.step`` and under ``nnx.scan``. It restores the "differentiable
rebatching" capability datarax advertises.

For non-differentiable, cross-step rebatching to a different loader batch size,
delegate to ``grain.experimental.RebatchIterDataset`` at the source/iteration
level instead — that is the correct layer for stateful buffering across
iterations, and mixing it into a jitted DAG node would break tracing.
"""

from __future__ import annotations

from typing import Any

import jax
from flax import nnx


class RebatchNode(nnx.Module):
    """DAG node that differentiably regroups the batch axis by ``group_size``.

    Each leaf ``x`` of shape ``(N, *rest)`` is reshaped to
    ``(N // group_size, group_size, *rest)``, grouping ``group_size`` consecutive
    records together. The transform is a pure reshape: it preserves gradients and
    uses only static shapes, so it is safe inside ``Pipeline.step`` (jitted) and
    ``Pipeline.scan``. Consumes the raw batch dict the Pipeline threads between
    nodes.
    """

    def __init__(self, group_size: int) -> None:
        """Store the regrouping factor.

        Args:
            group_size: Number of consecutive records to merge into each group;
                must be >= 1 and evenly divide the incoming batch size.

        Raises:
            ValueError: If ``group_size`` is less than 1.
        """
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self._group_size = group_size

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Regroup every array leaf's leading axis by ``group_size``.

        Args:
            batch: Batch dict of arrays sharing a leading (record) axis.

        Returns:
            Batch dict whose arrays have shape ``(N // group_size, group_size, *rest)``.

        Raises:
            ValueError: If the batch size is not divisible by ``group_size``.
        """
        leaves = jax.tree.leaves(batch)
        if leaves and leaves[0].shape[0] % self._group_size != 0:
            raise ValueError(
                f"batch size {leaves[0].shape[0]} is not divisible by group_size {self._group_size}"
            )
        return jax.tree.map(self._regroup, batch)

    def _regroup(self, x: jax.Array) -> jax.Array:
        """Reshape ``x`` from ``(N, *rest)`` to ``(N // group_size, group_size, *rest)``."""
        new_shape = (x.shape[0] // self._group_size, self._group_size, *x.shape[1:])
        return x.reshape(new_shape)
