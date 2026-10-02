"""A pipeline's stage graph as one module, and the raw ``Batch`` it runs over.

:class:`OperatorDag` is the part of a pipeline that transforms data: its stages and the static
plan :func:`~datarax.pipeline.topo.topological_sort` builds, and nothing else (no source, no
position, no ``Rngs``). It maps a ``Batch`` to a ``Batch``, so it runs inside a differentiated
train step over a raw batch whose records carry their identities, and its operators' parameters
train with the model. The pipeline applies it to every batch it serves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from datarax.core.element_batch import Batch
from datarax.pipeline.topo import topological_sort, validate_dag


class Records(NamedTuple):
    """The records a batch holds: each row's stable index and the epoch it belongs to.

    ``indices`` is uint32 ``(B, 2)``, each 64-bit index as its words ``(hi, lo)``. A batch
    crossing an epoch boundary holds records of two epochs, so the epoch is per row.
    """

    indices: jax.Array
    epochs: jax.Array


def name_records(batch: Batch, indices: ArrayLike, epochs: ArrayLike) -> Batch:
    """``batch`` with its rows named as records ``indices`` of ``epochs``.

    ``epochs`` is one epoch for the batch or one per row.

    Args:
        batch: The batch, from ``batch_ops.from_arrays`` over gathered or streamed values.
        indices: uint32 ``(B, 2)``, each row's 64-bit record index as its words ``(hi, lo)``.
        epochs: The epoch of the batch, or of each row ``(B,)``.

    Returns:
        The batch with ``indices`` and ``epochs`` set and ``draws`` 0.

    Raises:
        ValueError: If ``indices`` is not uint32 ``(B, 2)``.
    """
    words = jnp.asarray(indices)
    if words.dtype != jnp.uint32 or words.shape != (batch.batch_size, 2):
        raise ValueError(
            f"record indices are uint32 ({batch.batch_size}, 2) words (hi, lo); got "
            f"{words.dtype} {words.shape}"
        )
    return batch.replace(
        indices=words,
        epochs=jnp.broadcast_to(jnp.asarray(epochs, jnp.int32), (batch.batch_size,)),
        draws=jnp.zeros((batch.batch_size,), jnp.int32),
    )


class OperatorDag(nnx.Module):
    """A graph of stages from ``Batch`` to ``Batch``.

    Each node's ``__call__`` receives the input batch when it has no predecessors, otherwise its
    predecessors' outputs as positional arguments, in topological order, and returns a ``Batch``.
    An operator is a node; so is any ``nnx.Module`` taking and returning batches (field
    selection, a merge of branches). The plan (order, predecessors, sink) is static and
    hashable, so identically built DAGs share one compiled program, and tracing unrolls it into
    one graph.

    Attributes:
        stages: The nodes by name, graph children.
        order: Node names in topological order.
        predecessors: Each node's predecessor names, in ``order``.
        sink: The node whose output is the DAG's, or ``None`` for a DAG without stages.
    """

    stages: nnx.Dict
    order: tuple[str, ...]
    predecessors: tuple[tuple[str, ...], ...]
    sink: str | None

    def __init__(
        self,
        nodes: Mapping[str, nnx.Module],
        edges: Mapping[str, Sequence[str]],
        sink: str | None,
    ) -> None:
        """Validate and sort the graph.

        Args:
            nodes: The nodes by name.
            edges: Each node's predecessor names; an empty list reads the input batch.
            sink: The node whose output is returned, or ``None`` when there are no nodes.
        """
        order: tuple[str, ...] = ()
        if sink is not None:
            validate_dag(nodes, edges, sink)
            order = tuple(topological_sort(edges))
        self.stages = nnx.Dict(dict(nodes))
        self.order = order
        self.predecessors = tuple(tuple(edges[name]) for name in order)
        self.sink = sink

    @classmethod
    def from_stages(cls, stages: Sequence[nnx.Module]) -> OperatorDag:
        """A linear DAG: stage ``i`` reads stage ``i - 1``'s output, the first the input.

        Args:
            stages: The stages in order.

        Returns:
            The DAG; its nodes are named ``stage_0``, ``stage_1``, ...
        """
        names = [f"stage_{i}" for i in range(len(stages))]
        return cls(
            nodes=dict(zip(names, stages, strict=True)),
            edges={name: names[i - 1 : i] for i, name in enumerate(names)},
            sink=names[-1] if names else None,
        )

    def __call__(self, batch: Batch) -> Batch:
        """Run every node over ``batch`` and return the sink's output.

        Args:
            batch: The input batch.

        Returns:
            The sink node's output, or ``batch`` for a DAG without stages.

        Raises:
            TypeError: If a node returns anything but a ``Batch``.
        """
        if self.sink is None:
            return batch
        outputs: dict[str, Batch] = {}
        for name, predecessors in zip(self.order, self.predecessors, strict=True):
            inputs = tuple(outputs[p] for p in predecessors) if predecessors else (batch,)
            output = self.stages[name](*inputs)
            if not isinstance(output, Batch):
                raise TypeError(
                    f"DAG node {name!r} ({type(self.stages[name]).__name__}) returned "
                    f"{type(output).__name__}; a node returns a Batch, e.g. "
                    "batch.replace(data={**batch.data, ...})"
                )
            outputs[name] = output
        return outputs[self.sink]


__all__ = ["OperatorDag", "Records", "name_records"]
