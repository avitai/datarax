"""Topological sort and DAG validation for ``Pipeline``.

Used by ``Pipeline.from_dag`` to compile a user-specified DAG of stages
into a static execution plan. The plan is a list of node names in
topological order; ``Pipeline.__call__`` iterates this list at trace
time, so the DAG never appears as runtime tree-walking.

Public API:

- :func:`topological_sort` — Kahn's algorithm; raises on cycles.
- :func:`validate_dag` — structural checks (unknown nodes, unknown
  predecessors, missing sink) before topo-sort.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def validate_dag(
    nodes: Mapping[str, Any],
    edges: Mapping[str, Sequence[str]],
    sink: str,
) -> None:
    """Check that ``edges`` and ``sink`` reference known nodes.

    Args:
        nodes: Map from node name to node object.
        edges: Map from node name to its list of predecessor node names.
            Empty list means "consumes the source batch directly."
        sink: Name of the node whose output is the pipeline's output.

    Raises:
        ValueError: If ``sink`` is not in ``nodes``, or if any name in
            ``edges`` (key or value) references an unknown node.
    """
    if sink not in nodes:
        raise ValueError(f"sink {sink!r} is not in nodes (known: {sorted(nodes)!r})")

    for node_name, preds in edges.items():
        if node_name not in nodes:
            raise ValueError(
                f"edges key {node_name!r} is unknown; not in nodes (known: {sorted(nodes)!r})"
            )
        for predecessor in preds:
            if predecessor not in nodes:
                raise ValueError(
                    f"edges[{node_name!r}] references unknown predecessor "
                    f"{predecessor!r} (known: {sorted(nodes)!r})"
                )


def _build_successors(edges: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Invert ``edges`` (predecessor lists) into a successor adjacency map.

    Args:
        edges: Map from each node name to the list of predecessor names.

    Returns:
        Map from each node name to the list of nodes that depend on it.
    """
    successors: dict[str, list[str]] = {name: [] for name in edges}
    for name, preds in edges.items():
        for predecessor in preds:
            if predecessor in successors:
                successors[predecessor].append(name)
    return successors


def _kahn_order(in_degree: dict[str, int], successors: Mapping[str, list[str]]) -> list[str]:
    """Drain zero-in-degree nodes in Kahn's algorithm, mutating ``in_degree``.

    Ready nodes are processed in sorted order so the result is deterministic for
    DAGs with parallel branches.

    Args:
        in_degree: Map from node name to its remaining predecessor count
            (consumed/decremented in place).
        successors: Successor adjacency map from :func:`_build_successors`.

    Returns:
        Node names in one valid topological order (may be shorter than the input
        when a cycle prevents some nodes from ever becoming ready).
    """
    ready = sorted(name for name, deg in in_degree.items() if deg == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for successor in successors[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
        ready.sort()
    return order


def topological_sort(edges: Mapping[str, Sequence[str]]) -> list[str]:
    """Return node names in topological order via Kahn's algorithm.

    Args:
        edges: Map from each node name to the list of predecessor names
            it depends on. Empty list means "depends only on the source."

    Returns:
        A list of node names ordered such that every node appears after
        all its predecessors. Multiple valid orderings may exist for
        DAGs with parallel branches; this implementation returns one
        deterministic ordering.

    Raises:
        ValueError: If ``edges`` describes a cycle. The error message
            names the nodes participating in the cycle.
    """
    in_degree = {name: len(preds) for name, preds in edges.items()}
    successors = _build_successors(edges)
    order = _kahn_order(in_degree, successors)

    if len(order) != len(edges):
        unresolved = sorted(name for name in edges if name not in order)
        raise ValueError(f"DAG contains a cycle; could not resolve nodes: {unresolved!r}")

    return order
