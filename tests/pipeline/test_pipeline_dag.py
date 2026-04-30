"""Contracts for ``Pipeline.from_dag`` — Tier-2 declarative DAG topology.

``Pipeline.from_dag(source=..., nodes=..., edges=..., sink=...)`` builds
a pipeline whose stages execute in user-specified topological order.
Each node's ``__call__`` receives its predecessors' outputs as positional
arguments. The execution plan is compiled at construction (topological
sort + cycle / connectivity validation), so ``Pipeline.__call__`` and
``Pipeline.scan`` trace through the static plan with no runtime
tree-walking.

Test contract index:

A. Topo-sort and structural validation:

   1. ``test_from_dag_compiles_simple_linear_chain`` — a 3-node linear
      chain (filter → augment → batch) runs in source order.
   2. ``test_from_dag_compiles_branch_merge_topology`` — diamond DAG
      (A → B, A → C, (B, C) → D) executes B and C in either order but
      always before D.
   3. ``test_from_dag_rejects_cycles`` — a cyclic edges dict raises
      ``ValueError`` with a message naming the cycle.
   4. ``test_from_dag_rejects_unknown_predecessor`` — edges referencing
      a node name not in ``nodes`` raise ``ValueError``.
   5. ``test_from_dag_rejects_unknown_sink`` — sink not in ``nodes``
      raises ``ValueError``.

B. Execution semantics:

   6. ``test_from_dag_routes_source_to_root_nodes`` — nodes with empty
      predecessor lists receive the source batch directly.
   7. ``test_from_dag_threads_branch_outputs_to_merge`` — merge node
      sees branch outputs in declared edge order.
   8. ``test_from_dag_returns_only_sink_output`` — ``__call__`` returns
      the sink node's output, not the full intermediate dict.

C. Composition with the rest of Pipeline:

   9. ``test_from_dag_step_advances_position_like_linear`` — ``step()``
      and ``_position`` semantics match the linear-stages constructor.
   10. ``test_from_dag_scan_runs_under_nnx_scan`` — DAG pipelines
       integrate with ``Pipeline.scan`` end-to-end.
   11. ``test_from_dag_gradient_flows_through_both_branches`` — grads
       reach learnable params on both sides of a diamond DAG.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# ---------- Helpers ----------


def _source(num_elements: int = 16) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(num_elements, dtype=jnp.float32)},
    )


class _Double(nnx.Module):
    """Single-input stage: x → x * 2."""

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * 2.0}


class _AddOne(nnx.Module):
    """Single-input stage: x → x + 1."""

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] + 1.0}


class _AddTen(nnx.Module):
    """Single-input stage: x → x + 10."""

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] + 10.0}


class _SumMerge(nnx.Module):
    """Two-input merge: (a, b) → a["x"] + b["x"]."""

    def __call__(self, a: dict, b: dict) -> dict:
        return {"x": a["x"] + b["x"]}


class _LearnableScale(nnx.Module):
    """Holds a single nnx.Param multiplicative factor."""

    def __init__(self, init: float) -> None:
        super().__init__()
        self.factor = nnx.Param(jnp.float32(init))

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * self.factor[...]}


# ---------- A. Topo-sort and validation ----------


def test_from_dag_compiles_simple_linear_chain() -> None:
    pipeline = Pipeline.from_dag(
        source=_source(),
        nodes={"a": _Double(), "b": _AddOne()},
        edges={"a": [], "b": ["a"]},
        sink="b",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # First batch is arange(0, 4); * 2 + 1 = [1, 3, 5, 7]
    np.testing.assert_allclose(np.asarray(out["x"]), np.array([1.0, 3.0, 5.0, 7.0]))


def test_from_dag_compiles_branch_merge_topology() -> None:
    pipeline = Pipeline.from_dag(
        source=_source(),
        nodes={
            "a": _Double(),  # source → a (x*2)
            "b": _AddOne(),  # a → b   (+1)
            "c": _AddTen(),  # a → c   (+10)
            "d": _SumMerge(),  # (b, c) → d
        },
        edges={
            "a": [],
            "b": ["a"],
            "c": ["a"],
            "d": ["b", "c"],
        },
        sink="d",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # arange(0, 4) * 2 = [0, 2, 4, 6]
    # b: + 1 = [1, 3, 5, 7]; c: + 10 = [10, 12, 14, 16]
    # d: b + c = [11, 15, 19, 23]
    np.testing.assert_allclose(np.asarray(out["x"]), np.array([11.0, 15.0, 19.0, 23.0]))


def test_from_dag_rejects_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        Pipeline.from_dag(
            source=_source(),
            nodes={"a": _Double(), "b": _AddOne()},
            edges={"a": ["b"], "b": ["a"]},  # a depends on b which depends on a
            sink="b",
            batch_size=4,
            rngs=nnx.Rngs(0),
        )


def test_from_dag_rejects_unknown_predecessor() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Pipeline.from_dag(
            source=_source(),
            nodes={"a": _Double()},
            edges={"a": ["nonexistent"]},
            sink="a",
            batch_size=4,
            rngs=nnx.Rngs(0),
        )


def test_from_dag_rejects_unknown_sink() -> None:
    with pytest.raises(ValueError, match="sink"):
        Pipeline.from_dag(
            source=_source(),
            nodes={"a": _Double()},
            edges={"a": []},
            sink="nonexistent",
            batch_size=4,
            rngs=nnx.Rngs(0),
        )


# ---------- B. Execution semantics ----------


def test_from_dag_routes_source_to_root_nodes() -> None:
    """Nodes with empty predecessor lists receive the raw source batch."""
    pipeline = Pipeline.from_dag(
        source=_source(),
        nodes={"identity": _AddOne()},  # consumes source directly
        edges={"identity": []},
        sink="identity",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # arange(0,4) + 1 = [1, 2, 3, 4]
    np.testing.assert_allclose(np.asarray(out["x"]), np.array([1.0, 2.0, 3.0, 4.0]))


def test_from_dag_threads_branch_outputs_to_merge() -> None:
    """Merge node receives branch outputs in declared edge order."""

    class _OrderSensitiveMerge(nnx.Module):
        def __call__(self, first: dict, second: dict) -> dict:
            # 2 * first - second; non-commutative so order matters
            return {"x": 2.0 * first["x"] - second["x"]}

    pipeline = Pipeline.from_dag(
        source=_source(),
        nodes={
            "a": _AddOne(),  # x + 1
            "b": _AddTen(),  # x + 10
            "merge": _OrderSensitiveMerge(),
        },
        edges={
            "a": [],
            "b": [],
            "merge": ["a", "b"],  # a first, b second
        },
        sink="merge",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # a: arange(0,4) + 1 = [1,2,3,4]; b: arange(0,4) + 10 = [10,11,12,13]
    # merge: 2 * a - b = [2-10, 4-11, 6-12, 8-13] = [-8, -7, -6, -5]
    np.testing.assert_allclose(np.asarray(out["x"]), np.array([-8.0, -7.0, -6.0, -5.0]))


def test_from_dag_returns_only_sink_output() -> None:
    """__call__/step return just the sink's output, not the intermediate dict."""
    pipeline = Pipeline.from_dag(
        source=_source(),
        nodes={"a": _Double(), "b": _AddOne()},
        edges={"a": [], "b": ["a"]},
        sink="b",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    assert isinstance(out, dict)
    assert set(out.keys()) == {"x"}  # not {"a_out", "b_out"} or anything richer


# ---------- C. Composition with rest of Pipeline ----------


def test_from_dag_step_advances_position_like_linear() -> None:
    pipeline = Pipeline.from_dag(
        source=_source(num_elements=16),
        nodes={"a": _Double()},
        edges={"a": []},
        sink="a",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    assert int(pipeline._position[...]) == 0
    pipeline.step()  # type: ignore[reportCallIssue]
    assert int(pipeline._position[...]) == 4
    pipeline.step()  # type: ignore[reportCallIssue]
    assert int(pipeline._position[...]) == 8


def test_from_dag_scan_runs_under_nnx_scan() -> None:
    pipeline = Pipeline.from_dag(
        source=_source(num_elements=16),
        nodes={
            "a": _Double(),
            "b": _AddOne(),
            "c": _AddTen(),
            "d": _SumMerge(),
        },
        edges={"a": [], "b": ["a"], "c": ["a"], "d": ["b", "c"]},
        sink="d",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    outputs = pipeline.scan(step_fn, length=4)

    # Per step: arange(start,start+4) * 2 → b = +1, c = +10, d = b + c
    # = [2x+1] + [2x+10] = 4x + 11; sum over 4 elements
    # Steps 0..3: sums 4*sum(arange(start, start+4)) + 11*4
    expected_sums = []
    for start in (0, 4, 8, 12):
        x = np.arange(start, start + 4)
        expected_sums.append(float(np.sum(4 * x + 11)))
    np.testing.assert_allclose(np.asarray(outputs), np.array(expected_sums))


def test_from_dag_gradient_flows_through_both_branches() -> None:
    """grads must reach learnable params on both sides of a diamond DAG."""

    branch_a = _LearnableScale(init=2.0)
    branch_b = _LearnableScale(init=3.0)

    pipeline = Pipeline.from_dag(
        source=_source(num_elements=16),
        nodes={"a": branch_a, "b": branch_b, "merge": _SumMerge()},
        edges={"a": [], "b": [], "merge": ["a", "b"]},
        sink="merge",
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(pipe: Pipeline, batch: dict) -> jax.Array:
        # The DAG is already applied via pipe.step() inside scan;
        # here `batch` is the result of the DAG. Sum it as the loss.
        return jnp.sum(batch["x"])

    def loss_fn(model: Pipeline) -> jax.Array:
        # Manually drive: fetch + DAG forward, sum result.
        pipe_batch = model.step()  # type: ignore[reportCallIssue]
        return jnp.sum(pipe_batch["x"])

    grads = nnx.grad(loss_fn)(pipeline)
    grad_state = nnx.state(grads, nnx.Param)
    grad_leaves = jax.tree.leaves(grad_state)

    # Two learnable params, both should have non-zero grads.
    assert len(grad_leaves) == 2
    for g in grad_leaves:
        assert bool(jnp.any(jnp.abs(g) > 0)), "grad must be non-zero on both branches"
