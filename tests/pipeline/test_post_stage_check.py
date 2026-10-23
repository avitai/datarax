"""Every stage's output meets the naming rules (N2), checked in every trace and on eager calls.

A stage may write any key, declared or not. Its output is checked after the stage runs: keys are
non-empty strings neither starting nor ending with "/" nor holding "//", nodes are dicts, and no
two nodes share a spelling. The check reads structure only, so it runs while a program is traced
and never on a cached call; an eager call has no cached program and checks each time.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter

import datarax.pipeline.dag as dag_module
from datarax.core import batch_ops
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.pipeline.dag import OperatorDag
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


class _Nest(OperatorModule):
    """Moves ``id`` into the group ``meta``."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig())

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        data = dict(element.data)
        nested = {"id": data.pop("id")}
        return element.replace(data={**data, "meta": nested})


class _AddLiteral(OperatorModule):
    """Adds the literal key ``meta/id`` beside the group ``meta`` holding ``id`` (defect 13)."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig())
        self.scale = nnx.Param(jnp.asarray(2.0))

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        return element.update_data({"meta/id": element.data["x"] * self.scale[...]})


class _AddScaled(OperatorModule):
    """Adds ``scaled``, a field no other node is spelled as."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig())
        self.scale = nnx.Param(jnp.asarray(3.0))

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        return element.update_data({"scaled": element.data["x"] * self.scale[...]})


class _WritesKey(OperatorModule):
    """Adds ``data[key]``: the key or the value can break the naming rules."""

    def __init__(self, key: str, value: str = "array") -> None:
        super().__init__(OperatorConfig())
        self.key = key
        self.value = value

    def apply(self, element: Element, key: Any = None, stats: Any = None) -> Element:
        made = element.data["x"] + 1.0
        return element.update_data({self.key: [made, made] if self.value == "list" else made})


def _batch(size: int = 2) -> Batch:
    return batch_ops.from_arrays(
        {"x": jnp.arange(size, dtype=jnp.float32), "id": jnp.arange(size, dtype=jnp.int32)}
    )


def _clash() -> OperatorDag:
    return OperatorDag.from_stages([_Nest(), _AddLiteral()])


def _clean() -> OperatorDag:
    return OperatorDag.from_stages([_Nest(), _AddScaled()])


CLASH = (
    "stage 'stage_1' (_AddLiteral) returned a record in which 'meta/id' names two fields: "
    "('meta', 'id') and ('meta/id',). Return one of them under another key."
)


@pytest.fixture
def check_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the post-stage check's runs: one per stage per trace or eager call."""
    calls: list[int] = []
    rule = dag_module.record_problems

    def counted(tree: Any) -> Any:
        calls.append(1)
        return rule(tree)

    monkeypatch.setattr(dag_module, "record_problems", counted)
    return calls


class TestTheRefusal:
    def test_an_undeclared_write_making_a_spelling_name_two_fields_is_refused(self) -> None:
        with pytest.raises(ValueError) as caught:
            _clash()(_batch())
        assert str(caught.value) == CLASH

    @pytest.mark.parametrize(
        ("key", "value", "words"),
        [
            ("/score", "array", "the key '/score', which starts with '/'"),
            ("", "array", "an empty key"),
            ("score", "list", "'score' is a list"),
        ],
    )
    def test_an_exotic_key_an_empty_key_and_a_list_are_refused_by_label(
        self, key: str, value: str, words: str
    ) -> None:
        dag = OperatorDag.from_stages([_WritesKey(key, value)])
        with pytest.raises(ValueError, match="stage 'stage_0' \\(_WritesKey\\) returned a record"):
            dag(_batch())
        with pytest.raises(ValueError, match=words.replace("(", r"\(").replace(")", r"\)")):
            dag(_batch())

    def test_a_record_meeting_the_rules_is_served_unchanged(self) -> None:
        out = _clean()(_batch())
        assert set(out.data) == {"x", "meta", "scaled"}

    def test_a_dag_node_that_is_not_an_operator_is_checked_too(self) -> None:
        class _Node(nnx.Module):
            def __call__(self, batch: Batch) -> Batch:
                return batch.replace(data={**batch.data, "a//b": batch.data["x"]})

        dag = OperatorDag(nodes={"merge": _Node()}, edges={"merge": []}, sink="merge")
        with pytest.raises(ValueError, match="stage 'merge' \\(_Node\\) returned a record"):
            dag(_batch())


class TestEveryPipelinePath:
    """Iteration, step, scan and session all run the DAG inside one compiled program."""

    @staticmethod
    def _pipeline(stages: list[nnx.Module]) -> Pipeline:
        data = {"x": np.arange(8, dtype=np.float32), "id": np.arange(8, dtype=np.int32)}
        return Pipeline(
            source=MemorySource(MemorySourceConfig(), data),
            stages=stages,
            batch_size=2,
            rngs=nnx.Rngs(0),
        )

    @pytest.mark.parametrize("path", ["iteration", "step", "scan", "session"])
    def test_the_refusal_reaches_every_path(self, path: str) -> None:
        pipe = self._pipeline([_Nest(), _AddLiteral()])
        with pytest.raises(ValueError, match="'meta/id' names two fields"):
            if path == "iteration":
                next(iter(pipe))
            elif path == "step":
                pipe.step()
            elif path == "scan":
                pipe.scan(lambda batch: batch["x"].sum(), length=2)
            else:
                next(iter(pipe.session()))

    def test_iteration_checks_once_per_stage_and_never_per_batch(
        self, check_calls: list[int]
    ) -> None:
        pipe = self._pipeline([_Nest(), _AddScaled()])
        batches = list(pipe)
        assert len(batches) == 4
        assert len(check_calls) == 2  # two stages, one trace


@pytest.mark.parametrize("graph", [True, False])
class TestUnderTransforms:
    """The check traces with the DAG: no retrace, equal graphdefs, gradients unchanged."""

    def test_nnx_jit_traces_once_over_rebuilt_dags_and_checks_only_while_tracing(
        self, graph: bool, check_calls: list[int]
    ) -> None:
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda dag, batch: dag(batch)["scaled"]), graph=graph)
        with counter.expect(new_traces=1):
            for dag in (_clean(), _clean(), _clean()):
                apply(dag, _batch())
                apply(dag, _batch())
        assert len(check_calls) == 2
        with counter.expect(new_traces=1):  # a new batch size traces again, and checks again
            apply(_clean(), _batch(4))
        assert len(check_calls) == 4

    def test_five_cached_calls_run_no_check(self, graph: bool, check_calls: list[int]) -> None:
        dag = _clean()
        apply = nnx.jit(lambda dag, batch: dag(batch)["scaled"], graph=graph)
        apply(dag, _batch())
        traced = len(check_calls)
        for _ in range(5):
            apply(dag, _batch())
        assert traced == 2
        assert len(check_calls) == traced

    def test_the_refusal_under_nnx_jit_is_the_eager_one(self, graph: bool) -> None:
        with pytest.raises(ValueError) as caught:
            nnx.jit(lambda dag, batch: dag(batch), graph=graph)(_clash(), _batch())
        assert str(caught.value) == CLASH

    def test_graphdefs_are_unchanged_by_calls(self, graph: bool) -> None:
        dag = _clean()
        before = nnx.graphdef(dag, graph=graph)
        dag(_batch())
        nnx.jit(lambda dag, batch: dag(batch), graph=graph)(dag, _batch())
        after = nnx.graphdef(dag, graph=graph)
        assert before == after and hash(before) == hash(after)

    def test_value_and_grad_through_a_checked_dag(self, graph: bool) -> None:
        def loss(dag: OperatorDag, batch: Batch) -> jax.Array:
            return dag(batch)["scaled"].sum()

        grads = nnx.jit(nnx.value_and_grad(loss, graph=graph), graph=graph)(_clean(), _batch())[1]
        assert float(grads.stages["stage_1"].scale[...]) == pytest.approx(1.0)

    def test_nnx_vmap_over_k_batches_checks_once(self, graph: bool, check_calls: list[int]) -> None:
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch())
        apply = nnx.vmap(lambda dag, batch: dag(batch)["scaled"], in_axes=(None, 0), graph=graph)
        out = apply(_clean(), stacked)
        assert out.shape == (2, 2)
        assert len(check_calls) == 2  # two stages, one trace for both batches

    def test_nnx_scan_over_a_chunk_checks_once(self, graph: bool, check_calls: list[int]) -> None:
        chunk = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch(), _batch())

        def body(dag: OperatorDag, total: jax.Array, batch: Batch) -> jax.Array:
            return total + dag(batch)["scaled"].sum()

        scan = nnx.scan(body, in_axes=(None, nnx.Carry, 0), out_axes=nnx.Carry, graph=graph)
        total = scan(_clean(), jnp.float32(0), chunk)
        assert float(total) == pytest.approx(3 * 3.0 * 1.0)
        assert len(check_calls) == 2

    def test_eval_shape_runs_the_check(self, graph: bool) -> None:
        with pytest.raises(ValueError, match="'meta/id' names two fields"):
            nnx.eval_shape(lambda dag, batch: dag(batch), _clash(), _batch(), graph=graph)


class TestJaxTransforms:
    def test_jax_jit_over_a_static_graphdef_traces_once(self, check_calls: list[int]) -> None:
        graphdef, state = nnx.split(_clean())
        counter = TraceCounter()

        @jax.jit
        @counter.wrap
        def apply(state: Any, batch: Batch) -> jax.Array:
            return nnx.merge(graphdef, state)(batch)["scaled"]

        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(state, _batch())
            for _ in range(3):
                apply(nnx.split(_clean())[1], _batch())
        assert len(check_calls) == 2

    def test_jax_jit_with_a_closure_checks_only_while_tracing(self, check_calls: list[int]) -> None:
        dag = _clean()
        counter = TraceCounter()
        apply = jax.jit(counter.wrap(lambda batch: dag(batch)["scaled"]))
        with counter.expect(new_traces=1):
            for _ in range(3):
                apply(_batch())
        assert len(check_calls) == 2
        with counter.expect(new_traces=1):  # a new batch size traces again, and checks again
            apply(_batch(4))
        assert len(check_calls) == 4

    def test_vmap_over_k_batches(self) -> None:
        graphdef, state = nnx.split(_clean())
        stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch())
        out = jax.jit(jax.vmap(lambda batch: nnx.merge(graphdef, state)(batch)["scaled"]))(stacked)
        assert out.shape == (2, 2)

    def test_lax_scan_over_a_chunk(self) -> None:
        graphdef, state = nnx.split(_clean())
        chunk = jax.tree.map(lambda *leaves: jnp.stack(leaves), _batch(), _batch(), _batch())

        def body(carry: jax.Array, batch: Batch) -> tuple[jax.Array, None]:
            return carry + nnx.merge(graphdef, state)(batch)["scaled"].sum(), None

        total, _ = jax.jit(lambda chunk: jax.lax.scan(body, jnp.float32(0), chunk))(chunk)
        assert float(total) == pytest.approx(3 * 3.0 * 1.0)

    def test_grad_values_are_the_closed_form(self) -> None:
        graphdef, params, rest = nnx.split(_clean(), nnx.Param, ...)

        def loss(params: Any, batch: Batch) -> jax.Array:
            return nnx.merge(graphdef, params, rest)(batch)["scaled"].sum()

        grads = jax.jit(jax.grad(loss))(params, _batch())
        assert float(grads["stages"]["stage_1"]["scale"][...]) == pytest.approx(1.0)


class TestEagerCalls:
    """An eager call has no cached program: the check runs on every call (EAGER-CHECKS)."""

    def test_an_eager_dag_checks_every_call(self, check_calls: list[int]) -> None:
        dag = _clean()
        for _ in range(3):
            dag(_batch())
        assert len(check_calls) == 6

    def test_an_eager_dag_refuses_what_the_jitted_one_refuses(self) -> None:
        with pytest.raises(ValueError) as eager:
            _clash()(_batch())
        with pytest.raises(ValueError) as jitted:
            nnx.jit(lambda dag, batch: dag(batch))(_clash(), _batch())
        assert str(eager.value) == str(jitted.value)

    def test_disable_jit_still_refuses(self) -> None:
        with jax.disable_jit(), pytest.raises(ValueError, match="'meta/id' names two fields"):
            nnx.jit(lambda dag, batch: dag(batch))(_clash(), _batch())
