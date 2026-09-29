"""``OperatorDag``: a pipeline's stage graph as one ``nnx.Module`` from ``Batch`` to ``Batch``.

The DAG holds no source, no position and no ``Rngs``: it is the part of a pipeline that runs
inside a differentiated train step, over a raw batch whose records carry their identities, so
its operators' parameters train with the model (design D16).
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing import TraceCounter
from substrax.testing.compiles import expect_compiles

from datarax.core import batch_ops
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule, require_key
from datarax.operators.batch_mix_operator import BatchMixOperator, BatchMixOperatorConfig
from datarax.pipeline.dag import OperatorDag
from datarax.pipeline.nodes import SplitField
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


B = 8


class _Scale(OperatorModule):
    """Deterministic, learnable: ``image * scale``."""

    def __init__(self, value: float = 1.5) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.scale = nnx.Param(jnp.asarray(value))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del key, stats
        return element.replace(data={**data, "image": data["image"] * self.scale[...]})


class _JitteredScale(OperatorModule):
    """Stochastic, learnable: ``image * scale * jitter``, the jitter drawn from the record's key."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=seed)
        )
        self.scale = nnx.Param(jnp.asarray(0.75))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del stats
        jitter = jax.random.uniform(require_key(key, self), (), minval=0.5, maxval=1.5)
        return element.replace(data={**data, "image": data["image"] * self.scale[...] * jitter})


class _CountWrite(OperatorModule):
    """Writes a per-record state entry a later operator reads."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        state = element.state
        del key, stats
        return element.replace(state={**state, "seen": jnp.ones((), jnp.float32)})


class _CountRead(OperatorModule):
    """Adds the state entry the previous operator wrote to the image."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        state = element.state
        del key, stats
        return element.replace(data={**data, "image": data["image"] + state["seen"]})


class _Merge(nnx.Module):
    """A two-input node: the union of two branches' data."""

    def __call__(self, left: Batch, right: Batch) -> Batch:
        return left.replace(data={**left.data, **right.data})


class _ReturnsDict(nnx.Module):
    def __call__(self, batch: Batch) -> dict[str, Any]:
        return dict(batch.data)


class _Head(nnx.Module):
    """A model: a fixed-shape weight, loss ``sum(w * image)``."""

    def __init__(self) -> None:
        self.w = nnx.Param(jnp.linspace(-1.0, 1.0, 4 * 4 * 3).reshape(4, 4, 3))

    def __call__(self, image: jax.Array) -> jax.Array:
        return jnp.sum(self.w[...] * image)


def raw_batch(seed: int = 0, epoch: int = 0, first: int = 0) -> Batch:
    """A raw batch as a source serves it: values plus each record's identity."""
    rng = np.random.default_rng(seed)
    rows = np.arange(first, first + B, dtype=np.uint32)
    return batch_ops.from_arrays(
        {"image": jnp.asarray(rng.random((B, 4, 4, 3)), jnp.float32), "label": jnp.arange(B)}
    ).replace(
        indices=jnp.stack([jnp.zeros(B, jnp.uint32), jnp.asarray(rows)], -1),
        epochs=jnp.full((B,), epoch, jnp.int32),
    )


def linear_dag() -> OperatorDag:
    return OperatorDag.from_stages([_Scale(), _JitteredScale()])


def loss(model: _Head, dag: OperatorDag, batch: Batch) -> jax.Array:
    return model(dag(batch)["image"])


class TestStructure:
    """The DAG as a module: its plan is static, its stages are graph children."""

    def test_a_linear_dag_runs_its_stages_in_order(self) -> None:
        batch = raw_batch()
        dag = OperatorDag.from_stages([_Scale(2.0), _Scale(3.0)])

        out = dag(batch)

        assert isinstance(out, Batch)
        np.testing.assert_allclose(out["image"], batch["image"] * 6.0, rtol=1e-6)
        assert out.indices is batch.indices and out.epochs is batch.epochs

    def test_an_empty_dag_returns_its_batch(self) -> None:
        batch = raw_batch()

        assert OperatorDag.from_stages([])(batch) is batch

    def test_state_written_by_one_operator_reaches_the_next_and_the_output(self) -> None:
        batch = raw_batch()

        out = OperatorDag.from_stages([_CountWrite(), _CountRead()])(batch)

        np.testing.assert_allclose(out["image"], batch["image"] + 1.0, rtol=1e-6)
        np.testing.assert_array_equal(out.states["seen"], np.ones(B))

    def test_a_node_receives_its_predecessors_batches_in_order(self) -> None:
        batch = raw_batch()
        dag = OperatorDag(
            nodes={
                "image": SplitField(["image"]),
                "label": SplitField(["label"]),
                "scaled": _Scale(2.0),
                "merged": _Merge(),
            },
            edges={"image": [], "label": [], "scaled": ["image"], "merged": ["scaled", "label"]},
            sink="merged",
        )

        out = dag(batch)

        assert set(out.data) == {"image", "label"}
        np.testing.assert_allclose(out["image"], batch["image"] * 2.0, rtol=1e-6)
        np.testing.assert_array_equal(out["label"], batch["label"])

    def test_a_node_returning_something_other_than_a_batch_is_refused(self) -> None:
        dag = OperatorDag.from_stages([_ReturnsDict()])

        with pytest.raises(TypeError, match="stage_0.*Batch"):
            dag(raw_batch())

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_the_plan_is_static_and_hashable(self, graph: bool) -> None:
        """Identically built DAGs have equal graphdefs and share one compiled trace."""
        first, second = linear_dag(), linear_dag()
        graphdef_a, _ = nnx.split(first, graph=graph)
        graphdef_b, _ = nnx.split(second, graph=graph)
        counter = TraceCounter()
        apply = nnx.jit(counter.wrap(lambda dag, batch: dag(batch)["image"]), graph=graph)

        assert graphdef_a == graphdef_b and hash(graphdef_a) == hash(graphdef_b)
        with counter.expect(new_traces=1):
            apply(first, raw_batch())
        with counter.expect(new_traces=0):
            apply(second, raw_batch(1))

    def test_the_dag_holds_no_source_position_or_rngs(self) -> None:
        dag = linear_dag()

        assert not any(isinstance(node, nnx.Rngs) for _, node in nnx.iter_graph(dag))
        assert not any(isinstance(node, DataSourceModule) for _, node in nnx.iter_graph(dag))
        assert not hasattr(dag, "_position") and not hasattr(dag, "_epoch")


def _scale(dag: OperatorDag, name: str) -> jax.Array:
    """The learnable scale of the DAG's node ``name`` (or of its gradient)."""
    stage = dag.stages[name]
    assert isinstance(stage, _Scale | _JitteredScale)
    return stage.scale[...]


def _split_step(graph: bool):
    """A train step over ``(model, dag)``: the ``nnx.Param`` partition differentiated.

    Tree mode refuses a key leaf in the differentiated tree, as it does for flax's own
    key-holding modules, so the parameters are split out and differentiated alone.
    """
    model, dag = _Head(), linear_dag()
    graphdef, params, rest = nnx.split((model, dag), nnx.Param, ..., graph=graph)

    @jax.jit
    def step(params: nnx.State, batch: Batch) -> tuple[jax.Array, nnx.State]:
        def of_params(params: nnx.State) -> jax.Array:
            model, dag = nnx.merge(graphdef, params, rest)
            return loss(model, dag, batch)

        return jax.value_and_grad(of_params)(params)

    return step, params, graphdef, rest


class TestInsideADifferentiatedStep:
    """The DAG and a model in one ``value_and_grad``: operator parameters train with the model."""

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_operator_parameters_get_their_closed_form_gradients(self, graph: bool) -> None:
        step, params, graphdef, rest = _split_step(graph)
        batch = raw_batch()

        value, grads = step(params, batch)
        model, dag = nnx.merge(graphdef, params, rest)
        grad_model, grad_dag = nnx.merge(graphdef, grads, rest)

        # The output is linear in each scale, so dL/ds = L / s; dL/dw is the DAG's output.
        for name in ("stage_0", "stage_1"):
            np.testing.assert_allclose(_scale(grad_dag, name), value / _scale(dag, name), rtol=1e-5)
        expected_w = jnp.sum(dag(batch)["image"], axis=0)
        np.testing.assert_allclose(grad_model.w[...], expected_w, rtol=1e-5, atol=1e-5)
        assert float(value) != 0.0

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_batches_differing_in_values_and_identities_compile_nothing(self, graph: bool) -> None:
        step, params, _, _ = _split_step(graph)
        step(params, raw_batch())

        with expect_compiles(0):
            for seed in range(1, 4):
                step(params, raw_batch(seed, epoch=seed, first=seed * 100))

    def test_a_batch_s_identities_change_its_draws(self) -> None:
        step, params, _, _ = _split_step(True)

        first, _ = step(params, raw_batch(0))
        other, _ = step(params, raw_batch(0, epoch=1))

        assert float(first) != float(other)

    def test_jit_partial_binds_the_model_and_the_dag(self) -> None:
        """The Trainer's binding: modules bound once, the batch the only runtime argument."""
        model, dag = _Head(), linear_dag()
        batch = raw_batch()

        bound = nnx.jit_partial(loss, model, dag, graph=False)

        np.testing.assert_allclose(bound(batch), loss(model, dag, batch), rtol=1e-6)
        with expect_compiles(0):
            bound(raw_batch(2, epoch=3))

    def test_a_k_step_scan_over_a_chunk_equals_k_steps(self) -> None:
        """A ``(K, B, ...)`` chunk scanned in one call gives what K single steps give."""
        step, params, _, _ = _split_step(True)
        batches = [raw_batch(k, first=k * B) for k in range(3)]

        def sgd(params: nnx.State, batch: Batch) -> tuple[nnx.State, jax.Array]:
            value, grads = step(params, batch)
            return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads), value

        scanned, values = jax.jit(lambda p, c: jax.lax.scan(sgd, p, c))(
            params, batch_ops.stack(batches)
        )
        stepped, expected = params, []
        for batch in batches:
            stepped, value = sgd(stepped, batch)
            expected.append(value)

        np.testing.assert_allclose(values, jnp.stack(expected), rtol=1e-5)
        for got, want in zip(jax.tree.leaves(scanned), jax.tree.leaves(stepped), strict=True):
            np.testing.assert_allclose(got, want, rtol=1e-5)

    def test_a_whole_batch_operator_runs_as_a_node(self) -> None:
        """BatchMix mixes across the batch; as a DAG node it compiles once and mixes by identity."""
        mix = BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=nnx.Rngs(batch_mix=0))
        dag = OperatorDag.from_stages([mix])
        apply = nnx.jit(lambda dag, batch: dag(batch)["image"])
        batch = raw_batch()
        first = apply(dag, batch)

        with expect_compiles(0):
            again = apply(dag, batch)
            other = apply(dag, raw_batch(0, epoch=1))

        assert not jnp.allclose(first, batch["image"])
        np.testing.assert_array_equal(first, again)
        assert not jnp.allclose(first, other)


class TestPipeline:
    """The pipeline holds its DAG and yields ``Batch``es carrying their records' identities."""

    def test_iteration_yields_batches_named_by_their_records(self) -> None:
        data = {"image": np.arange(12 * 3, dtype=np.float32).reshape(12, 3)}
        pipe = Pipeline.from_arrays(data, batch_size=4, seed=0, shuffle=True)

        batches = list(pipe)

        assert all(isinstance(batch, Batch) for batch in batches)
        rows = np.concatenate([np.asarray(batch.indices)[:, 1] for batch in batches])
        assert sorted(rows.tolist()) == list(range(12))
        for batch in batches:
            np.testing.assert_array_equal(
                batch["image"], data["image"][np.asarray(batch.indices)[:, 1]]
            )
        assert all(int(epoch) == 0 for batch in batches for epoch in batch.epochs)

    def test_step_yields_a_batch_and_its_dag_is_the_pipelines_stages(self) -> None:
        stages = [_Scale(2.0)]
        source = MemorySource(
            MemorySourceConfig(), data={"image": np.ones((8, 4, 4, 3), np.float32)}
        )
        pipe = Pipeline(source=source, stages=stages, batch_size=4, rngs=nnx.Rngs(0))

        out = pipe.step()

        assert isinstance(out, Batch) and isinstance(pipe.dag, OperatorDag)
        assert pipe.stages == stages
        np.testing.assert_allclose(out["image"], 2.0)
        np.testing.assert_array_equal(np.asarray(out.indices)[:, 1], [0, 1, 2, 3])

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param({}, id="neither"),
            pytest.param({"nodes": {"a": _Scale()}, "edges": {"a": []}}, id="dag-without-sink"),
        ],
    )
    def test_an_incomplete_construction_shape_is_refused(self, shape: dict) -> None:
        source = MemorySource(
            MemorySourceConfig(), data={"image": np.ones((8, 4, 4, 3), np.float32)}
        )

        with pytest.raises(ValueError, match="Provide either stages="):
            Pipeline(source=source, batch_size=4, rngs=nnx.Rngs(0), **shape)

    def test_a_stream_yields_batches_named_by_their_positions(self) -> None:
        """A stream without record ids names each record by its arrival position (C5b gives
        streams their own identities)."""
        chunks = [{"image": np.full((4, 4, 4, 3), k, np.float32)} for k in range(3)]
        pipe = Pipeline(
            source=_Stream(chunks), stages=[_Scale(2.0)], batch_size=4, rngs=nnx.Rngs(0)
        )

        batches = list(pipe)

        assert all(isinstance(batch, Batch) for batch in batches)
        rows = np.concatenate([np.asarray(batch.indices)[:, 1] for batch in batches])
        np.testing.assert_array_equal(rows, np.arange(12))
        np.testing.assert_allclose(batches[2]["image"], 4.0)


class _Stream(DataSourceModule):
    """A forward-only source over prepared batches, then an empty batch."""

    def __init__(self, chunks: list[dict[str, np.ndarray]]) -> None:
        super().__init__(StructuralConfig(stochastic=False))
        self._chunks = nnx.data(list(chunks))
        self._served = 0

    def supports_indexed_access(self) -> bool:
        return False

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"image": jax.ShapeDtypeStruct((4, 4, 3), np.float32)}

    def get_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        del batch_size
        if self._served == len(self._chunks):
            return {}
        self._served += 1
        return self._chunks[self._served - 1]
