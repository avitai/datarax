"""A materialized source meets JAX and Flax NNX transforms as the source it was built from does.

Three pipelines serve the same records: over PNG records decoded per batch (the encoded source),
over their decoded copy on disk, and over the decoded columns in RAM (``materialize``'s two
results). A train step taking ``raw_batches()`` and ``pipe.dag`` inside ``nnx.value_and_grad``
runs under ``nnx.jit`` in Flax's three modes, ``(graph, graph_updates)`` = ``(True, True)``,
``(True, False)`` and ``(False, False)``: from cleared caches its program compiles once, then
never over ten more steps, which run with host-to-device and device-to-host transfers disallowed,
and its losses and gradients are equal bit for bit over the three sources. The gradient matches a
float64 finite difference (``jax.test_util.check_grads``); ``nnx.vmap`` of a per-example loss
equals the losses one by one; a ``lax.scan`` over a chunk equals its steps. No pipeline holds a
dataset-shaped leaf in its NNX state (the RAM result holds its columns as every eager source does,
in a host holder), so a transform taking the whole pipeline transfers no data either; the control
is a pipeline whose stage holds the columns as ``nnx.data``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.test_util import check_grads
from substrax.testing.compiles import compiled_programs

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import from_words, to_words
from datarax.core.operator import OperatorModule
from datarax.pipeline import materialize, Pipeline
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.placement import PlacementSpy
from tests.test_common.worker_reads import png_source, stream_of, write_png_records


_RECORDS = 48
_BATCH = 4
_MODES = [(True, True), (True, False), (False, False)]
"""Flax's ``(graph, graph_updates)`` modes: the default, the Trainer's binding, tree mode."""
_KINDS = ["encoded", "copy", "ram"]


class _Scale(OperatorModule):
    """A learnable scale on ``image``: one ``nnx.Param``, in the precision the mode gives it."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.scale = nnx.Param(jnp.asarray(0.5, jnp.float32))

    def apply(self, element: Element, key: jax.Array | None = None, stats: Any = None) -> Element:
        del key, stats
        scale = self.scale[...]
        image = element.data["image"].astype(jnp.promote_types(scale.dtype, jnp.float32))
        return element.update_data({"image": image / 255.0 * scale})


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Callable[[], Any]]:
    """Each kind's source, built once; the encoded one anew per call (its files stay)."""
    root = tmp_path_factory.mktemp("materialized")
    paths = write_png_records(root / "png", _RECORDS, shards=2)
    copy = materialize(png_source(paths), directory=root / "copy")
    ram = materialize(png_source(paths))
    # The same records with the shards swapped: other data, the same structure.
    swapped = materialize(png_source(list(reversed(paths))))
    return {
        "encoded": lambda: png_source(paths),
        "copy": lambda: copy,
        "ram": lambda: ram,
        "ram_swapped": lambda: swapped,
    }


def _pipeline(source: Any, *, num_epochs: int | None = None) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[_Scale()],
        batch_size=_BATCH,
        rngs=nnx.Rngs(4),
        shuffle=True,
        num_epochs=num_epochs,
    )


def _step(graph: bool, graph_updates: bool) -> Any:
    @nnx.jit(graph=graph, graph_updates=graph_updates)
    def step(dag: nnx.Module, batch: Batch) -> tuple[jax.Array, Any]:
        def loss(dag: nnx.Module) -> jax.Array:
            return jnp.mean(dag(batch)["image"] ** 2)

        return nnx.value_and_grad(loss)(dag)

    return step


def _steps(kind: str, sources: dict[str, Any], graph: bool, graph_updates: bool) -> list[bytes]:
    """Eleven steps: the first from cleared caches compiles the step, the next ten nothing."""
    pipe = _pipeline(sources[kind]())
    step = _step(graph, graph_updates)
    batches: Iterator[Batch] = iter(pipe.raw_batches())
    first = next(batches)
    with expect_first_call_compiles("jit(step)"):
        results = [jax.block_until_ready(step(pipe.dag, first))]
    window = [next(batches) for _ in range(10)]  # 44 of 48 records: no epoch end inside
    with (
        compiled_programs() as built,
        jax.transfer_guard_host_to_device("disallow"),
        jax.transfer_guard_device_to_host("disallow"),
    ):
        results.extend(jax.block_until_ready(step(pipe.dag, batch)) for batch in window)
    assert "jit(step)" not in built, built
    pipe.close()
    return [
        np.asarray(jax.device_get(leaf)).tobytes() for r in results for leaf in jax.tree.leaves(r)
    ]


class TestTheTrainStep:
    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_compiles_once_transfers_nothing_and_agrees_across_sources(
        self, sources: dict[str, Any], graph: bool, graph_updates: bool
    ) -> None:
        found = {kind: _steps(kind, sources, graph, graph_updates) for kind in _KINDS}

        assert found["copy"] == found["encoded"]
        assert found["ram"] == found["encoded"]

    @pytest.mark.parametrize("kind", _KINDS)
    def test_the_gradient_matches_a_float64_finite_difference(
        self, sources: dict[str, Any], kind: str
    ) -> None:
        pipe = _pipeline(sources[kind]())
        batch = jax.device_get(next(iter(pipe.raw_batches())))
        pipe.close()
        with jax.enable_x64(True):
            image = jnp.asarray(np.asarray(batch["image"], np.float64))
            operator = _Scale()
            operator.scale = nnx.Param(jnp.asarray(0.5, jnp.float64))
            graphdef, state = nnx.split(operator)

            def loss(scale: jax.Array, image: jax.Array) -> jax.Array:
                held = nnx.merge(graphdef, jax.tree.map(lambda _: scale, state))
                return jnp.mean(held.apply(Element({"image": image})).data["image"] ** 2)

            check_grads(loss, (jnp.asarray(0.5, jnp.float64), image), order=1, modes=("fwd", "rev"))

    @pytest.mark.parametrize("kind", _KINDS)
    def test_a_vmapped_per_example_loss_equals_the_losses_one_by_one(
        self, sources: dict[str, Any], kind: str
    ) -> None:
        pipe = _pipeline(sources[kind]())
        batch = next(iter(pipe.raw_batches()))
        pipe.close()
        operator = _Scale()

        def per_example(op: _Scale, image: jax.Array) -> jax.Array:
            return jnp.mean(op.apply(Element({"image": image})).data["image"] ** 2)

        vmapped = nnx.vmap(per_example, in_axes=(None, 0))(operator, batch["image"])
        one_by_one = [per_example(operator, image) for image in batch["image"]]
        np.testing.assert_array_equal(np.asarray(vmapped), np.asarray(jnp.stack(one_by_one)))

    @pytest.mark.parametrize("kind", _KINDS)
    def test_a_scan_over_a_chunk_equals_its_steps(self, sources: dict[str, Any], kind: str) -> None:
        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
            value = jnp.sum(batch["image"].astype(jnp.float32))
            return total + value, value

        pipe = _pipeline(sources[kind](), num_epochs=1)
        chunks = [c for c in pipe.raw_batches(3) if c["image"].ndim == 5]
        pipe.close()
        single_pipe = _pipeline(sources[kind](), num_epochs=1)
        single = [body(jnp.float32(0), b)[1] for b in single_pipe.raw_batches()]
        single_pipe.close()
        scanned = np.concatenate(
            [np.asarray(jax.lax.scan(body, jnp.float32(0), c)[1]) for c in chunks]
        )
        np.testing.assert_array_equal(scanned, np.asarray(single[: len(scanned)]))

    @pytest.mark.parametrize("kind", _KINDS)
    def test_a_batch_sharded_over_a_data_axis_gives_each_record_its_unsharded_result(
        self, sources: dict[str, Any], kind: str
    ) -> None:
        """``shard_map`` over the batch's record axis: each device runs the DAG on its rows."""
        devices = jax.devices()[:_BATCH]
        if len(devices) < 2:
            pytest.skip("needs at least two devices")
        mesh = Mesh(np.array(devices), ("data",))
        pipe = _pipeline(sources[kind]())
        batch = next(iter(pipe.raw_batches()))
        pipe.close()
        dag = pipe.dag

        def per_record(rows: Batch) -> jax.Array:
            return jnp.mean(dag(rows)["image"] ** 2, axis=tuple(range(1, 4)))

        sharded = jax.jit(
            jax.shard_map(per_record, mesh=mesh, in_specs=P("data"), out_specs=P("data"))
        )(jax.device_put(batch, NamedSharding(mesh, P("data"))))

        assert sharded.sharding.spec == P("data")
        assert len(sharded.sharding.device_set) == len(devices)
        np.testing.assert_array_equal(np.asarray(sharded), np.asarray(jax.jit(per_record)(batch)))

    @pytest.mark.parametrize("kind", _KINDS)
    def test_one_placement_per_unit_on_the_consumer_s_thread(
        self, sources: dict[str, Any], kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = PlacementSpy(monkeypatch)
        pipe = _pipeline(sources[kind](), num_epochs=1)
        served = list(pipe.raw_batches())
        pipe.close()

        assert spy.placed_on_the_caller(len(served))


class TestState:
    @pytest.mark.parametrize("kind", _KINDS)
    @pytest.mark.parametrize("graph", [True, False])
    def test_round_trips_and_holds_no_dataset_shaped_leaf(
        self, sources: dict[str, Any], kind: str, graph: bool
    ) -> None:
        pipe = _pipeline(sources[kind](), num_epochs=1)
        with nnx.set_graph_mode(graph):
            graphdef, state = nnx.split(pipe)
            merged = nnx.merge(graphdef, state)
        for leaf in jax.tree.leaves(nnx.state(pipe)):
            assert np.ndim(leaf) == 0 or np.shape(leaf)[0] != _RECORDS, np.shape(leaf)
        assert not [leaf for leaf in jax.tree.leaves(nnx.state(pipe.dag)) if np.ndim(leaf) > 1]
        assert stream_of(merged.raw_batches()) == stream_of(
            _pipeline(sources[kind](), num_epochs=1).raw_batches()
        )

    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    @pytest.mark.parametrize("kind", _KINDS)
    def test_a_transform_taking_the_whole_pipeline_transfers_no_data(
        self, sources: dict[str, Any], kind: str, graph: bool, graph_updates: bool
    ) -> None:
        pipe = _pipeline(sources[kind](), num_epochs=1)
        step = nnx.jit(
            lambda p: jax.tree.leaves(nnx.state(p.dag, nnx.Param))[0] * 2,
            graph=graph,
            graph_updates=graph_updates,
            keep_unused=True,
        )
        step(pipe)  # the first call places the parameter and the pipeline's key

        with jax.transfer_guard_host_to_device("disallow"):
            for _ in range(3):
                step(pipe)

    @pytest.mark.parametrize(("graph", "graph_updates"), _MODES)
    def test_positive_control_columns_held_as_state_beside_the_pipeline_are_transferred(
        self, sources: dict[str, Any], graph: bool, graph_updates: bool
    ) -> None:
        held = sources["ram"]()
        words = np.arange(len(held), dtype=np.uint64)
        columns = held.get_batch(to_words(words), contiguous=True).data
        holder = _ColumnsAsState(_pipeline(held, num_epochs=1), dict(columns))
        step = nnx.jit(
            lambda h: jax.tree.leaves(nnx.state(h.pipe.dag, nnx.Param))[0] * 2,
            graph=graph,
            graph_updates=graph_updates,
            keep_unused=True,
        )

        with (
            pytest.raises(jax.errors.JaxRuntimeError, match="Disallowed host-to-device"),
            jax.transfer_guard_host_to_device("disallow"),
        ):
            for _ in range(3):
                step(holder)


class _ColumnsAsState(nnx.Module):
    """The control: a pipeline beside the same columns held as ``nnx.data``, which is state."""

    def __init__(self, pipe: Pipeline, columns: dict[str, Any]) -> None:
        self.pipe = pipe
        self.columns = nnx.data(columns)


class TestTheRamResultIsAnEagerSource:
    """The RAM result keeps an eager source's guarantees (B5): programs keyed on its structure."""

    def test_a_second_ram_result_of_the_structure_compiles_nothing_and_serves_its_own_data(
        self, sources: dict[str, Any]
    ) -> None:
        first, second = sources["ram"](), sources["ram_swapped"]()
        assert nnx.graphdef(first) == nnx.graphdef(second)
        _pipeline(first).step()
        other = _pipeline(second)

        with compiled_programs() as built:
            batch = other.step()

        assert built == []
        rows = from_words(np.asarray(batch.indices))
        expected = second.get_batch(to_words(rows)).data["label"]
        np.testing.assert_array_equal(np.asarray(batch["label"]), expected)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
    def test_a_caller_s_transform_takes_its_device_columns_and_is_refused_without_them(
        self, sources: dict[str, Any], graph: bool
    ) -> None:
        pipeline = _pipeline(sources["ram"]())
        with pytest.raises(TypeError, match=r"device_columns\(\)"):
            nnx.jit(lambda p: p.step(), graph=graph)(pipeline)
        columns = pipeline.device_columns()
        step = nnx.jit(lambda p, c: p.step(columns=c), graph=graph)
        step(pipeline, columns)

        with compiled_programs() as built, jax.transfer_guard_host_to_device("disallow"):
            jax.block_until_ready(step(pipeline, columns))

        assert built == []


@pytest.mark.parametrize("kind", ["copy", "ram"])
def test_a_tier_a_epoch_equals_the_encoded_source_s(sources: dict[str, Any], kind: str) -> None:
    def epoch(source: Any) -> list[Any]:
        pipe = _pipeline(source, num_epochs=1)
        served = stream_of(iter(list(pipe)))
        pipe.close()
        return served

    assert epoch(sources[kind]()) == epoch(sources["encoded"]())
