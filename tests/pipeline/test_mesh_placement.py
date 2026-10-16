"""A pipeline given a ``sharding`` places each batch on its data mesh.

Full batches have their rows on the sharding and their batch-level state replicated; a chunk
``(K, B, ...)`` has its rows on the sharding's axes after the chunk axis; a bounded run's short
final batch is placed replicated, since its rows cannot be split, so an evaluation covers every
record once without padding. A ``batch_size`` the sharding cannot split is refused when the
pipeline is built.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec as P
from substrax.testing.compiles import compiled_programs, expect_compiles

from datarax.core.config import BatchMixOperatorConfig, ElementOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.core.state_keys import MIX_LAMBDA
from datarax.operators import ElementOperator
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.streams import RecordStream


pytestmark = pytest.mark.devices(8)

_LENGTH, _BATCH = 100, 32  # three full batches and a short final batch of 4


def _mesh(axis_type: AxisType = AxisType.Auto) -> jax.sharding.Mesh:
    return jax.make_mesh((8,), ("data",), axis_types=(axis_type,))


def _rows(axis_type: AxisType = AxisType.Auto) -> NamedSharding:
    return NamedSharding(_mesh(axis_type), P("data"))


def _source(length: int = _LENGTH) -> MemorySource:
    values = np.arange(length, dtype=np.float32)
    return MemorySource(MemorySourceConfig(), {"value": values, "image": np.stack([values] * 4, 1)})


def _pipeline(
    sharding: NamedSharding | None,
    *,
    length: int = _LENGTH,
    stages: list[nnx.Module] | None = None,
    num_epochs: int | None = 1,
) -> Pipeline:
    return Pipeline(
        source=_source(length),
        stages=[] if stages is None else stages,
        batch_size=_BATCH,
        rngs=nnx.Rngs(0),
        shuffle=True,
        num_epochs=num_epochs,
        sharding=sharding,
    )


def _indices(batch: Batch) -> list[int]:
    return [int(index) for index in from_words(np.asarray(batch.indices).reshape(-1, 2))]


def _leaf_shardings(batch: Batch) -> set[Any]:
    per_record = batch.replace(batch_state={})
    return {leaf.sharding for leaf in jax.tree.leaves(per_record)}


class TestWhereEachUnitIsPlaced:
    def test_full_batches_have_their_rows_on_the_sharding_and_the_short_batch_is_replicated(
        self,
    ) -> None:
        rows = _rows()
        replicated = NamedSharding(rows.mesh, P())

        units = list(_pipeline(rows).raw_batches())

        assert [unit["value"].shape for unit in units] == [(32,), (32,), (32,), (4,)]
        assert all(_leaf_shardings(unit) == {rows} for unit in units[:3])
        assert _leaf_shardings(units[3]) == {replicated}

    def test_a_chunk_has_its_rows_on_the_axes_after_the_chunk_axis(self) -> None:
        rows = _rows()

        units = list(_pipeline(rows).raw_batches(chunk=2))

        assert [unit["value"].shape for unit in units] == [(2, 32), (32,), (4,)]
        assert _leaf_shardings(units[0]) == {NamedSharding(rows.mesh, P(None, "data"))}
        assert _leaf_shardings(units[1]) == {rows}
        assert _leaf_shardings(units[2]) == {NamedSharding(rows.mesh, P())}

    def test_rows_on_one_axis_of_a_two_axis_mesh_are_replicated_along_the_other(self) -> None:
        mesh = jax.make_mesh((2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)
        rows = NamedSharding(mesh, P("data"))
        pipe = Pipeline(source=_source(), stages=[], batch_size=8, rngs=nnx.Rngs(0), sharding=rows)

        value = next(iter(pipe.raw_batches()))["value"]

        shards = value.addressable_shards
        assert value.sharding == rows
        assert {shard.data.shape for shard in shards} == {(4,)}
        assert sorted(shard.index[0].start for shard in shards) == [0] * 4 + [4] * 4

    def test_batch_level_state_is_replicated_on_every_batch(self) -> None:
        rows = _rows()
        mix = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", alpha=0.2, data_field="image", stream_name="mix"),
            rngs=nnx.Rngs(mix=1),
        )

        batches = list(_pipeline(rows, stages=[mix]))

        replicated = NamedSharding(rows.mesh, P())
        assert all(batch.batch_state[MIX_LAMBDA].sharding == replicated for batch in batches)
        assert batches[0]["image"].sharding == rows

    def test_without_a_sharding_batches_stay_on_the_default_device_uncommitted(self) -> None:
        batch = next(iter(_pipeline(None).raw_batches()))

        assert batch["value"].devices() == {jax.devices()[0]}
        assert not batch["value"].committed


class TestAnEvaluationOnTheMesh:
    @pytest.mark.parametrize("form", ["iteration", "raw_batches"])
    def test_totals_equal_the_unsharded_run(self, form: str) -> None:
        def run(sharding: NamedSharding | None) -> tuple[float, list[int]]:
            pipe = _pipeline(sharding)
            units = list(pipe) if form == "iteration" else list(pipe.raw_batches())
            total = sum(float(jnp.sum(unit["value"])) for unit in units)
            return total, sorted(index for unit in units for index in _indices(unit))

        assert run(_rows()) == run(None)
        assert run(_rows())[1] == list(range(_LENGTH))

    def test_a_scan_over_placed_chunks_and_the_singles_covers_every_record_once(self) -> None:
        rows = _rows()
        units = list(_pipeline(rows, length=300).raw_batches(chunk=4))

        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
            return total + jnp.sum(batch["value"]), jnp.asarray(batch["value"].shape[0])

        scan_chunk = jax.jit(lambda total, chunk: jax.lax.scan(body, total, chunk))
        step = jax.jit(body)
        total, records = jnp.zeros(()), 0
        for unit in units:
            if unit["value"].ndim == 2:
                total, counts = scan_chunk(total, unit)
                records += int(jnp.sum(counts))
            else:
                total, count = step(total, unit)
                records += int(count)

        assert [unit["value"].shape for unit in units] == [(4, 32), (4, 32), (32,), (12,)]
        assert records == 300
        assert float(total) == float(np.arange(300, dtype=np.float64).sum())

    def test_the_dag_compiles_once_per_placement_and_a_second_pass_compiles_nothing(self) -> None:
        def double(element: Any, key: Any = None) -> Any:
            del key
            return element.update_data({"value": element.data["value"] * 2.0})

        stage = ElementOperator(ElementOperatorConfig(stochastic=False), fn=double)
        rows = _rows()

        jax.clear_caches()
        with compiled_programs() as first:
            totals = [float(jnp.sum(batch["value"])) for batch in _pipeline(rows, stages=[stage])]
        with expect_compiles(0):
            again = [float(jnp.sum(batch["value"])) for batch in _pipeline(rows, stages=[stage])]

        assert first.count("jit(step)") == 2  # the sharded full batches and the replicated tail
        assert sum(totals) == sum(again) == 2.0 * float(np.arange(_LENGTH).sum())


class TestTheEndToEndStepOnTheMesh:
    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_operator_gradients_equal_the_unsharded_run(self, graph: bool) -> None:
        class Scale(nnx.Module):
            def __init__(self) -> None:
                self.weight = nnx.Param(jnp.asarray(0.5))

            def __call__(self, batch: Batch) -> Batch:
                return batch.replace(
                    data={**batch.data, "value": batch["value"] * self.weight[...]}
                )

        def gradients(sharding: NamedSharding | None) -> list[float]:
            pipe = _pipeline(sharding, stages=[Scale()])

            @nnx.jit(graph=graph)
            def grad(dag: nnx.Module, batch: Batch) -> Any:
                return nnx.grad(lambda dag: jnp.mean(dag(batch)["value"] ** 2), graph=graph)(dag)

            return [
                float(jax.tree.leaves(grad(pipe.dag, batch))[0]) for batch in pipe.raw_batches()
            ]

        np.testing.assert_allclose(gradients(_rows()), gradients(None), rtol=1e-6, atol=1e-6)


class TestAnExplicitAxisMesh:
    def test_an_explicit_axis_mesh_is_refused_naming_auto(self) -> None:
        with pytest.raises(ValueError, match="AxisType.Auto"):
            _pipeline(_rows(AxisType.Explicit))

    def test_the_operator_that_refusal_protects_fails_there(self) -> None:
        """MixUp gathers each record's partner from another row, which Explicit axes refuse."""
        mesh = _mesh(AxisType.Explicit)
        mix = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", alpha=0.2, data_field="image", stream_name="mix"),
            rngs=nnx.Rngs(mix=1),
        )
        pipe = _pipeline(None)
        batch = next(iter(pipe.raw_batches()))
        pipe.close()
        placed = jax.device_put(batch, NamedSharding(mesh, P("data")))

        # jax's ShardingTypeError, which jax exports under no public name.
        with jax.set_mesh(mesh), pytest.raises(Exception, match="out_sharding") as raised:
            mix(placed)
        assert raised.typename == "ShardingTypeError"


class TestAStream:
    def test_a_stream_s_short_final_batch_is_replicated(self) -> None:
        rows = _rows()
        stream = RecordStream({"value": np.arange(20, dtype=np.float32)})
        pipe = Pipeline(
            source=stream, stages=[], batch_size=8, rngs=nnx.Rngs(0), num_epochs=1, sharding=rows
        )

        units = list(pipe.raw_batches())

        assert [unit["value"].shape for unit in units] == [(8,), (8,), (4,)]
        assert _leaf_shardings(units[-1]) == {NamedSharding(rows.mesh, P())}
        assert all(_leaf_shardings(unit) == {rows} for unit in units[:2])


class TestRefusals:
    def test_a_batch_size_the_sharding_cannot_split_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch_size 30"):
            Pipeline(source=_source(), stages=[], batch_size=30, rngs=nnx.Rngs(0), sharding=_rows())

    def test_a_batch_split_over_two_axes_counts_both(self) -> None:
        mesh = jax.make_mesh((2, 4), ("data", "fsdp"), axis_types=(AxisType.Auto,) * 2)
        both = NamedSharding(mesh, P(("data", "fsdp")))

        with pytest.raises(ValueError, match="batch_size 6"):
            Pipeline(source=_source(), stages=[], batch_size=6, rngs=nnx.Rngs(0), sharding=both)
        pipe = Pipeline(source=_source(), stages=[], batch_size=8, rngs=nnx.Rngs(0), sharding=both)
        assert next(iter(pipe.raw_batches()))["value"].sharding == both

    def test_a_spec_beyond_the_rows_is_refused(self) -> None:
        mesh = jax.make_mesh((2, 4), ("data", "model"), axis_types=(AxisType.Auto,) * 2)

        with pytest.raises(ValueError, match="rows"):
            Pipeline(
                source=_source(),
                stages=[],
                batch_size=8,
                rngs=nnx.Rngs(0),
                sharding=NamedSharding(mesh, P("data", "model")),
            )

    def test_an_abstract_mesh_is_refused(self) -> None:
        abstract = AbstractMesh((8,), ("data",), axis_types=(AxisType.Auto,))

        with pytest.raises(ValueError, match="devices"):
            Pipeline(
                source=_source(),
                stages=[],
                batch_size=8,
                rngs=nnx.Rngs(0),
                sharding=NamedSharding(abstract, P("data")),
            )

    def test_a_sharding_that_is_not_named_is_refused(self) -> None:
        with pytest.raises(TypeError, match="NamedSharding"):
            Pipeline(
                source=_source(),
                stages=[],
                batch_size=8,
                rngs=nnx.Rngs(0),
                sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("call", ["step", "scan", "session"])
    def test_the_compiled_path_is_refused_with_a_sharding(self, call: str) -> None:
        pipe = _pipeline(_rows(), num_epochs=None)

        with pytest.raises(ValueError, match="raw_batches"):
            if call == "step":
                pipe.step()
            elif call == "scan":
                pipe.scan(lambda batch: jnp.sum(batch["value"]), length=1)
            else:
                pipe.session()


class TestThePipelineHoldsTheSharding:
    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_it_splits_and_merges_and_pipelines_built_alike_share_a_graph(
        self, graph: bool
    ) -> None:
        first, second = _pipeline(_rows()), _pipeline(_rows())

        graphdef, state = nnx.split(first, graph=graph) if not graph else nnx.split(first)
        other, _ = nnx.split(second, graph=graph) if not graph else nnx.split(second)

        hash(graphdef)
        assert graphdef == other
        assert nnx.merge(graphdef, state).sharding == first.sharding
        assert not any(isinstance(leaf, NamedSharding) for leaf in jax.tree.leaves(state))

    def test_a_state_saved_on_the_mesh_resumes_exactly(self) -> None:
        rows = _rows()
        pipe = _pipeline(rows)
        served = pipe.raw_batches()
        next(served)
        state = pipe.get_state()
        rest = [_indices(batch) for batch in served]

        resumed = _pipeline(rows)
        resumed.set_state(state)

        assert [_indices(batch) for batch in resumed.raw_batches()] == rest

    def test_one_placement_call_per_unit_and_no_batch_goes_through_device_put(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"global": 0, "batch_device_put": 0}
        make, put = jax.make_array_from_process_local_data, jax.device_put

        def counted_make(*args: Any, **kwargs: Any) -> Any:
            calls["global"] += 1
            return make(*args, **kwargs)

        def counted_put(value: Any, *args: Any, **kwargs: Any) -> Any:
            calls["batch_device_put"] += isinstance(value, Batch)
            return put(value, *args, **kwargs)

        monkeypatch.setattr(jax, "make_array_from_process_local_data", counted_make)
        monkeypatch.setattr(jax, "device_put", counted_put)
        units = list(_pipeline(_rows()).raw_batches(chunk=2))

        assert calls == {"global": len(units), "batch_device_put": 0}
