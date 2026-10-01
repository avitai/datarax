"""``Pipeline.step()``: one compiled batch fetch that copies nothing and composes with transforms.

``step()`` is used eagerly and inside the caller's own transforms (a jitted train step, ``grad``
to learnable stages, ``nnx.scan``, ``nnx.vmap``, a functional ``jax.jit`` over split state), so
each contract is checked in each setting. The source's arrays are never written by a step, so a
step must return and copy none of them: every call leaves the source's buffers where they are.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.config import MapOperatorConfig
from datarax.core.element_batch import Batch
from datarax.operators.map_operator import MapOperator
from datarax.pipeline import Pipeline
from datarax.pipeline.iteration import _host_copies, _session_step
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.step_jaxpr import host_callbacks, traced_step


_N = 64
_BATCH = 8


class _Scale(nnx.Module):
    """Learnable stage multiplying ``x`` by a parameter."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = nnx.Param(jnp.float32(factor))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "x": batch["x"] * self.factor[...]})


class _Shift(nnx.Module):
    """A stage of a different structure, adding a parameter."""

    def __init__(self) -> None:
        self.offset = nnx.Param(jnp.float32(1.0))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "x": batch["x"] + self.offset[...]})


class _Counter(nnx.Module):
    """Stage counting batches in a Variable that starts as a Python int, as user code writes it."""

    def __init__(self) -> None:
        self.seen = nnx.Variable(0)

    def __call__(self, batch: Batch) -> Batch:
        self.seen[...] = self.seen[...] + 1
        return batch


class _KeepsRngs(nnx.Module):
    """A stage keeping the ``Rngs`` it was given, so it shares that ``Rngs``'s Variables."""

    def __init__(self, rngs: nnx.Rngs) -> None:
        self.rngs = rngs

    def __call__(self, batch: Batch) -> Batch:
        return batch


class _GrowingStage(nnx.Module):
    """Stage adding state while it runs, which changes the module structure."""

    def __call__(self, batch: Batch) -> Batch:
        self.seen = nnx.Variable(jnp.zeros((), jnp.int32))
        return batch


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.float32(3.0))


def _pipeline(
    stage: nnx.Module | None = None, *, shuffle: bool = False, device: bool = False
) -> Pipeline:
    host = np.arange(_N, dtype=np.float32)[:, None] + 1.0
    data = {"x": jnp.asarray(host) if device else host}
    source = MemorySource(MemorySourceConfig(), data=data)
    stages = [stage] if stage is not None else []
    return Pipeline(
        source=source,
        stages=stages,
        batch_size=_BATCH,
        num_epochs=None,
        rngs=nnx.Rngs(0),
        shuffle=shuffle,
    )


def _source_data(pipeline: Pipeline) -> dict:
    """The data mapping of the pipeline's ``MemorySource``."""
    source = pipeline.source
    assert isinstance(source, MemorySource)
    assert isinstance(source.data, dict)
    return source.data


def _source_records(pipeline: Pipeline) -> np.ndarray:
    """The host array holding the source's records (the array a copy would replace).

    Compared by identity, not by buffer address: a freed copy's address can be reused by the next
    copy, so equal addresses do not show that nothing was copied, while holding the object keeps its
    buffer alive and makes identity exact.
    """
    records = _source_data(pipeline)["x"]
    assert isinstance(records, np.ndarray)
    return records


class TestNoCopy:
    """A step returns and copies none of the source's arrays."""

    def test_device_columns_are_held_once_on_the_host_and_kept(self) -> None:
        pipeline = _pipeline(_Scale(), shuffle=True, device=True)
        before = _source_records(pipeline)
        pipeline.step()
        for _ in range(3):
            pipeline.step()
        assert _source_records(pipeline) is before

    def test_host_data_is_not_transferred_on_every_step(self) -> None:
        """NumPy data is uploaded once, not on every step (the transfer guard raises otherwise)."""
        pipeline = _pipeline(_Scale(), shuffle=True)
        pipeline.step()
        pipeline.step()
        with jax.transfer_guard_host_to_device("disallow"):
            jax.block_until_ready(pipeline.step())

    def test_host_data_stays_on_the_host(self) -> None:
        """Stepping leaves the caller's NumPy arrays in the source, so host-side reads stay host."""
        pipeline = _pipeline(_Scale())
        pipeline.step()
        assert isinstance(_source_data(pipeline)["x"], np.ndarray)

    def test_step_and_iteration_share_one_device_copy_of_host_data(self) -> None:
        pipeline = _pipeline(_Scale(), shuffle=True)
        pipeline.step()
        after_step = dict(_host_copies(pipeline))
        next(iter(pipeline))
        after_iteration = _host_copies(pipeline)
        assert len(after_step) == 1
        assert list(after_iteration) == list(after_step)
        assert all(after_iteration[key][1] is copy for key, (_, copy) in after_step.items())

    def test_host_data_replaced_between_steps_is_served(self) -> None:
        pipeline = _pipeline()
        pipeline.step()
        _source_data(pipeline)["x"] = np.full((_N, 1), 7.0, dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(pipeline.step()["x"]), np.full((_BATCH, 1), 7.0))

    def test_batches_equal_iteration(self) -> None:
        stepped, iterated = _pipeline(_Scale(), shuffle=True), _pipeline(_Scale(), shuffle=True)
        batches = iter(iterated)
        for _ in range(5):
            expected, served = next(batches), stepped.step()
            assert set(served.data) == set(expected.data)
            np.testing.assert_array_equal(served["x"], expected["x"])
            np.testing.assert_array_equal(served.indices, expected.indices)
            np.testing.assert_array_equal(served.epochs, expected.epochs)


class TestStructure:
    """Writes are kept, structure changes between steps are honored, changes within one refused."""

    def test_a_repeated_step_compiles_nothing(self) -> None:
        pipeline = _pipeline(_Scale())
        pipeline.step()
        with compiled_programs() as compiled:
            pipeline.step()
        assert compiled == []

    def test_a_structural_change_between_steps_compiles_once_and_is_served(self) -> None:
        pipeline = _pipeline(_Scale())
        pipeline.step()
        pipeline.batch_size = _BATCH // 2
        with compiled_programs() as compiled:
            smaller = pipeline.step()
            pipeline.step()
        assert smaller["x"].shape[0] == _BATCH // 2
        assert len(compiled) == 1

    def test_state_written_from_a_python_int_compiles_the_step_once(self) -> None:
        """A written Variable that starts host-typed is an array after its first write.

        Every later call, session or ``step()``, then passes an array where the first passed a
        Python int; the step must not be traced again for it.
        """
        pipeline = _pipeline(_Counter())
        with compiled_programs() as compiled:
            pipeline.step()
            pipeline.step()
            session = iter(pipeline)
            next(session)
            next(session)
            next(iter(pipeline))
            pipeline.step()
        assert len([name for name in compiled if "session_step" in name]) == 1

    def test_step_and_iteration_share_one_dispatch_entry(self) -> None:
        """Both paths present the shared step one call signature: one executable, one entry.

        ``jax.jit`` keys its dispatch cache on the arguments' shardings and committedness too, so
        a source's Python-int counter passed as is by one path and converted to an array by the
        other would add a second entry for the same executable.
        """
        pipeline = _pipeline(_Scale())
        graphdef = nnx.graphdef(pipeline, graph=True)
        pipeline.step()
        next(iter(pipeline))
        pipeline.step()
        step = _session_step(graphdef, type(pipeline)._next_batch, pipeline.batch_size)
        assert step._cache_size() == 1

    def test_a_stage_adding_state_while_it_runs_is_refused(self) -> None:
        with pytest.raises(ValueError, match="changed the module structure"):
            _pipeline(_GrowingStage()).step()


class TestGraphMode:
    """Flax plans tree mode as the default; pipelines share Variables, so they run in graph mode."""

    @staticmethod
    def _sharing_rngs() -> Pipeline:
        """Stages and pipeline built from one ``Rngs``, a stage keeping it: they share it."""
        rngs = nnx.Rngs(0, augment=1)
        stage = MapOperator(
            MapOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda x, key: x + jax.random.normal(key, x.shape),
            rngs=rngs,
        )
        data = {"x": np.arange(_N, dtype=np.float32)[:, None]}
        source = MemorySource(MemorySourceConfig(), data=data)
        return Pipeline(
            source=source,
            stages=[stage, _KeepsRngs(rngs)],
            batch_size=_BATCH,
            num_epochs=None,
            rngs=rngs,
            shuffle=True,
        )

    def test_the_fixture_shares_variables(self) -> None:
        with pytest.raises(ValueError, match="Duplicate"):
            nnx.split(self._sharing_rngs(), graph=False)

    def test_step_and_iteration_run_with_tree_mode_as_the_default(self) -> None:
        stepped, iterated = self._sharing_rngs(), self._sharing_rngs()
        with nnx.set_graph_mode(False), nnx.set_graph_updates(False):
            served = stepped.step()
            expected = next(iter(iterated))
        np.testing.assert_array_equal(np.asarray(served["x"]), np.asarray(expected["x"]))

    def test_scan_runs_with_tree_mode_as_the_default(self) -> None:
        scanned, stepped = self._sharing_rngs(), self._sharing_rngs()
        with nnx.set_graph_mode(False), nnx.set_graph_updates(False):
            sums = scanned.scan(lambda batch: batch["x"].sum(), length=3)
        expected = [np.asarray(stepped.step()["x"]).sum(dtype=np.float32) for _ in range(3)]
        # Two compiled programs may associate a float32 sum of 8 values differently: up to
        # about 7 eps apart.
        np.testing.assert_allclose(
            np.asarray(sums), expected, rtol=8 * float(np.finfo(np.float32).eps)
        )


class TestTransforms:
    """``step()`` inside the caller's transforms."""

    def test_inside_a_jitted_train_step_the_gradient_reaches_the_model(self) -> None:
        model, pipeline = _Model(), _pipeline(_Scale(2.0))

        @nnx.jit
        def train_step(model: _Model, pipeline: Pipeline) -> jax.Array:
            def loss(model: _Model, pipeline: Pipeline) -> jax.Array:
                return model.weight[...] * pipeline.step()["x"].sum()

            return nnx.grad(loss)(model, pipeline).weight[...]

        with jax.checking_leaks():
            gradient = train_step(model, pipeline)
        with compiled_programs() as compiled:
            train_step(model, pipeline)
        # First batch x = 1..8, scaled by 2.
        assert float(gradient) == pytest.approx(2.0 * sum(range(1, _BATCH + 1)))
        assert compiled == []
        assert int(pipeline._position[...]) == 2 * _BATCH

    def test_the_gradient_reaches_a_replaced_stage_only(self) -> None:
        pipeline = _pipeline(_Scale(2.0))
        original = pipeline.stages[0]
        pipeline.dag.stages[pipeline.dag.order[0]] = _Shift()

        grads = nnx.grad(lambda pipeline: pipeline.step()["x"].sum())(pipeline)

        params = jax.tree.leaves(nnx.state(grads, nnx.Param))
        assert len(params) == 1
        assert float(params[0]) == pytest.approx(_BATCH)
        assert isinstance(original, _Scale)
        assert float(original.factor[...]) == 2.0

    def test_under_nnx_scan(self) -> None:
        pipeline, reference = _pipeline(_Scale()), _pipeline(_Scale())

        @nnx.scan(in_axes=(nnx.StateAxes({...: nnx.Carry}), 0), out_axes=0)
        def sums(pipeline: Pipeline, index: jax.Array) -> jax.Array:
            del index
            return pipeline.step()["x"].sum()

        scanned = np.asarray(sums(pipeline, jnp.arange(4)))
        expected = [float(reference.step()["x"].sum()) for _ in range(4)]
        np.testing.assert_allclose(scanned, expected)
        assert int(pipeline._position[...]) == 4 * _BATCH

    def test_under_nnx_vmap_with_the_pipeline_broadcast(self) -> None:
        model, pipeline = _Model(), _pipeline()

        @nnx.vmap(in_axes=(None, None, 0))
        def scaled(model: _Model, pipeline: Pipeline, factor: jax.Array) -> jax.Array:
            return factor * model.weight[...] * pipeline.step()["x"].sum()

        out = np.asarray(scaled(model, pipeline, jnp.arange(3.0)))
        np.testing.assert_allclose(out, np.arange(3.0) * 3.0 * sum(range(1, _BATCH + 1)))

    def test_inside_a_functional_jax_jit_over_split_state(self) -> None:
        pipeline, reference = _pipeline(_Scale()), _pipeline(_Scale())
        graphdef, state = nnx.split(pipeline)

        @jax.jit
        def fetch(state: nnx.State) -> tuple[jax.Array, nnx.State]:
            module = nnx.merge(graphdef, state)
            return module.step()["x"].sum(), nnx.state(module)

        first, state = fetch(state)
        second, state = fetch(state)
        nnx.update(pipeline, state)
        assert float(first) == pytest.approx(float(reference.step()["x"].sum()))
        assert float(second) == pytest.approx(float(reference.step()["x"].sum()))
        assert int(pipeline._position[...]) == 2 * _BATCH


class TestShuffledAugmentingTrainStep:
    """A shuffling pipeline and a stochastic stage inside the caller's compiled gradient step.

    The step names its records through the shuffle, gathers them by those names, and keys the
    stage's randomness on them, so every part of the record path runs inside the user's
    ``nnx.jit`` and ``grad``.
    """

    @staticmethod
    def _pipeline() -> Pipeline:
        stage = MapOperator(
            MapOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda x, key: x + jax.random.normal(key, x.shape),
            rngs=nnx.Rngs(augment=1),
        )
        data = {"x": np.arange(_N, dtype=np.float32)[:, None] + 1.0}
        source = MemorySource(MemorySourceConfig(), data=data)
        return Pipeline(
            source=source,
            stages=[stage],
            batch_size=_BATCH,
            num_epochs=None,
            rngs=nnx.Rngs(0),
            shuffle=True,
        )

    def test_one_compile_serves_three_epochs_of_the_steps_batches(self) -> None:
        model, pipeline, reference = _Model(), self._pipeline(), self._pipeline()

        @nnx.jit
        def train_step(model: _Model, pipeline: Pipeline) -> jax.Array:
            def loss(model: _Model, pipeline: Pipeline) -> jax.Array:
                return model.weight[...] * pipeline.step()["x"].sum()

            return nnx.grad(loss)(model, pipeline).weight[...]

        steps = 3 * _N // _BATCH
        first = train_step(model, pipeline)
        with compiled_programs() as compiled:
            gradients = [first] + [train_step(model, pipeline) for _ in range(steps - 1)]
        expected = [np.asarray(reference.step()["x"]).sum(dtype=np.float32) for _ in range(steps)]
        assert compiled == []
        # Two compiled programs may associate a float32 sum of 8 values differently.
        np.testing.assert_allclose(
            np.asarray(gradients), expected, rtol=8 * float(np.finfo(np.float32).eps)
        )
        assert (int(pipeline._position[...]), int(pipeline._epoch[...])) == (_N, 2)

    def test_the_step_holds_no_host_callback(self) -> None:
        closed, _ = traced_step(self._pipeline())
        assert host_callbacks(closed) == []

    def test_inside_a_functional_jax_jit_over_split_state(self) -> None:
        pipeline, reference = self._pipeline(), self._pipeline()
        graphdef, state = nnx.split(pipeline)

        @jax.jit
        def fetch(state: nnx.State) -> tuple[jax.Array, nnx.State]:
            module = nnx.merge(graphdef, state)
            return module.step()["x"], nnx.state(module)

        served, state = fetch(state)
        batches = [served]
        with compiled_programs() as compiled:
            for _ in range(_N // _BATCH):  # into the second epoch
                served, state = fetch(state)
                batches.append(served)
        nnx.update(pipeline, state)
        assert compiled == []
        for batch in batches:
            np.testing.assert_allclose(np.asarray(batch), np.asarray(reference.step()["x"]))
        assert int(pipeline._epoch[...]) == 1
