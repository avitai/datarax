"""A host-read ``Batch`` through the pipeline's traced surfaces (section 7 of step C5a-2).

``source.get_batch(indices, epochs=...)`` names its rows as the session names the same records,
so operators key them identically; ``pipe.dag`` over it runs inside a differentiated, jitted
step whose gradients match float64 finite differences, compiling once for batches differing in
values and identities. Sources hold NumPy columns, so a session and a user's jitted step upload
no records per batch after warm-up and read nothing back to the host. Every transfer-guard check
has an implicit-transfer positive control in the same process.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

from datarax.core.config import ElementOperatorConfig, StructuralConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import to_words
from datarax.core.prng import per_record_keys
from datarax.operators.element_operator import ElementOperator
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import EagerSource, MemorySource, MemorySourceConfig


_N = 16
_BATCH = 4


def _columns() -> dict[str, np.ndarray]:
    return {"x": np.linspace(0.5, 2.0, _N * 3, dtype=np.float32).reshape(_N, 3)}


class _Columns(EagerSource):
    """An eager-base source over NumPy columns."""

    def __init__(self) -> None:
        super().__init__(StructuralConfig())
        self._store(_columns())


class _Scale(nnx.Module):
    """A learnable stage."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(1.5))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "x": batch["x"] * self.factor[...]})


def _jitter(element: Element, key: jax.Array) -> Element:
    x = element.data["x"]
    return element.update_data({"x": x + 0.1 * jax.random.normal(key, x.shape, x.dtype)})


def _pipeline(source: EagerSource | None = None, *, shuffle: bool = True) -> Pipeline:
    noise = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="jitter"),
        fn=_jitter,
        rngs=nnx.Rngs(jitter=4),
    )
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), _columns()) if source is None else source,
        stages=[_Scale(), noise],
        batch_size=_BATCH,
        rngs=nnx.Rngs(0),
        num_epochs=None,
        shuffle=shuffle,
    )


class _Train(nnx.Module):
    """A model after the pipeline's DAG: what a train step differentiates."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.dag = pipeline.dag
        self.weight = nnx.Param(jnp.float32(0.7))

    def loss(self, batch: Batch) -> jax.Array:
        return jnp.sum((self.dag(batch)["x"] * self.weight[...]) ** 2)


def _host_read(source: object, rows: list[int], epoch: int) -> Batch:
    assert isinstance(source, EagerSource)
    return source.get_batch(to_words(np.asarray(rows, np.uint64)), epochs=epoch)


class TestTheHostReadNamesRecordsAsTheSessionDoes:
    def test_operators_key_a_host_read_batch_as_the_session_s(self) -> None:
        pipeline = _pipeline()
        served = next(iter(pipeline))
        rows = [int(r) for r in np.asarray(served.indices)[:, 1]]
        host = _host_read(pipeline.source, rows, int(np.asarray(served.epochs)[0]))
        base = jax.random.key(9)
        np.testing.assert_array_equal(
            jax.random.key_data(per_record_keys(base, host.indices, host.epochs, host.draws)),
            jax.random.key_data(per_record_keys(base, served.indices, served.epochs, served.draws)),
        )

    def test_the_dag_over_a_host_read_batch_is_the_session_s_batch(self) -> None:
        pipeline = _pipeline(shuffle=False)
        served = next(iter(pipeline))
        host = pipeline.dag(_host_read(pipeline.source, [0, 1, 2, 3], 0))
        np.testing.assert_allclose(np.asarray(host["x"]), np.asarray(served["x"]), rtol=1e-6)


class TestGradientsThroughTheDag:
    """D16: the DAG over a host-read batch is differentiable end to end, the read excepted."""

    def test_parameter_gradients_match_float64_finite_differences(self) -> None:
        pipeline = _pipeline()
        batch = _host_read(pipeline.source, [5, 2, 9, 14], 1)
        gradient = check_parameter_gradients(_Train(pipeline), lambda train: train.loss(batch))
        assert len(jax.tree.leaves(gradient)) == 2  # the stage's factor and the model's weight

    def test_the_input_gradient_matches_float64_finite_differences(self) -> None:
        pipeline = _pipeline()
        batch = _host_read(pipeline.source, [5, 2, 9, 14], 1)

        def loss(train: _Train, x: jax.Array) -> jax.Array:
            return train.loss(batch.replace(data={**batch.data, "x": x}))

        check_input_gradients(_Train(pipeline), loss, batch["x"])

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_one_compile_serves_batches_of_other_values_and_records(self, graph: bool) -> None:
        pipeline = _pipeline()
        train = _Train(pipeline)
        step = nnx.jit(
            lambda train, batch: nnx.value_and_grad(lambda m: m.loss(batch))(train), graph=graph
        )
        batches = [
            _host_read(pipeline.source, rows, epoch)
            for rows, epoch in (([5, 2, 9, 14], 1), ([0, 3, 7, 11], 2), ([15, 15, 1, 8], 7))
        ]
        with expect_compiles(1):
            first = step(train, batches[0])
        with expect_compiles(0):
            rest = [step(train, batch) for batch in batches[1:]]
        values = [float(value) for value, _ in (first, *rest)]
        assert len(set(values)) == 3


def _implicit_transfer_raises() -> bool:
    """The control: a NumPy argument to a jitted function is an implicit host-to-device transfer."""
    double = jax.jit(lambda x: x * 2)
    jax.block_until_ready(double(jnp.ones(2)))
    with jax.transfer_guard_host_to_device("disallow"):
        try:
            double(np.ones(2))
        except RuntimeError:
            return True
    return False


def _device_to_host_raises() -> bool:
    """The control: reading a device array back is a device-to-host transfer."""
    value = jax.block_until_ready(jnp.ones(2) * 3)
    with jax.transfer_guard_device_to_host("disallow"):
        try:
            np.asarray(value)
        except RuntimeError:
            return True
    return False


_SOURCES = {"memory": lambda: MemorySource(MemorySourceConfig(), _columns()), "eager": _Columns}


@pytest.mark.parametrize("name", sorted(_SOURCES))
class TestNoPerBatchTransfer:
    def test_a_session_uploads_no_records_after_warm_up(self, name: str) -> None:
        assert _implicit_transfer_raises(), "the host-to-device guard does not fire here"
        session = iter(_pipeline(_SOURCES[name]()))
        for _ in range(2):
            jax.block_until_ready(next(session).indices)
        with jax.transfer_guard_host_to_device("disallow"):
            for _ in range(4):
                jax.block_until_ready(next(session).indices)

    def test_a_user_jitted_step_uploads_no_records_after_its_first_call(self, name: str) -> None:
        assert _implicit_transfer_raises(), "the host-to-device guard does not fire here"
        pipeline = _pipeline(_SOURCES[name]())
        step = nnx.jit(lambda p: p.step())
        jax.block_until_ready(step(pipeline).indices)
        with jax.transfer_guard_host_to_device("disallow"):
            for _ in range(3):
                jax.block_until_ready(step(pipeline).indices)

    def test_neither_reads_back_to_the_host(self, name: str) -> None:
        if not _device_to_host_raises():
            pytest.skip("the device-to-host guard does not fire on this backend (host memory)")
        pipeline = _pipeline(_SOURCES[name]())
        session = iter(_pipeline(_SOURCES[name]()))
        step = nnx.jit(lambda p: p.step())
        jax.block_until_ready(next(session).indices)
        jax.block_until_ready(step(pipeline).indices)
        with jax.transfer_guard_device_to_host("disallow"):
            for _ in range(3):
                jax.block_until_ready(next(session).indices)
                jax.block_until_ready(step(pipeline).indices)
