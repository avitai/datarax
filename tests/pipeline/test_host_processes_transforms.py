"""Batches read in worker processes meet JAX and Flax NNX transforms as thread-read batches do.

A worker reads into shared memory; the consumer opens the unit (``SharedMemoryArray`` leaves) and
places it, on its own thread, in its own JAX contexts (transfer guard, precision mode, default
device). Everything after the read runs on the default device: the Tier-A DAG call, a user's
``jax.jit`` or ``nnx.jit`` step (graph and tree mode), ``pipe.dag`` inside ``nnx.value_and_grad``,
and a ``lax.scan`` over a chunk. Each compiles once, then never again; one placement per unit.
On the CPU backend a placed batch is its shared-memory segment (a 64-byte-aligned buffer is
aliased by ``device_put``), which lives as long as the batch.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.pipeline import host_stage, Pipeline
from datarax.pipeline.run_units import HostElement
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.worker_reads import png_source, resources, stream_of, write_png_records


_SHM = Path("/dev/shm")


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """24 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 24)


class _Scale(OperatorModule):
    """A learnable scale on ``image``: one ``nnx.Param``, its gradient a mean of the image."""

    def __init__(self) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.scale = nnx.Param(jnp.asarray(0.5, jnp.float32))

    def apply(self, element: Element, key: jax.Array | None = None, stats: Any = None) -> Element:
        del key, stats
        image = element.data["image"].astype(jnp.float32) / 255.0
        return element.update_data({"image": image * self.scale[...]})


def _pipeline(
    paths: list[str], *, workers: int = 2, stages: Any = (), num_epochs: int | None = 1
) -> Pipeline:
    return Pipeline(
        source=png_source(paths),
        stages=list(stages),
        batch_size=4,
        rngs=nnx.Rngs(2),
        shuffle=True,
        num_epochs=num_epochs,
        host_resources=resources(workers) if workers else None,
    )


class _PlacementSpy:
    """Every unit the host stage places, and the thread that placed it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.threads: list[int] = []
        self.leaf_types: set[type] = set()
        place = host_stage._place  # noqa: SLF001 - the one placement the spy counts

        def counted(element: HostElement) -> Any:
            self.threads.append(threading.get_ident())
            self.leaf_types |= {type(leaf) for leaf in jax.tree.leaves(element.batch)}
            return place(element)

        monkeypatch.setattr(host_stage, "_place", counted)


class TestPlacementOnTheConsumer:
    def test_one_placement_per_unit_on_the_consumer_s_thread(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _PlacementSpy(monkeypatch)
        pipe = _pipeline(png_paths)
        served = list(pipe.raw_batches())
        assert len(spy.threads) == len(served)
        assert set(spy.threads) == {threading.get_ident()}
        assert any("SharedMemoryArray" in t.__name__ for t in spy.leaf_types), spy.leaf_types

    def test_the_caller_s_transfer_guard_refuses_the_placement(self, png_paths: list[str]) -> None:
        pipe = _pipeline(png_paths)
        with (
            jax.transfer_guard_host_to_device("disallow_explicit"),
            pytest.raises(Exception, match="host-to-device transfer"),
        ):
            next(iter(pipe.raw_batches()))
        pipe.close()

    def test_every_batch_lands_on_the_caller_s_default_device(self, png_paths: list[str]) -> None:
        default = jax.devices()[0]
        other = next(d for d in (*jax.devices(), *jax.devices("cpu")) if d != default)
        pipe = _pipeline(png_paths)
        with jax.default_device(other):
            devices = {leaf.device for b in pipe.raw_batches() for leaf in jax.tree.leaves(b)}
        assert devices == {other}


class TestTransforms:
    def test_the_tier_a_dag_call_compiles_once_and_runs_on_the_device(
        self, png_paths: list[str]
    ) -> None:
        pipe = _pipeline(png_paths, stages=[_Scale()])
        batches = iter(pipe)
        first = next(batches)
        rest = list(batches)
        default = jax.devices()[0]
        for batch in (first, *rest):
            assert {leaf.device for leaf in jax.tree.leaves(batch)} == {default}
        reference = list(_pipeline(png_paths, workers=0, stages=[_Scale()]))
        assert stream_of(iter([first, *rest])) == stream_of(iter(reference))

    @pytest.mark.parametrize("graph", [True, False])
    def test_a_user_s_nnx_jit_step_compiles_once_then_never(
        self, png_paths: list[str], graph: bool
    ) -> None:
        model = nnx.Linear(8 * 8 * 3, 2, rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)

        @nnx.jit(graph=graph)
        def step(model: nnx.Linear, optimizer: nnx.Optimizer, batch: Batch) -> jax.Array:
            def loss(model: nnx.Linear) -> jax.Array:
                x = batch["image"].reshape(batch.batch_size, -1).astype(jnp.float32) / 255.0
                return jnp.mean(model(x) ** 2)

            value, grads = nnx.value_and_grad(loss)(model)
            optimizer.update(model, grads)
            return value

        pipe = _pipeline(png_paths, num_epochs=2)
        batches = iter(pipe.raw_batches())
        first = next(batches)
        with expect_first_call_compiles("jit(step)"):
            jax.block_until_ready(step(model, optimizer, first))
        default = jax.devices()[0]
        for batch in batches:
            if batch.batch_size != first.batch_size:
                continue
            assert {leaf.device for leaf in jax.tree.leaves(batch)} == {default}
            with expect_compiles(0):
                jax.block_until_ready(step(model, optimizer, batch))
        pipe.close()

    def test_dag_gradients_equal_the_thread_run_s_and_a_float64_difference(
        self, png_paths: list[str]
    ) -> None:
        """``pipe.dag`` inside ``nnx.value_and_grad``: d mean(image * s) / ds = mean(image).

        The reference is a central difference of the same loss in float64 NumPy over the host
        batch. The float32 gradient sums 192 values a record over the batch, so it is held to
        ``rtol=1e-5`` with ``atol=1e-6``, the error of a float32 mean of a few hundred values of
        order 0.5, not a value tuned to a run.
        """

        @nnx.jit
        def loss_and_grad(dag: nnx.Module, batch: Batch) -> tuple[jax.Array, Any]:
            def loss(dag: nnx.Module) -> jax.Array:
                return jnp.mean(dag(batch)["image"])

            return nnx.value_and_grad(loss)(dag)

        def gradients(workers: int) -> list[tuple[float, float]]:
            pipe = _pipeline(png_paths, workers=workers, stages=[_Scale()])
            found = []
            for batch in pipe.raw_batches():
                _, grads = loss_and_grad(pipe.dag, batch)
                image = np.asarray(jax.device_get(batch["image"]), np.float64) / 255.0
                h = 1e-6
                difference = (np.mean(image * (0.5 + h)) - np.mean(image * (0.5 - h))) / (2 * h)
                found.append((float(jax.tree.leaves(grads)[0]), float(difference)))
            pipe.close()
            return found

        processes, threads = gradients(2), gradients(0)
        assert processes == threads
        for value, reference in processes:
            np.testing.assert_allclose(value, reference, rtol=1e-5, atol=1e-6)

    def test_a_scan_over_a_chunk_equals_its_steps(self, png_paths: list[str]) -> None:
        def body(total: jax.Array, batch: Batch) -> tuple[jax.Array, jax.Array]:
            value = jnp.sum(batch["image"].astype(jnp.float32))
            return total + value, value

        pipe = _pipeline(png_paths)
        chunked = [
            jax.lax.scan(body, jnp.float32(0), chunk)[1]
            for chunk in pipe.raw_batches(3)
            if chunk["image"].ndim == 5
        ]
        pipe.close()
        single = [body(jnp.float32(0), b)[1] for b in _pipeline(png_paths, workers=0).raw_batches()]
        np.testing.assert_array_equal(
            np.concatenate([np.asarray(c) for c in chunked]), np.asarray(single[: 3 * len(chunked)])
        )


@pytest.mark.skipif(not _SHM.is_dir(), reason="no /dev/shm to inspect (not Linux)")
def test_on_the_cpu_a_placed_batch_outlives_its_producer_and_frees_its_segment(
    png_paths: list[str],
) -> None:
    """The aliased segment stays mapped while the batch lives, and goes when it is dropped."""
    if jax.default_backend() != "cpu":
        pytest.skip("the aliasing is the CPU backend's")
    before = {p.name for p in _SHM.iterdir()}
    pipe = _pipeline(png_paths, num_epochs=None)
    batches = iter(pipe.raw_batches())
    kept = next(batches)
    copy = np.array(jax.device_get(kept["image"]))
    for _ in range(12):  # later units create and drop segments while ``kept`` lives
        next(batches)
    gc.collect()
    np.testing.assert_array_equal(np.asarray(jax.device_get(kept["image"])), copy)
    pipe.close()
    while_kept = {p.name for p in _SHM.iterdir()} - before
    del kept
    for _ in range(100):
        gc.collect()
        if not {p.name for p in _SHM.iterdir()} - before:
            break
        threading.Event().wait(0.05)
    assert not {p.name for p in _SHM.iterdir()} - before
    assert while_kept, "the kept batch held its segment while it lived"
