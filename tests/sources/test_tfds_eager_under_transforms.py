"""A TFDS-backed eager source under the transforms its records meet (step C5a-3, section 7).

The loader is host code and is never traced; what reaches traced code is the columns it stores
(the label as int64 on the host, the text feature as provenance). Over the offline fixture: a
session and a user's jitted step compile once; ``scan`` equals its steps; the compiled step is the
very program a ``MemorySource`` holding the old loader's int32 label builds; a tree-mode split and
merge steps alike; sources differing only in provenance share one graphdef and one compile; a
host-read batch through ``pipe.dag`` is differentiable inside ``nnx.value_and_grad`` in graph and
tree mode with one compile for batches of other values and records; ``record_indices_at`` holds
under ``vmap`` and ``lax.scan``; and steady steps upload no records and read nothing back to the
host.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles
from substrax.testing.gradients import check_input_gradients, check_parameter_gradients

from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.index_words import from_words, to_words
from datarax.operators.element_operator import ElementOperator
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig, TFDSEagerConfig, TFDSEagerSource
from datarax.sources.eager_source import HostProvenance
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture, TRAIN_RECORDS


pytestmark = pytest.mark.tfds

_BATCH = 4


def _source(
    fixture: TFDSFixture, split: str = "train", include_keys: set[str] | None = None
) -> TFDSEagerSource:
    return TFDSEagerSource(
        TFDSEagerConfig(
            name=FIXTURE,
            split=split,
            data_dir=str(fixture.array_record),
            include_keys=include_keys,
        )
    )


def _as_float(element: Element, key: jax.Array) -> Element:
    del key
    return element.update_data({"image": element.data["image"].astype(jnp.float32) / 255.0})


class _Scale(nnx.Module):
    """A learnable stage on the image."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(1.5))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "image": batch["image"] * self.factor[...]})


def _pipeline(source: Any, *, shuffle: bool = True) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[ElementOperator(ElementOperatorConfig(), fn=_as_float, rngs=nnx.Rngs(0)), _Scale()],
        batch_size=_BATCH,
        rngs=nnx.Rngs(0),
        num_epochs=None,
        shuffle=shuffle,
    )


def _named(batch: Batch) -> tuple[np.ndarray, np.ndarray]:
    return from_words(np.asarray(batch.indices)), np.asarray(batch.epochs)


def _same_batches(got: Batch, want: Batch) -> None:
    np.testing.assert_array_equal(_named(got)[0], _named(want)[0])
    np.testing.assert_array_equal(_named(got)[1], _named(want)[1])
    np.testing.assert_array_equal(np.asarray(got["image"]), np.asarray(want["image"]))
    np.testing.assert_array_equal(np.asarray(got["label"]), np.asarray(want["label"]))


class TestTheCompiledPathsOverATFDSSource:
    def test_a_session_compiles_once(self, tfds_fixture: TFDSFixture) -> None:
        session = iter(_pipeline(_source(tfds_fixture)))
        with expect_first_call_compiles("jit(session_step)"):
            jax.block_until_ready(next(session).indices)
        with expect_compiles(0):
            for _ in range(TRAIN_RECORDS // _BATCH + 2):  # past an epoch boundary
                jax.block_until_ready(next(session).indices)

    def test_a_jitted_step_compiles_once_and_steps_as_the_pipeline_does(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        pipe = _pipeline(_source(tfds_fixture))
        reference = _pipeline(_source(tfds_fixture))
        step = nnx.jit(lambda p: p.step())
        with expect_compiles(1):
            first = step(pipe)
        with expect_compiles(0):
            rest = [step(pipe) for _ in range(3)]
        for got in (first, *rest):
            _same_batches(got, reference.step())

    def test_scan_equals_its_steps(self, tfds_fixture: TFDSFixture) -> None:
        pipe = _pipeline(_source(tfds_fixture))
        reference = _pipeline(_source(tfds_fixture))

        def body(batch: Batch) -> tuple[jax.Array, jax.Array, jax.Array]:
            return jnp.asarray(batch.indices), jnp.asarray(batch.epochs), batch["image"]

        with expect_first_call_compiles("jit(iota)", "jit(scan)"):
            indices, epochs, images = pipe.scan(body, length=3)
        for k in range(3):
            want = reference.step()
            np.testing.assert_array_equal(from_words(np.asarray(indices[k])), _named(want)[0])
            np.testing.assert_array_equal(np.asarray(epochs[k]), _named(want)[1])
            np.testing.assert_array_equal(np.asarray(images[k]), np.asarray(want["image"]))

    def test_the_step_program_is_the_one_the_old_int32_label_built(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        """The int64 host label reaches the device as int32 (64-bit off): no new program."""
        source = _source(tfds_fixture, "train[:16]", include_keys={"image", "label"})
        old_columns = {
            "image": source.data["image"],
            "label": source.data["label"].astype(np.int32),
        }
        memory = MemorySource(MemorySourceConfig(), old_columns)

        def program(pipe: Pipeline) -> str:
            graphdef, state = nnx.split(pipe)
            return jax.jit(lambda s: nnx.merge(graphdef, s).step().data).lower(state).as_text()

        assert program(_pipeline(source)) == program(_pipeline(memory))

    def test_a_tree_mode_split_and_merge_steps_as_the_pipeline_does(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        pipe = _pipeline(_source(tfds_fixture))
        reference = _pipeline(_source(tfds_fixture))
        graphdef, state = nnx.split(pipe, graph=False)
        merged = nnx.merge(graphdef, state)
        for _ in range(3):
            _same_batches(merged.step(), reference.step())

    def test_sources_differing_in_provenance_share_one_graphdef_and_one_compile(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        """DD-D1: the text a record carries never decides a program.

        The split is static configuration (it names the source), so two splits are two
        programs, as before; within one configuration, provenance is out of the graph.
        """
        first, second = _source(tfds_fixture), _source(tfds_fixture)
        second._provenance = HostProvenance(
            tuple({"name": record["name"][::-1]} for record in first._provenance.value)
        )
        assert first._provenance.value != second._provenance.value
        assert nnx.split(first, graph=False)[0] == nnx.split(second, graph=False)[0]
        gather = nnx.jit(lambda s, words: s.get_records(words)["image"])
        words = jnp.asarray(to_words(np.asarray([1, 3], np.uint64)))
        with expect_compiles(1):
            jax.block_until_ready(gather(first, words))
            jax.block_until_ready(gather(second, words))


class _Train(nnx.Module):
    """A model after the pipeline's DAG: what a train step differentiates."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.dag = pipeline.dag
        self.weight = nnx.Param(jnp.float32(0.7))

    def loss(self, batch: Batch) -> jax.Array:
        return jnp.sum((self.dag(batch)["image"] * self.weight[...]) ** 2) / 1000.0


def _host_read(source: object, rows: list[int], epoch: int) -> Batch:
    assert isinstance(source, TFDSEagerSource)
    return source.get_batch(to_words(np.asarray(rows, np.uint64)), epochs=epoch)


class TestGradientsThroughTheDagOverAHostReadBatch:
    """D16: the DAG over a TFDS host read is differentiable end to end, the read excepted."""

    def test_parameter_gradients_match_float64_finite_differences(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        pipeline = _pipeline(_source(tfds_fixture))
        batch = _host_read(pipeline.source, [5, 2, 9, 14], 1)

        gradient = check_parameter_gradients(_Train(pipeline), lambda train: train.loss(batch))

        assert len(jax.tree.leaves(gradient)) == 2  # the stage's factor and the model's weight

    def test_the_image_gradient_matches_float64_finite_differences(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        pipeline = _pipeline(_source(tfds_fixture))
        batch = _host_read(pipeline.source, [5, 2, 9, 14], 1)

        def loss(train: _Train, image: jax.Array) -> jax.Array:
            return train.loss(batch.replace(data={**batch.data, "image": image}))

        check_input_gradients(_Train(pipeline), loss, np.asarray(batch["image"], np.float32))

    @pytest.mark.parametrize("graph", [True, False], ids=["graph", "tree"])
    def test_one_compile_serves_batches_of_other_values_and_records(
        self, tfds_fixture: TFDSFixture, graph: bool
    ) -> None:
        pipeline = _pipeline(_source(tfds_fixture))
        train = _Train(pipeline)
        step = nnx.jit(
            lambda train, batch: nnx.value_and_grad(lambda m: m.loss(batch))(train), graph=graph
        )
        batches = [
            _host_read(pipeline.source, rows, epoch)
            for rows, epoch in (([5, 2, 9, 14], 1), ([0, 3, 7, 11], 2), ([19, 19, 1, 8], 7))
        ]
        with expect_compiles(1):
            first = step(train, batches[0])
        with expect_compiles(0):
            rest = [step(train, batch) for batch in batches[1:]]
        assert len({float(value) for value, _ in (first, *rest)}) == 3


class TestRecordIndicesUnderTransforms:
    def test_under_vmap_over_keys(self, tfds_fixture: TFDSFixture) -> None:
        source = _source(tfds_fixture)
        keys = jax.random.split(jax.random.key(3), 4)

        served = jax.vmap(lambda key: source.record_indices_at(2, 6, key))(keys)

        for k in range(4):
            np.testing.assert_array_equal(served[k], source.record_indices_at(2, 6, keys[k]))

    def test_under_scan_over_starts(self, tfds_fixture: TFDSFixture) -> None:
        source = _source(tfds_fixture)
        key = jax.random.key(5)
        starts = jnp.arange(0, 30, 6, dtype=jnp.int32)

        @jax.jit
        def scanned(starts: jax.Array) -> jax.Array:
            def body(carry: None, start: jax.Array) -> tuple[None, jax.Array]:
                return carry, source.record_indices_at(start, 6, key)

            return jax.lax.scan(body, None, starts)[1]

        served = scanned(starts)
        for step, start in enumerate(np.asarray(starts)):
            np.testing.assert_array_equal(
                served[step], source.record_indices_at(int(start), 6, key)
            )


def _device_to_host_raises() -> bool:
    """The control: reading a device array back is a device-to-host transfer."""
    value = jax.block_until_ready(jnp.ones(2) * 3)
    with jax.transfer_guard_device_to_host("disallow"):
        try:
            np.asarray(value)
        except RuntimeError:
            return True
    return False


def _implicit_upload_raises() -> bool:
    """The control: a NumPy argument to a jitted function is an implicit host-to-device transfer."""
    double = jax.jit(lambda x: x * 2)
    jax.block_until_ready(double(jnp.ones(2)))
    with jax.transfer_guard_host_to_device("disallow"):
        try:
            double(np.ones(2))
        except RuntimeError:
            return True
    return False


def test_steady_steps_upload_no_records(tfds_fixture: TFDSFixture) -> None:
    """The int64 host label is not converted and uploaded again on every step."""
    assert _implicit_upload_raises(), "the host-to-device guard does not fire here"
    session = iter(_pipeline(_source(tfds_fixture)))
    pipeline = _pipeline(_source(tfds_fixture))
    step = nnx.jit(lambda p: p.step())
    for _ in range(2):
        jax.block_until_ready(next(session).indices)
        jax.block_until_ready(step(pipeline).indices)
    with jax.transfer_guard_host_to_device("disallow"):
        for _ in range(3):
            jax.block_until_ready(next(session).indices)
            jax.block_until_ready(step(pipeline).indices)


def test_steady_steps_read_nothing_back_to_the_host(tfds_fixture: TFDSFixture) -> None:
    if not _device_to_host_raises():
        pytest.skip("the device-to-host guard does not fire on this backend (host memory)")
    session = iter(_pipeline(_source(tfds_fixture)))
    pipeline = _pipeline(_source(tfds_fixture))
    step = nnx.jit(lambda p: p.step())
    jax.block_until_ready(next(session).indices)
    jax.block_until_ready(step(pipeline).indices)
    with jax.transfer_guard_device_to_host("disallow"):
        for _ in range(3):
            jax.block_until_ready(next(session).indices)
            jax.block_until_ready(step(pipeline).indices)
