"""The final batch of an epoch: padding with a validity mask, dropping, and continuous streams.

A random-access source keeps every batch at ``batch_size`` so the compiled step has one
shape. What the rows past the end of an epoch mean is the pipeline's contract:

- ``drop_last=False`` (the default) serves ceil(N / B) batches and marks the rows past the
  epoch's end invalid in a top-level ``valid_mask`` leaf, so a masked loss ignores them;
- ``drop_last=True`` serves floor(N / B) batches, PyTorch's rule;
- ``num_epochs=None`` is a continuous stream in which a batch crosses the epoch boundary
  and no row is padding.
"""

from __future__ import annotations

from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_N = 100
_BATCH = 32


def _source(n: int = _N, *, shuffle: bool = False, seed: int = 0) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data={"x": np.arange(n, dtype=np.float32)},
        rngs=nnx.Rngs(seed, shuffle=seed),
    )


def _pipeline(n: int = _N, **kwargs) -> Pipeline:
    return Pipeline(source=_source(n), stages=[], batch_size=_BATCH, rngs=nnx.Rngs(0), **kwargs)


class TestPaddedFinalBatch:
    """The default serves every record once, padded to the batch size and masked."""

    def test_serves_ceil_batches_with_exactly_n_valid_rows(self) -> None:
        batches = list(_pipeline())

        assert len(batches) == 4  # ceil(100 / 32)
        assert all(batch["valid_mask"].shape == (_BATCH,) for batch in batches)
        assert all(batch["valid_mask"].dtype == jnp.bool_ for batch in batches)
        assert sum(int(batch["valid_mask"].sum()) for batch in batches) == _N
        np.testing.assert_array_equal(
            np.asarray(batches[-1]["valid_mask"]), np.arange(_BATCH) < _N - 3 * _BATCH
        )
        valid_rows = np.concatenate(
            [np.asarray(b["x"])[np.asarray(b["valid_mask"])] for b in batches]
        )
        np.testing.assert_array_equal(valid_rows, np.arange(_N, dtype=np.float32))

    def test_len_is_the_batches_per_epoch(self) -> None:
        assert len(_pipeline()) == 4
        assert len(_pipeline(drop_last=True)) == 3
        assert len(_pipeline(96)) == 3

    def test_step_attaches_the_same_mask(self) -> None:
        pipeline = _pipeline()
        for _ in range(3):
            assert bool(pipeline.step()["valid_mask"].all())
        last = pipeline.step()
        assert int(last["valid_mask"].sum()) == _N - 3 * _BATCH

    def test_a_full_epoch_is_all_valid(self) -> None:
        batches = list(_pipeline(96))
        assert len(batches) == 3
        assert all(bool(batch["valid_mask"].all()) for batch in batches)

    def test_the_mask_is_the_only_leaf_added(self) -> None:
        batch = next(iter(_pipeline()))
        assert set(batch) == {"x", "valid_mask"}


class TestDropLast:
    """``drop_last=True`` serves floor(N / B) full batches and nothing else."""

    def test_serves_floor_batches_all_valid(self) -> None:
        batches = list(_pipeline(drop_last=True))

        assert len(batches) == 3
        assert all(bool(batch["valid_mask"].all()) for batch in batches)
        served = np.concatenate([np.asarray(b["x"]) for b in batches])
        np.testing.assert_array_equal(served, np.arange(3 * _BATCH, dtype=np.float32))

    def test_a_source_smaller_than_a_batch_serves_nothing(self) -> None:
        pipeline = Pipeline(
            source=_source(4), stages=[], batch_size=8, rngs=nnx.Rngs(0), drop_last=True
        )
        assert len(pipeline) == 0
        assert list(pipeline) == []


class TestMultipleEpochs:
    """``num_epochs=k`` serves ``k`` epochs, each ending as the final-batch rule says."""

    def test_two_padded_epochs(self) -> None:
        pipeline = _pipeline(num_epochs=2)
        batches = list(pipeline)

        assert len(batches) == 8
        assert sum(int(batch["valid_mask"].sum()) for batch in batches) == 2 * _N
        first = np.concatenate([np.asarray(b["x"]) for b in batches[:4]])
        second = np.concatenate([np.asarray(b["x"]) for b in batches[4:]])
        np.testing.assert_array_equal(first, second)  # a sequential source repeats its order
        assert int(pipeline._epoch[...]) == 1

    def test_two_dropped_epochs(self) -> None:
        batches = list(_pipeline(num_epochs=2, drop_last=True))
        assert len(batches) == 6
        assert all(bool(batch["valid_mask"].all()) for batch in batches)

    def test_a_shuffled_source_serves_a_new_permutation_each_epoch(self) -> None:
        source = _source(96, shuffle=True)
        pipeline = Pipeline(
            source=source, stages=[], batch_size=_BATCH, rngs=nnx.Rngs(0), num_epochs=2
        )
        batches = list(pipeline)

        first = np.concatenate([np.asarray(b["x"]) for b in batches[:3]])
        second = np.concatenate([np.asarray(b["x"]) for b in batches[3:]])
        assert sorted(first.tolist()) == sorted(second.tolist()) == list(range(96))
        assert not np.array_equal(first, second)

    def test_rejects_a_non_positive_epoch_count(self) -> None:
        with pytest.raises(ValueError, match="num_epochs"):
            _pipeline(num_epochs=0)


class TestContinuousStream:
    """``num_epochs=None`` crosses the epoch boundary inside a batch and never pads."""

    def test_the_boundary_batch_holds_the_tail_and_the_next_head(self) -> None:
        pipeline = _pipeline(num_epochs=None)
        batches = list(islice(pipeline, 6))

        assert all(bool(batch["valid_mask"].all()) for batch in batches)
        boundary = np.asarray(batches[3]["x"])
        np.testing.assert_array_equal(boundary[:4], np.arange(96, 100, dtype=np.float32))
        np.testing.assert_array_equal(boundary[4:], np.arange(28, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(batches[4]["x"]), np.arange(28, 60, dtype=np.float32)
        )
        assert int(pipeline._epoch[...]) == 1
        assert int(pipeline._position[...]) == 92

    def test_a_shuffled_boundary_batch_takes_the_head_of_the_next_permutation(self) -> None:
        pipeline = Pipeline(
            source=_source(shuffle=True),
            stages=[],
            batch_size=_BATCH,
            rngs=nnx.Rngs(0),
            num_epochs=None,
        )
        batches = list(islice(pipeline, 4))
        boundary = np.asarray(batches[3]["x"])

        # The head rows are the first 28 records of epoch 1's order, which a fresh pipeline
        # reaching epoch 1 serves first.
        reference = Pipeline(
            source=_source(shuffle=True), stages=[], batch_size=_BATCH, rngs=nnx.Rngs(0)
        )
        reference.reset()
        head = np.asarray(reference.step()["x"])[:28]
        served_epoch_0 = np.concatenate([np.asarray(b["x"]) for b in batches[:3]] + [boundary[:4]])
        assert sorted(served_epoch_0.tolist()) == list(range(_N))
        np.testing.assert_array_equal(boundary[4:], head)

    def test_the_stream_does_not_stop(self) -> None:
        pipeline = _pipeline(num_epochs=None)
        assert len(list(islice(pipeline, 13))) == 13
        assert len(pipeline) == 4  # batches per epoch, still

    def test_iterator_state_reports_the_crossed_epoch(self) -> None:
        pipeline = _pipeline(num_epochs=None)
        iterator = iter(pipeline)
        for _ in range(4):
            next(iterator)
        state = iterator.get_state()  # type: ignore[union-attr]
        assert (state["position"], state["epoch"]) == (28, 1)


class TestScanLength:
    """``scan(length)`` is checked against what the epoch still holds."""

    def test_refuses_a_length_beyond_the_epoch(self) -> None:
        pipeline = _pipeline()
        with pytest.raises(ValueError, match="4 batches"):
            pipeline.scan(lambda batch: jnp.sum(batch["x"]), length=5)

    def test_runs_the_epoch_and_masks_the_last_batch(self) -> None:
        pipeline = _pipeline()
        counts = pipeline.scan(lambda batch: batch["valid_mask"].sum(), length=4)
        np.testing.assert_array_equal(np.asarray(counts), [32, 32, 32, 4])

    def test_counts_from_the_current_position(self) -> None:
        pipeline = _pipeline()
        pipeline.step()
        with pytest.raises(ValueError, match="3 batches"):
            pipeline.scan(lambda batch: jnp.sum(batch["x"]), length=4)

    def test_a_continuous_stream_scans_any_length(self) -> None:
        pipeline = _pipeline(num_epochs=None)
        totals = pipeline.scan(lambda batch: batch["valid_mask"].sum(), length=9)
        np.testing.assert_array_equal(np.asarray(totals), [_BATCH] * 9)


class TestStreamingSourceMask:
    """A streaming source, which may yield a short last batch, carries an all-true mask."""

    def test_each_batch_is_all_valid_over_its_length(self) -> None:
        from tests.pipeline.test_pipeline_streaming import _ListStream  # noqa: PLC0415

        batches = [
            {"x": jnp.array([0.0, 1.0])},
            {"x": jnp.array([2.0, 3.0])},
            {"x": jnp.array([4.0])},
        ]
        source = _ListStream(batches, {"x": jax.ShapeDtypeStruct((), jnp.float32)})
        pipeline = Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))

        masks = [np.asarray(batch["valid_mask"]) for batch in pipeline]
        assert [mask.shape[0] for mask in masks] == [2, 2, 1]
        assert all(mask.all() for mask in masks)


def test_jit_and_grad_see_the_mask_as_a_plain_leaf() -> None:
    """A training step can weight its loss by the mask under the compiled session."""
    pipeline = _pipeline()

    @jax.jit
    def masked_mean(batch: dict) -> jax.Array:
        mask = batch["valid_mask"].astype(jnp.float32)
        return jnp.sum(batch["x"] * mask) / jnp.sum(mask)

    means = [float(masked_mean(batch)) for batch in pipeline]
    assert means[-1] == pytest.approx(np.mean(np.arange(96, 100)))
