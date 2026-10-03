"""A ``Pipeline`` over a ``MixDataSourcesNode``: one record once per epoch, in the pipeline's order.

An epoch of a mix is Grain's length, so each child record is served at most once per epoch and
every record keeps its draw 0: no two rows of an epoch share a key. The pipeline owns the order:
with ``shuffle=True`` each child is ordered by the pipeline's epoch key folded with the child's
position, and each epoch reads a fresh part of every child; with ``shuffle=False`` the mix is a
fixed interleave, the same records every epoch. Until the host stage serves it, a mix of equal
children iterates through the compiled session like any indexed source.
"""

from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.core.prng import per_record_keys
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.mixing import offsets_of


_LAYOUTS = {
    "two": ((9, 14), (0.3, 0.7), 3),
    "three": ((5, 9, 14), (0.2, 0.3, 0.5), 4),
}


def _pipeline(
    layout: str, *, shuffle: bool, drop_last: bool = False, seed: int = 0, epochs: int = 3
) -> Pipeline:
    lengths, weights, batch = _LAYOUTS[layout]
    children = [
        MemorySource(MemorySourceConfig(), {"x": 100.0 * c + np.arange(n, dtype=np.float32)})
        for c, n in enumerate(lengths)
    ]
    return Pipeline(
        source=MixDataSourcesNode(MixDataSourcesConfig(weights=weights), children),
        stages=[],
        batch_size=batch,
        num_epochs=epochs,
        drop_last=drop_last,
        rngs=nnx.Rngs(seed),
        shuffle=shuffle,
    )


def _served(pipe: Pipeline) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every row the pipeline serves: its index, its epoch and its value."""
    batches: list[Batch] = list(pipe)
    return (
        np.concatenate([from_words(np.asarray(b.indices)) for b in batches]).astype(np.int64),
        np.concatenate([np.asarray(b.epochs) for b in batches]),
        np.concatenate([np.asarray(b["x"]) for b in batches]),
    )


def _by_epoch(indices: np.ndarray, epochs: np.ndarray) -> list[np.ndarray]:
    return [indices[epochs == epoch] for epoch in range(int(epochs.max()) + 1)]


@pytest.mark.parametrize("drop_last", [False, True], ids=["keep", "drop"])
@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
class TestOneRecordOncePerEpoch:
    def test_no_record_repeats_within_an_epoch_and_no_two_rows_share_a_key(
        self, layout: str, drop_last: bool
    ) -> None:
        indices, epochs, _ = _served(_pipeline(layout, shuffle=True, drop_last=drop_last))
        for epoch, named in enumerate(_by_epoch(indices, epochs)):
            assert len(np.unique(named)) == len(named), f"epoch {epoch}"
            keys = per_record_keys(
                jax.random.key(0),
                jnp.asarray(np.stack([named >> 32, named & 0xFFFFFFFF], -1).astype(np.uint32)),
                jnp.full(len(named), epoch, jnp.int32),
                jnp.zeros(len(named), jnp.int32),
            )
            data = np.asarray(jax.random.key_data(keys))
            assert len({tuple(row) for row in data.tolist()}) == len(named)

    def test_each_child_is_served_in_its_own_order_under_the_epoch_key(
        self, layout: str, drop_last: bool
    ) -> None:
        pipe = _pipeline(layout, shuffle=True, drop_last=drop_last)
        lengths = _LAYOUTS[layout][0]
        offsets = np.asarray(offsets_of(lengths))
        indices, epochs, values = _served(pipe)
        owners = np.searchsorted(offsets, indices, side="right") - 1
        np.testing.assert_array_equal(values, 100.0 * owners + (indices - offsets[owners]))
        children = cast(MixDataSourcesNode, pipe.source).sources
        for epoch in range(int(epochs.max()) + 1):
            key = pipe._key_of(jnp.int32(epoch))
            for child, source in enumerate(children):
                mine = indices[(epochs == epoch) & (owners == child)] - offsets[child]
                own = from_words(
                    source.record_indices_at(0, len(mine), jax.random.fold_in(key, child))
                )
                np.testing.assert_array_equal(mine, own.astype(np.int64))


class TestThePipelineOwnsTheOrder:
    def test_two_seeds_serve_different_orders(self) -> None:
        first = _served(_pipeline("two", shuffle=True, seed=0))[0]
        second = _served(_pipeline("two", shuffle=True, seed=1))[0]
        assert not np.array_equal(first, second)

    def test_without_a_shuffle_every_epoch_serves_the_same_interleave(self) -> None:
        indices, epochs, _ = _served(_pipeline("two", shuffle=False, drop_last=True))
        first, *rest = _by_epoch(indices, epochs)
        for epoch in rest:
            np.testing.assert_array_equal(epoch, first)
        lengths = _LAYOUTS["two"][0]
        owners = np.searchsorted(np.asarray(offsets_of(lengths)), first, side="right") - 1
        for child in range(len(lengths)):
            local = first[owners == child] - offsets_of(lengths)[child]
            np.testing.assert_array_equal(local, np.arange(len(local)))

    def test_with_a_shuffle_epochs_differ_and_reach_other_child_records(self) -> None:
        """An epoch takes 10 of the larger child's 1000 records: two epochs drawing the same 10
        has probability 1 / C(1000, 10), below 1e-23, for any key."""
        children = [
            MemorySource(MemorySourceConfig(), {"x": np.arange(n, dtype=np.float32)})
            for n in (1000, 10)
        ]
        pipe = Pipeline(
            source=MixDataSourcesNode(MixDataSourcesConfig(weights=(0.5, 0.5)), children),
            stages=[],
            batch_size=5,
            num_epochs=2,
            drop_last=True,
            rngs=nnx.Rngs(0),
            shuffle=True,
        )
        indices, epochs, _ = _served(pipe)
        served = _by_epoch(indices, epochs)
        assert not np.array_equal(served[0], served[1])
        assert set(served[0].tolist()) != set(served[1].tolist())


class TestTheCompiledSession:
    """A mix of equal children iterates through the compiled session, as any indexed source."""

    def test_a_session_compiles_once(self) -> None:
        pipe = _pipeline("two", shuffle=True, drop_last=True)
        session = iter(pipe)
        with expect_first_call_compiles("jit(session_step)"):
            jax.block_until_ready(next(session).indices)
        with expect_compiles(0):
            for batch in session:
                jax.block_until_ready(batch.indices)

    @pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
    def test_step_and_scan_serve_what_iteration_serves(self, shuffle: bool) -> None:
        iterated = _served(_pipeline("three", shuffle=shuffle))[0]
        count = len(_pipeline("three", shuffle=shuffle))
        stepper = _pipeline("three", shuffle=shuffle)
        stepped = np.concatenate(
            [from_words(np.asarray(stepper.step().indices)) for _ in range(count)]
        )
        scanned, _ = _pipeline("three", shuffle=shuffle).scan(
            lambda batch: (batch.indices, batch.epochs), length=count
        )
        # Iteration ends with the run's short batch; step() and scan serve full batches.
        served = len(iterated)
        np.testing.assert_array_equal(stepped.astype(np.int64)[:served], iterated)
        np.testing.assert_array_equal(
            from_words(np.asarray(scanned).reshape(-1, 2)).astype(np.int64)[:served], iterated
        )

    def test_a_restored_state_resumes_the_stream_bit_for_bit(self) -> None:
        running = _pipeline("three", shuffle=True)
        for _ in range(5):
            running.step()
        state = running.get_state()
        expected = [np.asarray(running.step().indices) for _ in range(9)]
        resumed = _pipeline("three", shuffle=True)
        resumed.set_state(state)
        for want in expected:
            np.testing.assert_array_equal(np.asarray(resumed.step().indices), want)

    def test_equal_seeds_serve_equal_streams(self) -> None:
        first = _served(_pipeline("three", shuffle=True, seed=4))
        second = _served(_pipeline("three", shuffle=True, seed=4))
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)
