"""``MixDataSourcesNode.get_batch``: the host read of mixed records, with the union of their fields.

``get_batch(indices, *, epochs, contiguous)`` is the indexed host read every in-memory source
serves: a ``Batch`` named by the given words and epochs, draws 0, rows in the order named, NumPy
only. The mix finds each record's child against its word offsets, reads each child once with the
child's own ``get_batch`` and joins the rows. Its records carry the union of its children's
fields: a field some child lacks is ``Maybe(value, present)`` in every batch of the mix, zeros
where a record has none, so the batch's structure never depends on which children its rows come
from (design D8, the four presence cases).
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core import batch_ops
from datarax.core.element_batch import Batch
from datarax.core.index_words import to_words
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig
from tests.test_common.device_arrays import arrays_made_since
from tests.test_common.mixing import Sized
from tests.test_common.streams import non_array_state_leaves
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


H, W, C, T = 4, 4, 3, 6


def _memory(columns: dict[str, Any]) -> MemorySource:
    return MemorySource(MemorySourceConfig(), columns)


def _mix(children: list[Any], weights: tuple[float, ...] | None = None) -> MixDataSourcesNode:
    weights = weights or tuple(1.0 / len(children) for _ in children)
    return MixDataSourcesNode(MixDataSourcesConfig(weights=weights), children)


def _words(values: list[int]) -> np.ndarray:
    return to_words(np.asarray(values, np.uint64))


def _pair() -> MixDataSourcesNode:
    return _mix(
        [
            _memory({"x": np.arange(9, dtype=np.float32), "y": np.arange(9, dtype=np.int32)}),
            _memory(
                {"x": 100 + np.arange(14, dtype=np.float32), "y": np.arange(14, dtype=np.int32)}
            ),
        ]
    )


class _Counted:
    """Counts each source's host reads, by source."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[tuple[int, bool]] = []
        original = MemorySource.get_batch

        def counted(source: MemorySource, indices: Any, **options: Any) -> Batch:
            self.reads.append((id(source), bool(options.get("contiguous", False))))
            return original(source, indices, **options)

        monkeypatch.setattr(MemorySource, "get_batch", counted)


class TestTheHostRead:
    def test_a_batch_named_by_the_given_words_in_the_order_named(self) -> None:
        mix = _pair()
        words = _words([12, 3, 9, 0, 22, 4])
        batch = mix.get_batch(words, epochs=np.arange(6, dtype=np.int32))

        np.testing.assert_array_equal(batch.indices, words)
        np.testing.assert_array_equal(batch.epochs, np.arange(6))
        np.testing.assert_array_equal(batch.draws, np.zeros(6))
        np.testing.assert_array_equal(batch["x"], [103, 3, 100, 0, 113, 4])
        np.testing.assert_array_equal(batch["y"], [3, 3, 0, 0, 13, 4])
        for leaf in jax.tree.leaves(batch):
            assert isinstance(leaf, np.ndarray)

    def test_each_child_is_read_once_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mix = _pair()
        counted = _Counted(monkeypatch)
        mix.get_batch(_words([12, 3, 9, 0, 22, 4, 5]))
        assert sorted(source for source, _ in counted.reads) == sorted(
            id(child) for child in mix.sources
        )

    def test_a_run_is_read_as_each_child_s_views(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mix = _pair()
        counted = _Counted(monkeypatch)
        inside = mix.get_batch(_words([2, 3, 4, 5]), contiguous=True)
        assert np.shares_memory(inside["x"], cast(MemorySource, mix.sources[0]).data["x"])
        across = mix.get_batch(_words([7, 8, 9, 10, 11]), contiguous=True)
        np.testing.assert_array_equal(across["x"], [7, 8, 100, 101, 102])
        assert all(contiguous for _, contiguous in counted.reads)

    def test_a_read_declared_contiguous_that_is_not_a_run_is_refused(self) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            _pair().get_batch(_words([2, 4]), contiguous=True)

    def test_indices_past_two_to_the_32_are_read_from_their_child(self) -> None:
        big = (1 << 32) + 5
        mix = _mix([Sized(big), _memory({"x": 7 + np.arange(3, dtype=np.float32)})])
        words = _words([big + 2, 1 << 32, big])
        batch = mix.get_batch(words)
        np.testing.assert_array_equal(batch["x"], [9.0, float(1 << 32), 7.0])
        np.testing.assert_array_equal(batch.indices, words)

    @pytest.mark.parametrize(
        ("words", "match"),
        [
            (np.full((1, 2), 0xFFFFFFFF, np.uint32), "padding"),
            (_words([23]), "outside"),
            (_words([1 << 32]), "outside"),
        ],
        ids=["padding", "past-the-end", "high-word"],
    )
    def test_words_naming_no_record_are_refused(self, words: np.ndarray, match: str) -> None:
        with pytest.raises(IndexError, match=match):
            _pair().get_batch(words)

    def test_indices_that_are_not_words_are_refused(self) -> None:
        with pytest.raises(ValueError, match="uint32"):
            _pair().get_batch(np.asarray([1, 2], np.int64))

    def test_the_read_creates_no_device_array(self) -> None:
        mix = _pair()
        words = _words([12, 3, 9, 0])
        before = jax.live_arrays()
        batch = mix.get_batch(words)
        assert arrays_made_since(before) == []
        # Control: the instrument sees a read that keeps device arrays (a child's own read over
        # columns held on the device).
        child = cast(MemorySource, mix.sources[0])
        child.data = {name: jnp.asarray(column) for name, column in child.data.items()}
        before = jax.live_arrays()
        kept = child.get_batch(_words([3, 0]))
        assert arrays_made_since(before)
        del batch, kept

    def test_the_read_builds_no_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The union's fields come from the children's reads, not a per-call spec rebuild."""
        mix = _pair()

        def refused(_: Any) -> None:
            raise AssertionError("element_spec rebuilt by the host read")

        monkeypatch.setattr(MixDataSourcesNode, "element_spec", refused)
        batch = mix.get_batch(_words([12, 3]))
        np.testing.assert_array_equal(batch["x"], [103.0, 3.0])

    def test_equal_calls_read_equal_batches_and_change_nothing(self) -> None:
        mix = _pair()
        words = _words([5, 20, 1])
        before = nnx.state(mix)
        first, second = mix.get_batch(words, epochs=2), mix.get_batch(words, epochs=2)
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
            np.testing.assert_array_equal(a, b)
        assert jax.tree.all(jax.tree.map(np.array_equal, before, nnx.state(mix)))
        assert non_array_state_leaves(mix) == []

    def test_a_streaming_disk_child_is_read_on_the_host(self, tmp_path: Any) -> None:
        path = tmp_path / "x.npy"
        np.save(path, 50 + np.arange(6, dtype=np.float32))
        disk = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))
        mix = _mix([disk, _memory({"x": np.arange(4, dtype=np.float32)})])
        batch = mix.get_batch(_words([1, 7, 5]))
        np.testing.assert_array_equal(batch["x"], [51, 1, 55])


@pytest.mark.tfds
def test_a_mix_of_tfds_eager_sources_is_read_on_the_host(tfds_fixture: TFDSFixture) -> None:
    """Two TFDS eager children (int64 host labels, int32 on the device) read through the mix."""
    from datarax.sources import TFDSEagerConfig, TFDSEagerSource  # noqa: PLC0415

    def source() -> TFDSEagerSource:
        directory = str(tfds_fixture.array_record)
        return TFDSEagerSource(TFDSEagerConfig(name=FIXTURE, split="train", data_dir=directory))

    first, second = source(), source()
    mix = _mix([first, second])
    words = np.asarray(mix.record_indices_at(0, 6))
    batch = mix.get_batch(words)
    rows = np.arange(6)
    for child, tfds in enumerate((first, second)):
        mine = rows[rows % 2 == child]
        expected = tfds.get_batch(_words((mine // 2).tolist()))
        served = batch_ops.take(batch, mine)
        for got, want in zip(
            jax.tree.leaves(served.data), jax.tree.leaves(expected.data), strict=True
        ):
            np.testing.assert_array_equal(got, want)
    assert batch["label"].dtype == np.int64
    held = device_spec(jax.tree.map(array_to_spec_strip_leading, batch.data))
    assert held == mix.element_spec()
