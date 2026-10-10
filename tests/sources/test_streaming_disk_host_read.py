"""``StreamingDiskSource`` reads on the host from its memory map (W2b-28, design section 8 item 6).

The host read, ``get_batch(indices, *, epochs=0, contiguous=False)``, takes the eager sources'
signature and checks: it gathers the named rows of the memory map with NumPy (a contiguous run as a
view of the map), names the ``Batch`` with the given words and epochs, and creates no device array.
The traced read, ``get_records``, still serves the compiled session.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs, expect_compiles

from datarax.core.element_batch import Batch, PADDING_INDEX
from datarax.core.index_words import from_words, to_words
from datarax.pipeline import Pipeline
from datarax.sources import EagerSource
from datarax.sources.streaming_disk_source import (
    _HostArray,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.device_arrays import arrays_made_since
from tests.test_common.host_resources import live, mapped, open_descriptors, released


_ROWS = 40
_FEATURES = 3


def _array() -> np.ndarray:
    return np.arange(_ROWS * _FEATURES, dtype=np.float32).reshape(_ROWS, _FEATURES)


@pytest.fixture
def source(tmp_path: Path) -> StreamingDiskSource:
    path = tmp_path / "data.npy"
    np.save(path, _array())
    return StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))


def _words(rows: Sequence[int] | np.ndarray) -> np.ndarray:
    return to_words(np.asarray(rows, dtype=np.uint64))


@pytest.mark.parametrize(
    "rows",
    [[17, 3, 39, 0, 3, 22], [0, 4, 9, 31, 38], list(range(12, 20))],
    ids=["random", "sorted", "contiguous"],
)
def test_the_host_read_equals_the_traced_read_row_for_row(
    source: StreamingDiskSource, rows: list[int]
) -> None:
    words = _words(rows)

    batch = source.get_batch(words)

    assert isinstance(batch, Batch)
    assert isinstance(batch["x"], np.ndarray)
    traced = source.get_records(jnp.asarray(words))
    np.testing.assert_array_equal(batch["x"], np.asarray(traced["x"]))
    np.testing.assert_array_equal(batch["x"], _array()[rows])
    np.testing.assert_array_equal(from_words(np.asarray(batch.indices)), rows)
    assert batch.indices.dtype == np.uint32
    np.testing.assert_array_equal(np.asarray(batch.epochs), np.zeros(len(rows), np.int32))


def test_a_contiguous_run_is_a_view_of_the_memory_map(source: StreamingDiskSource) -> None:
    batch = source.get_batch(_words(range(8, 16)), contiguous=True)

    np.testing.assert_array_equal(batch["x"], _array()[8:16])
    assert np.shares_memory(batch["x"], source._host.array)
    np.testing.assert_array_equal(from_words(np.asarray(batch.indices)), np.arange(8, 16))


def test_a_run_declared_contiguous_that_is_not_one_is_refused(
    source: StreamingDiskSource,
) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        source.get_batch(_words([4, 5, 9]), contiguous=True)


def test_the_epochs_given_name_the_rows(source: StreamingDiskSource) -> None:
    every = source.get_batch(_words([1, 2]), epochs=3)
    each = source.get_batch(_words([1, 2]), epochs=np.asarray([4, 5], np.int32))

    np.testing.assert_array_equal(np.asarray(every.epochs), [3, 3])
    np.testing.assert_array_equal(np.asarray(each.epochs), [4, 5])


def test_a_host_read_creates_no_device_array(source: StreamingDiskSource) -> None:
    words = _words([5, 1, 30, 2])
    source.get_batch(words)  # first use builds nothing on the device either
    before = jax.live_arrays()

    batches = [source.get_batch(words), source.get_batch(_words(range(4)), contiguous=True)]

    assert arrays_made_since(before) == []
    assert all(isinstance(leaf, np.ndarray) for b in batches for leaf in jax.tree.leaves(b))


@pytest.mark.parametrize(
    ("words", "error", "match"),
    [
        (np.full((1, 2), PADDING_INDEX, np.uint32), IndexError, "padding"),
        (_words([3, _ROWS]), IndexError, "outside"),
        (np.asarray([[1, 0]], np.uint32), IndexError, "outside"),
        (np.asarray([1, 2], np.int64), ValueError, "uint32"),
    ],
    ids=["padding", "past-the-end", "high-word", "not-words"],
)
def test_words_naming_no_row_are_refused(
    source: StreamingDiskSource, words: np.ndarray, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        source.get_batch(words)


def test_the_host_read_takes_the_eager_sources_signature() -> None:
    disk = inspect.signature(StreamingDiskSource.get_batch).parameters
    eager = inspect.signature(EagerSource.get_batch).parameters

    assert list(disk) == list(eager)
    assert {name: p.kind for name, p in disk.items()} == {name: p.kind for name, p in eager.items()}


def test_the_source_and_its_config_are_exported() -> None:
    import datarax.sources as sources  # noqa: PLC0415

    assert sources.StreamingDiskSource is StreamingDiskSource
    assert sources.StreamingDiskSourceConfig is StreamingDiskSourceConfig
    assert {"StreamingDiskSource", "StreamingDiskSourceConfig"} <= set(sources.__all__)


def test_the_session_over_the_disk_source_still_compiles_once(
    source: StreamingDiskSource,
) -> None:
    """The traced read is untouched: the compiled session builds one step, then reuses it."""
    from flax import nnx  # noqa: PLC0415

    from datarax.pipeline import Pipeline  # noqa: PLC0415

    session = Pipeline(
        source=source, stages=[], batch_size=8, rngs=nnx.Rngs(0), num_epochs=2
    ).session()
    with expect_first_call_compiles("jit(session_step)"):
        first = next(session)
        jax.block_until_ready(first.indices)
    with expect_compiles(0):
        rest = list(session)
    served = np.concatenate([np.asarray(b["x"]) for b in (first, *rest)])
    np.testing.assert_array_equal(served, np.concatenate([_array(), _array()]))


class TestNothingOfADroppedSourceIsKept:
    """A dropped pipeline leaves no memory map, mapping or descriptor of its file.

    The host naming caches a program per source structure; it keeps the structure (the record
    count, the configuration), never the source's memory map.
    """

    def test_no_map_is_left(self, tmp_path: Path) -> None:
        path = tmp_path / "data.npy"
        np.save(path, _array())
        source = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))
        pipe = Pipeline(
            source=source, stages=[], batch_size=8, rngs=nnx.Rngs(0), shuffle=True, num_epochs=1
        )
        assert len(list(pipe.raw_batches())) == 5
        pipe.close()
        del pipe, source

        def holders() -> int:
            return live(lambda item: type(item) is _HostArray and item.path == str(path))

        assert released(
            lambda: holders() == 0 and not mapped([path]) and not open_descriptors([path])
        ), (holders(), mapped([path]), open_descriptors([path]))

    def test_two_sources_of_one_file_compile_the_naming_once(self, tmp_path: Path) -> None:
        path = tmp_path / "data.npy"
        np.save(path, _array())

        def run() -> None:
            source = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))
            pipe = Pipeline(
                source=source,
                stages=[],
                batch_size=8,
                rngs=nnx.Rngs(0),
                shuffle=True,
                num_epochs=1,
            )
            list(pipe.raw_batches())
            pipe.close()

        jax.clear_caches()
        with compiled_programs() as programs:
            run()
            run()
        assert sum(str(program).startswith("jit(_names") for program in programs) == 1
