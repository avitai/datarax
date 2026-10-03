"""A stream over records held in memory, for tests of the stream contract and the stream route.

:class:`RecordStream` builds on :class:`~datarax.sources.StreamingSourceBase` like every stream: it
reads one pass at a time, in its records' order or, given the pipeline's key, in a permutation of
them drawn from ``fold_in(key, pass)``, in chunks of a fixed size or of the pass's read size. It
names records by their position (``STREAM_IDS``) or by arrival (``ARRIVAL``), and carries a text
field as provenance.

The checks below hold for every stream and are run on each (brief T11): the chunks a stream's
passes yield, no Python value in a Variable, a graph definition that does not move as the stream
advances, and a tree-mode split and merge that reads on from the same place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import RecordIdentity
from datarax.sources import StreamChunk, StreamingSourceBase
from datarax.sources._source_base import chunk_size, pass_generator
from datarax.sources.eager_source import HostValue


# The provenance of a record without text: read-only, so every such record shares it.
_NO_TEXT: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen=True)
class RecordStreamConfig(StructuralConfig):
    """No settings: the records and the kind are constructor arguments."""


class RecordStream(StreamingSourceBase):
    """A stream of in-memory records, read in chunks."""

    def __init__(
        self,
        columns: Mapping[str, np.ndarray],
        *,
        kind: RecordIdentity = RecordIdentity.STREAM_IDS,
        texts: list[str] | None = None,
        chunk: int | None = 3,
    ) -> None:
        """Hold ``columns`` (one row per record) and, if given, one text per record.

        Args:
            columns: Host arrays sharing a leading record axis.
            kind: ``STREAM_IDS`` (records named by position) or ``ARRIVAL``.
            texts: Each record's text, kept as its provenance.
            chunk: Records per chunk the pass yields, or ``None`` for the pass's read size.
        """
        super().__init__(RecordStreamConfig())
        self._columns = nnx.data({name: np.asarray(column) for name, column in columns.items()})
        self._size = len(next(iter(self._columns.values())))
        self._texts = HostValue(texts)
        self._kind = kind
        self._chunk = chunk
        self._opened = HostValue([])
        self._read_sizes = HostValue([])

    @property
    def opened(self) -> list[tuple[int, np.ndarray | None]]:
        """The passes ``_open_pass`` was asked for, in order, and the key words each was given."""
        return self._opened.value

    @property
    def read_sizes(self) -> list[int]:
        """The read size each pass was opened with, in order."""
        return self._read_sizes.value

    @property
    def record_identity(self) -> RecordIdentity:
        """The kind the stream was built with."""
        return self._kind

    def __len__(self) -> int:
        """The records one pass serves."""
        return self._size

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        self._opened.value.append((pass_index, key))
        self._read_sizes.value.append(read_size)
        chunk = read_size if self._chunk is None else self._chunk
        order = np.arange(self._size)
        if key is not None:
            order = pass_generator(key, pass_index).permutation(self._size)
        for start in range(0, self._size, chunk):
            rows = order[start : start + chunk]
            texts = self._texts.value
            provenance: tuple[Mapping[str, Any], ...] = (
                (_NO_TEXT,) * len(rows)
                if texts is None
                else tuple(MappingProxyType({"text": texts[r]}) for r in rows)
            )
            yield StreamChunk(
                {name: column[rows] for name, column in self._columns.items()},
                provenance,
                rows.astype(np.uint64) if self._kind is RecordIdentity.STREAM_IDS else None,
            )


def record_chunks(
    monkeypatch: pytest.MonkeyPatch, stream_type: type[StreamingSourceBase]
) -> list[tuple[int, int, int]]:
    """Record ``(pass, read size, records)`` of every chunk ``stream_type``'s passes yield.

    Args:
        monkeypatch: Restores the stream type's ``_open_pass`` after the test.
        stream_type: The stream class to watch.

    Returns:
        The list the chunks are recorded in, as they are read.
    """
    seen: list[tuple[int, int, int]] = []
    original = stream_type._open_pass  # noqa: SLF001 - the hook every stream implements

    def watched(
        self: StreamingSourceBase, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        for chunk in original(self, pass_index, key, read_size):
            seen.append((pass_index, read_size, chunk_size(chunk)))
            yield chunk

    monkeypatch.setattr(stream_type, "_open_pass", watched)
    return seen


def non_array_state_leaves(module: nnx.Module) -> list[type]:
    """The types of a module's NNX state leaves that are not arrays: none, for a source."""
    leaves = jax.tree.leaves(nnx.state(module))
    return [type(leaf) for leaf in leaves if not isinstance(leaf, np.ndarray | jax.Array)]


def graph_definitions_across_a_pass(stream: StreamingSourceBase, size: int) -> tuple[Any, Any]:
    """The stream's graph definition before and after a keyed pull and a whole pass."""
    before = nnx.graphdef(stream)
    stream.get_batch(size, key=jax.random.key(0))
    while stream.get_batch(size).batch_size:
        pass
    return before, nnx.graphdef(stream)


def second_pulls_after_a_tree_round_trip(
    make: Callable[[], StreamingSourceBase], size: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The second pull of a stream split and merged in tree mode after its first, and of a twin.

    Args:
        make: Builds the stream.
        size: Records per pull.

    Returns:
        The two second pulls, each as its ids and its array leaves.
    """
    stream, twin = make(), make()
    stream.get_batch(size)
    twin.get_batch(size)
    graphdef, state = nnx.split(stream, graph=False)
    merged = nnx.merge(graphdef, state)

    def leaves(batch: Any) -> list[np.ndarray]:
        return [np.asarray(batch.indices), *(np.asarray(x) for x in jax.tree.leaves(batch.data))]

    return leaves(merged.get_batch(size)), leaves(twin.get_batch(size))
