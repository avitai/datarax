"""A stream over records held in memory, for tests of the stream contract and the stream route.

:class:`RecordStream` builds on :class:`~datarax.sources.StreamingSourceBase` like every stream: it
reads one pass at a time, in its records' order or, given the pipeline's key, in a permutation of
them drawn from ``fold_in(key, pass)``, in chunks of a fixed size. It names records by their
position (``STREAM_IDS``) or by arrival (``ARRIVAL``), and carries a text field as provenance.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import RecordIdentity
from datarax.sources import StreamChunk, StreamingSourceBase
from datarax.sources._source_base import pass_generator
from datarax.sources.eager_source import HostValue


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
        chunk: int = 3,
    ) -> None:
        """Hold ``columns`` (one row per record) and, if given, one text per record.

        Args:
            columns: Host arrays sharing a leading record axis.
            kind: ``STREAM_IDS`` (records named by position) or ``ARRIVAL``.
            texts: Each record's text, kept as its provenance.
            chunk: Records per chunk the pass yields.
        """
        super().__init__(RecordStreamConfig())
        self._columns = nnx.data({name: np.asarray(column) for name, column in columns.items()})
        self._size = len(next(iter(self._columns.values())))
        self._texts = HostValue(texts)
        self._kind = kind
        self._chunk = chunk
        self._opened = HostValue([])

    @property
    def opened(self) -> list[tuple[int, jax.Array | None]]:
        """The passes ``_open_pass`` was asked for, in order, and the key each was given."""
        return self._opened.value

    @property
    def record_identity(self) -> RecordIdentity:
        """The kind the stream was built with."""
        return self._kind

    def __len__(self) -> int:
        """The records one pass serves."""
        return self._size

    def _open_pass(
        self, pass_index: int, key: jax.Array | None, size_hint: int
    ) -> Iterator[StreamChunk]:
        del size_hint
        self._opened.value.append((pass_index, key))
        order = np.arange(self._size)
        if key is not None:
            order = pass_generator(key, pass_index).permutation(self._size)
        for start in range(0, self._size, self._chunk):
            rows = order[start : start + self._chunk]
            provenance: tuple[Mapping[str, Any], ...] = tuple(
                MappingProxyType(
                    {} if self._texts.value is None else {"text": self._texts.value[r]}
                )
                for r in rows
            )
            yield StreamChunk(
                {name: column[rows] for name, column in self._columns.items()},
                provenance,
                rows.astype(np.uint64) if self._kind is RecordIdentity.STREAM_IDS else None,
            )
