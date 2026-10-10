"""An indexed source over ArrayRecord files, read in batches on the host."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import jax
import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from jax.typing import ArrayLike

from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    host_rows,
    NO_PROVENANCE,
    record_words,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.sources.eager_source import named_batch, parts_of_records, run_words


type RecordDecoder = Callable[[Sequence[bytes]], Sequence[Mapping[str, Any]]]
"""Turns a batch's ``bytes`` records into one mapping of values per record, in one call."""


@dataclass(frozen=True)
class ArrayRecordSourceConfig(StructuralConfig):
    """Configuration for ``ArrayRecordSourceModule``.

    Attributes:
        local_files_only: If True, every path is checked to exist at construction, and a missing
            one is refused with ``FileNotFoundError`` naming it. ArrayRecord sources never
            download, so this replaces the reader's lower-level error with one naming the paths.
    """

    local_files_only: bool = False


class _Records:
    """ArrayRecord files read by position, kept on the host and out of NNX state.

    ArrayRecord's data source (the reader Grain and TFDS read the format with) opens each file at
    its first read, under its own lock, so threads reading one source at once share one reader
    per file and read in parallel. It pickles without open readers, so a copy sent to a worker
    process reopens the files where it reads them.
    """

    __slots__ = ("paths", "source")

    def __init__(self, paths: Any) -> None:
        self.paths = paths
        self.source = ArrayRecordDataSource(paths)

    def read(self, rows: Sequence[int]) -> Sequence[bytes]:
        """The records at ``rows``, in one batched read (a parallel read per file)."""
        return self.source.__getitems__(rows)

    def close(self) -> None:
        """Close the open readers; the files reopen at the next read."""
        self.source.__exit__(None, None, None)


def _path_of(path: Any) -> str:
    """A path's file name: the path itself, or a ``FileInstruction``'s ``filename``."""
    return str(getattr(path, "filename", path))


class ArrayRecordSourceModule(DataSourceModule):
    """An indexed source over ArrayRecord files: a record's index is its position in the files.

    ArrayRecord records are ``bytes``. A batch's records are read with one batched read of
    ArrayRecord's ``ArrayRecordDataSource`` (a parallel read per file) and decoded with one call of
    ``decode``, which returns one mapping of values per record. Numeric values become the
    batch's columns and every other value the record's provenance, as in the eager sources
    (:func:`~datarax.sources.eager_source.parts_of_records`). ``paths`` may be file paths or
    ``FileInstruction`` s, which read a part of each file (TFDS's split slices).

    The source holds no order, epoch or cursor: the pipeline that serves it names the records
    of each batch (``Pipeline(shuffle=...)``) and its host stage reads them with
    :meth:`get_batch`. It has no traced read, so a compiled ``step()`` over it is refused.

    ArrayRecord readers hold file handles that garbage collection does not reliably release;
    use the source as a context manager or call :meth:`close` between phases that open new
    sources, to avoid "Too many open files" on long-running jobs.
    """

    config: ArrayRecordSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def record_identity(self) -> RecordIdentity:
        """A record's index is its position in the files, which the pipeline orders."""
        return RecordIdentity.INDEXED

    def __init__(
        self,
        config: ArrayRecordSourceConfig,
        paths: Any,
        *,
        decode: RecordDecoder,
        name: str | None = None,
    ) -> None:
        """Open the ArrayRecord files, reading no record.

        Args:
            config: Configuration for the source.
            paths: A path or ``FileInstruction``, or a sequence of them.
            decode: Turns a batch's ``bytes`` records into one mapping of values per record.
            name: Optional name for the module.

        Raises:
            FileNotFoundError: If ``local_files_only`` is set and a path does not exist.
        """
        super().__init__(config, name=name)
        if config.local_files_only:
            listed = [paths] if isinstance(paths, str) or not isinstance(paths, Sequence) else paths
            missing = [_path_of(p) for p in listed if not Path(_path_of(p)).exists()]
            if missing:
                raise FileNotFoundError(
                    f"ArrayRecordSourceModule: local_files_only=True but the following path(s) "
                    f"do not exist: {missing}. Either populate the paths or set "
                    "local_files_only=False to defer the error to the reader."
                )
        self._records = _Records(paths)
        self._decode = decode
        # The record count as a plain static: the source's order reads it, never the reader.
        self._length = len(self._records.source)

    def __len__(self) -> int:
        """The number of records in the files."""
        return self._length

    def __repr__(self) -> str:
        """The files and the record count, which identify the records a checkpoint names."""
        return f"ArrayRecordSourceModule(paths={self._records.paths!r}, num_records={len(self)})"

    def _parts(self, rows: np.ndarray) -> tuple[Any, tuple[dict[str, Any], ...]]:
        """The columns and provenance of the records at ``rows``: one read, one decode call."""
        records = self._records.read([int(row) for row in rows])
        return parts_of_records(self._decode(records))

    def get_batch(  # noqa: DOC502 - record_words, host_rows and run_words raise
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read and decode the records ``indices`` names, as a ``Batch`` named with them.

        One batched read of the named records and one call of ``decode``; the batch's
        ``indices`` are the given words, its ``epochs`` the given epochs and its draws 0. It
        reads and changes no state and creates no device array. With ``contiguous=True`` the
        caller states that ``indices`` is a run of consecutive records; only its ends are
        checked, and its rows are named as the run.

        Args:
            indices: uint32 ``(n, 2)`` record indices, each as its words ``(hi, lo)``.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records.

        Returns:
            The records' numeric values as a host ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, a run declared
                contiguous is not one, or the decoded records' numeric fields or shapes differ.
            IndexError: If an index is the padding index or outside the files.
        """
        return self.read_with_provenance(indices, epochs=epochs, contiguous=contiguous)[0]

    def read_with_provenance(  # noqa: DOC502 - record_words, host_rows and run_words raise
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> tuple[Batch, tuple[Mapping[str, Any], ...]]:
        """Read and decode the records ``indices`` names once: the batch and their provenance.

        The batch is :meth:`get_batch`'s and the provenance :meth:`provenance`'s, from one batched
        read and one call of ``decode``
        (:class:`~datarax.core.data_source.IndexedHostReadWithProvenance`).

        Args:
            indices: uint32 ``(n, 2)`` record indices, each as its words ``(hi, lo)``.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records.

        Returns:
            The records' numeric values as a host ``Batch``, and each record's mapping of strings
            and objects, in row order; an empty mapping for a record that holds nothing but
            numbers.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, a run declared
                contiguous is not one, or the decoded records' numeric fields or shapes differ.
            IndexError: If an index is the padding index or outside the files.
        """
        words = record_words(indices)
        rows = host_rows(words, len(self))
        if contiguous and len(rows):
            words = run_words(rows)
        columns, provenance = self._parts(rows)
        return named_batch(columns, words, epochs), provenance or (NO_PROVENANCE,) * len(rows)

    def provenance(  # noqa: DOC502 - record_words and host_rows raise
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """The non-numeric values of the records ``indices`` names, read and decoded.

        Args:
            indices: uint32 ``(n, 2)`` record indices.

        Returns:
            Each named record's mapping of strings and objects, in the order named; an empty
            mapping for a record that holds nothing but numbers.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index is the padding index or outside the files.
        """
        rows = host_rows(record_words(indices), len(self))
        _, provenance = self._parts(rows)
        return provenance or (NO_PROVENANCE,) * len(rows)

    def element_spec(self) -> Any:
        """The spec of one decoded record's numbers as the device holds them (record 0's).

        Returns:
            A pytree of ``jax.ShapeDtypeStruct``, the dtypes stated as JAX arrays hold them.
        """
        columns, _ = self._parts(np.zeros(1, np.int64))
        return device_spec(jax.tree.map(array_to_spec_strip_leading, columns))

    def close(self) -> None:
        """Release the ArrayRecord file handles; the files reopen at the next read.

        ArrayRecord readers hold file handles that garbage collection does not reliably release,
        so long-running jobs that create sources repeatedly can exhaust the descriptor limit.
        Safe to call more than once. Call it between phases, once no read of the source is in
        flight: like ArrayRecord's own ``__exit__``, it does not wait for a read to finish.
        """
        self._records.close()

    def __enter__(self) -> Self:
        """Enter a context that guarantees ``close()`` on exit."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Release ArrayRecord file handles on context exit."""
        self.close()


__all__ = ["ArrayRecordSourceConfig", "ArrayRecordSourceModule", "RecordDecoder"]
