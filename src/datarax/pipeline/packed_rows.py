"""Packed rows: a source's variable-length records planned into rows, served as an indexed source.

``Pipeline(packing=...)`` plans a ragged source's records once into rows of a static length
(:mod:`datarax.core.pack_plan`) and serves the rows, not the records. :class:`PackedRows` is an
``INDEXED`` source whose record ``r`` is row ``r`` of the plan, as levanter's
``GreedyPrepackedDataset`` is a dataset of its packed examples (``levanter/data/packing.py``) over
which a permutation orders each epoch (``levanter/main/sft.py``'s ``PermutationDataset``). The host
stage therefore names, orders, batches, resumes and shards rows with the machinery it serves every
indexed source with: the order is the base cipher over the rows, a new one each epoch. A row's
host read takes its records from the plan, reads every packed field through the source's ragged
read and lays the rows out with :func:`~datarax.core.batch_ops.pack_rows`; a row is named
``(2**31, row)``, the index of no record (:mod:`datarax.core.index_words`), and its provenance is
one mapping per slot.

The planning order is sequential without a shuffle, and with one the pipeline's cipher over the
records under its key folded with :data:`PLANNING_STREAM`, as the block order folds its named
streams into its key (:mod:`datarax.core.block_order`). The plan is kept in a decoded copy
(``decoded_copy._plans.copy_plan``), planned once for every pipeline built alike, or in a
temporary directory removed when nothing reads the plan any more: its pipeline and that
pipeline's runs.
"""

from __future__ import annotations

import dataclasses
import functools
import shutil
import tempfile
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax.typing import ArrayLike

from datarax.core import batch_ops
from datarax.core.block_order import order_positions
from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    host_rows,
    known_length,
    NO_PROVENANCE,
    PackedProvenance,
    record_words,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.core.host_values import HostValue
from datarax.core.index_words import from_words, to_words
from datarax.core.pack_plan import PackPlan, plan_rows
from datarax.core.prng import host_device, NAMING_PRNG_IMPL
from datarax.core.spec import device_spec
from datarax.pipeline.run_configuration import Packing
from datarax.sources._field_paths import nested
from datarax.sources.decoded_copy._plans import copy_plan
from datarax.sources.eager_source import named_batch
from datarax.sources.streaming_disk_source import StreamingDiskSource
from datarax.typing import DataDict


PLANNING_STREAM = np.uint32(2**32 - 1)
"""``fold_in`` datum of the planning order's key. Every epoch's key is ``fold_in(key, epoch)`` for
a non-negative int32 epoch, below ``2**31``, so no epoch's order is the planning order."""

_ROW_BIT = np.uint32(2**31)
"""The top bit of a row's name, ``(2**31, row)``: an index with it set names no record."""


@functools.cache
def _order() -> Callable[..., jax.Array]:
    """``order_positions`` compiled for the host device, built at first use (no JAX at import)."""
    return jax.jit(order_positions, static_argnames=("length", "block"))


def refuse_unpackable(source: DataSourceModule, packing: Packing) -> None:  # noqa: DOC503 - field_names raises the NotImplementedError and the TypeError
    """Refuse a packing that does not name every field of the source, or names one it lacks.

    Every field is packed, as levanter packs every leaf of its dataset
    (``GreedyPrepackedDataset``'s ``max_length`` per leaf) and Grain's ``length_struct`` names
    every feature: a scalar field takes one cell a record.

    Args:
        source: The source to pack, serving the ragged read.
        packing: The packing asked for.

    Raises:
        NotImplementedError: If the source does not read ragged records.
        TypeError: If the source is a stream.
        ValueError: If a field is missing from the packing or unknown to the source, naming it.
    """
    names = set(source.field_names)
    missing = sorted(names - set(packing.lengths))
    unknown = sorted(set(packing.lengths) - names)
    if missing or unknown:
        raise ValueError(
            f"packing packs every field of {type(source).__name__}, whose fields are "
            f"{sorted(names)}: give a row length for {missing}"
            + (f"; unknown: {unknown}" if unknown else "")
            + " (a scalar field takes one cell a record, so its row length is max_segments)"
        )


def planning_order(length: int, key_data: ArrayLike | None) -> Callable[[int, int], np.ndarray]:
    """The records at positions ``[start, stop)`` of the planning order.

    Args:
        length: The source's records.
        key_data: The pipeline's key data, or ``None`` when it does not shuffle.

    Returns:
        A function of ``(start, stop)`` giving uint64 record indices.
    """
    if key_data is None:
        return lambda start, stop: np.arange(start, stop, dtype=np.uint64)
    device = host_device()
    words = jax.device_put(np.asarray(key_data), device)
    key = jax.random.fold_in(
        jax.random.wrap_key_data(words, impl=NAMING_PRNG_IMPL), PLANNING_STREAM
    )

    def records_at(start: int, stop: int) -> np.ndarray:
        positions = jax.device_put(to_words(np.arange(start, stop, dtype=np.uint64)), device)
        return from_words(np.asarray(jax.device_get(_order()(positions, length, key, None))))

    return records_at


def _copy_directory(source: DataSourceModule) -> Path | None:
    """The decoded copy's directory when ``source`` is a decoded copy, where its plans are kept."""
    if isinstance(source, StreamingDiskSource) and source.config.copy_directory is not None:
        return Path(source.config.copy_directory)
    return None


def plan_packing(
    source: DataSourceModule, packing: Packing, key_data: ArrayLike | None
) -> tuple[PackPlan, Path | None]:
    """Plan ``source``'s records into rows: kept in its decoded copy, else in a new directory.

    Args:
        source: The source, serving the ragged read.
        packing: The rows' layout and the window.
        key_data: The pipeline's key data, or ``None`` when it does not shuffle.

    Returns:
        The plan, and the temporary directory holding it (to be removed with its readers), or
        ``None`` when it is kept in a decoded copy.
    """
    length = known_length(source) or 0
    records_at = planning_order(length, key_data)
    fields = tuple(packing.lengths)

    def lengths_of(records: np.ndarray) -> Mapping[str, np.ndarray]:
        words = to_words(records)
        return {field: source.lengths(words, field) for field in fields}

    def plan(directory: Path) -> PackPlan:
        return plan_rows(
            length=length,
            records_at=records_at,
            lengths_of=lengths_of,
            row_lengths=packing.lengths,
            max_segments=packing.max_segments,
            window=packing.window,
            directory=directory,
        )

    copy = _copy_directory(source)
    if copy is not None:
        key = None if key_data is None else np.asarray(key_data).tolist()
        return copy_plan(
            copy, parameters={"packing": packing.as_plain(), "key": key}, plan=plan
        ), None
    temporary = Path(tempfile.mkdtemp(prefix="datarax-pack-plan-"))
    return plan(temporary), temporary


@dataclasses.dataclass(frozen=True)
class PackedRowsConfig(StructuralConfig):
    """The rows' source takes no configuration: its source, packing and plan are its arguments."""


class PackedRows(DataSourceModule):
    """The rows of a pack plan over a ragged source, as an ``INDEXED`` source of ``plan.rows``.

    A row names no record: its provenance is its slots' (:meth:`row_provenance`,
    :class:`~datarax.core.data_source.PackedRowProvenance`), and the record lookup by index is
    refused.
    """

    config: PackedRowsConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        source: DataSourceModule,
        packing: Packing,
        plan: PackPlan,
        *,
        temporary: Path | None = None,
    ) -> None:
        """Serve ``plan``'s rows of ``source``'s records.

        Args:
            source: The packed source, serving the ragged read.
            packing: The rows' layout.
            plan: The rows.
            temporary: The plan's directory when nothing else keeps it: removed when this source
                is garbage-collected or the process exits.
        """
        super().__init__(PackedRowsConfig())
        self.source = source
        self.packing = nnx.static(packing)
        self._plan = HostValue(plan)
        # Plain values beside the holder: the host naming's stand-in of this source keeps them.
        self._rows = nnx.static(plan.rows)
        self._lengths_digest = nnx.static(plan.lengths_digest)
        if temporary is not None:
            # The directory is this pipeline's own mkdtemp; one the system's temporary cleaning
            # already removed leaves nothing to report.
            weakref.finalize(self, shutil.rmtree, temporary, ignore_errors=True)

    @property
    def record_identity(self) -> RecordIdentity:
        """A row's index is its place in the plan."""
        return RecordIdentity.INDEXED

    @property
    def host_read(self) -> Any:
        """The packed source's: reading a row reads its records."""
        return self.source.host_read

    @property
    def plan(self) -> PackPlan:
        """The rows' plan, read through memory maps."""
        return self._plan.value

    @property
    def lengths_digest(self) -> str:
        """The SHA-256 of the lengths the rows were planned from."""
        return self._lengths_digest

    def __len__(self) -> int:
        """The rows."""
        return self._rows

    def __repr__(self) -> str:
        """The packed source and the rows' layout, for checkpoint validation."""
        return f"PackedRows(source={self.source!r}, packing={self.packing}, rows={self._rows})"

    def element_spec(self) -> Any:
        """One row's spec: each packed field's row and its segment ids, positions and identities.

        Returns:
            Nested ``jax.ShapeDtypeStruct`` per leaf, as the device holds it.
        """
        first = to_words(np.zeros(1, np.uint64))
        leaves: dict[str, jax.ShapeDtypeStruct] = {}
        for field, row_length in self.packing.lengths.items():
            values, _ = self.source.get_ragged(first, field)
            leaves[field] = jax.ShapeDtypeStruct((row_length, *values.shape[1:]), values.dtype)
            leaves[f"{field}_segment_ids"] = jax.ShapeDtypeStruct((row_length,), np.int32)
            leaves[f"{field}_positions"] = jax.ShapeDtypeStruct((row_length,), np.int32)
            leaves[f"{field}_segment_indices"] = jax.ShapeDtypeStruct(
                (self.packing.max_segments, 2), np.uint32
            )
        return device_spec(nested(leaves))

    def _records_of(self, indices: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The named rows, their records as words, and each row's start into the records."""
        rows = host_rows(record_words(indices), self._rows)
        records, starts = self.plan.take(rows)
        return rows, to_words(records), starts

    def get_batch(  # noqa: DOC502 - the index checks raise
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """The named rows, packed: every field's row, segment ids, positions and identities.

        Args:
            indices: uint32 ``(n, 2)`` row indices.
            epochs: The epoch of every row, or of each ``(n,)``.
            contiguous: Unused: a row's records are wherever the plan put them.

        Returns:
            The rows as a host ``Batch``, each named ``(2**31, row)``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index names no row.
        """
        del contiguous
        rows, words, starts = self._records_of(indices)
        values, offsets = {}, {}
        for field in self.packing.lengths:
            values[field], offsets[field] = self.source.get_ragged(words, field)
        packed = batch_ops.pack_rows(
            values,
            offsets,
            words,
            starts,
            row_lengths=self.packing.lengths,
            max_segments=self.packing.max_segments,
        )
        names = to_words(rows.astype(np.uint64))
        names[:, 0] |= _ROW_BIT
        return named_batch(nested(packed), names, epochs)

    def provenance(  # noqa: DOC201 - it never returns
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """Refused: a row names no record, so no record's provenance is found by its index.

        Args:
            indices: Unused.

        Raises:
            TypeError: Always, naming :meth:`row_provenance`.
        """
        del indices
        raise TypeError(
            "a packed row names no record, so no record's provenance is found by its index: "
            "`row_provenance(indices)` serves each row's slots' provenance"
        )

    def row_provenance(  # noqa: DOC502 - the index checks raise
        self, indices: ArrayLike
    ) -> PackedProvenance:
        """Each named row's records' provenance: one mapping per slot, empty past its records.

        Args:
            indices: uint32 ``(n, 2)`` row indices.

        Returns:
            One tuple of ``max_segments`` mappings per row.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index names no row.
        """
        _, words, starts = self._records_of(indices)
        records = self.source.provenance(words)
        slots = self.packing.max_segments
        return tuple(
            tuple(records[starts[k] : starts[k + 1]])
            + (NO_PROVENANCE,) * (slots - int(starts[k + 1] - starts[k]))
            for k in range(len(starts) - 1)
        )

    def get_records(self, indices: jax.Array) -> DataDict:  # noqa: DOC201 - it never returns
        """Refused: packed rows are read on the host, by iteration.

        Args:
            indices: Unused.

        Raises:
            ValueError: Always, naming iteration.
        """
        del indices
        raise ValueError(
            "packed rows are planned and laid out on the host and served by iteration "
            "(`for batch in pipe` or `pipe.raw_batches()`), not by the compiled step"
        )


__all__ = ["PLANNING_STREAM", "PackedRows", "plan_packing", "planning_order", "refuse_unpackable"]
