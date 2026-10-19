"""Decode a source's records once, into RAM or as a decoded copy, through the budget's workers.

``materialize(source, ...)`` reads an indexed source whose read decodes every record on every
epoch (a TFDS split read per batch, ArrayRecord files of encoded images) once, and returns a
source that serves the same records, record for record, without decoding them again:

- without ``directory``, a ``MemorySource`` over the decoded columns (``MemorySource.of_columns``),
  held on the host out of NNX state as every eager source's columns are. They are O(dataset) host
  memory, a term of the budget (the plan's ``C``);
- with ``directory``, the source's decoded copy there (:mod:`datarax.sources.decoded_copy`),
  opened as a ``StreamingDiskSource``: reused when the directory holds a complete copy of this
  source, else written, in place, one process at a time.

Every record is decoded by the one per-unit writer (``write_unit``), on the calling thread or in
the training process's Grain worker processes. Which runs it follows the read, as for a run of a
pipeline: a read that holds the GIL (``HostRead.GIL_BOUND``) is decoded by the workers a RAM budget
admits (``host_resources``), and only from two of them, since one worker is slower than the
calling thread; a GIL-free read (columns, memory maps) on the calling thread, which outpaces any
worker count. Without a budget, the calling thread decodes. Workers write a copy's fixed rows in
place and send back each unit's provenance and digest, so no fixed row crosses ``/dev/shm``; a
ragged field's total is known only once decoded, so its values and offsets are sent back and the
training process appends them in unit order. For RAM workers send each unit's decoded arrays,
which the training process copies into its columns; a source holding a ragged field is written
into a copy only.

Nothing is placed on a device and nothing is traced: the result is a source for a ``Pipeline``,
whose host stage reads it. The argument is never changed.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import grain
import numpy as np

from datarax.core.data_source import (
    DataSourceModule,
    has_indexed_host_read,
    HostRead,
    known_length,
    RecordIdentity,
    SourceIdentified,
)
from datarax.core.host_resources import HostResources
from datarax.core.ragged import RaggedColumn
from datarax.pipeline import host_workers
from datarax.pipeline.host_workers import (
    check_all_arrived,
    check_arrival,
    crossing_provenance,
    in_workers,
    JaxSettings,
)
from datarax.pipeline.read_plan import (
    budget_plan,
    HostReadPath,
    HostTerms,
    measure_terms,
    refuse_below_fixed,
    warn_one_thread,
    WORKER_BUFFER,
)
from datarax.sources.decoded_copy import open_decoded_copy
from datarax.sources.decoded_copy._files import (
    begin_copy,
    clear_copy,
    CopyFiles,
    CopyLock,
    finish_copy,
    refuse_unsupported_filesystem,
)
from datarax.sources.decoded_copy._layout import ColumnLayout, unit_count, UNIT_RECORDS, unit_rows
from datarax.sources.decoded_copy._manifest import identity_differences, read_manifest
from datarax.sources.decoded_copy._units import (
    first_read,
    in_layout_order,
    ragged_parts,
    read_unit,
    refuse_ragged_in_memory,
    unit_digest,
    UnitRead,
    write_unit,
)
from datarax.sources.memory_source import MemorySource
from datarax.sources.streaming_disk_source import StreamingDiskSource


logger = logging.getLogger(__name__)

_FEWEST_WORKERS = 2
"""Fewer workers than this decode slower than the calling thread (one worker loses to it)."""


def materialize(  # noqa: DOC503 - begin_copy raises the OSError
    source: DataSourceModule,
    *,
    host_resources: HostResources | None = None,
    directory: str | Path | None = None,
    unit_records: int = UNIT_RECORDS,
    rebuild: bool = False,
) -> MemorySource | StreamingDiskSource:
    """Decode ``source``'s records once: into RAM, or as its decoded copy in ``directory``.

    Without ``directory`` the result is a ``MemorySource`` whose record ``r`` is ``source``'s
    record ``r``: O(dataset) host memory, held out of NNX state, so a transform of a module
    holding it transfers nothing. With ``directory`` it is the copy, a ``StreamingDiskSource``:
    a complete copy of this source already there is returned as it is (checked by what the
    source declares, never by paths, and by file sizes); otherwise, or with ``rebuild``, the
    copy is written under ``directory``'s lock (a process that waited for another's write returns
    the finished copy) on a local disk, its old files removed manifest first. N training processes
    on one host calling it on one directory build once.

    With ``host_resources`` the build is planned like a pipeline's run and refused when the
    budget is below what the training process holds before decoding, ``M_main + C + E``
    (``C``: the decoded columns in RAM, 0 for a copy); ``from_tfds(..., in_memory=True,
    host_resources=...)`` checks the split's decoded size alone, so a budget can pass there and
    be refused here.

    Args:
        source: An ``INDEXED`` source with the stateless host read, whose index space is its
            length. A copy also needs it to declare its identity (``source_identity()``).
        host_resources: The training process's RAM budget and cap, or ``None`` to decode on the
            calling thread.
        directory: Where the decoded copy is, or is written, or ``None`` for RAM.
        unit_records: Records decoded at a time.
        rebuild: Whether a matching copy in ``directory`` is written again.

    Returns:
        A source serving ``source``'s records without decoding them.

    Raises:
        TypeError: If ``source`` is not indexed with the stateless host read, or a copy is asked
            of a source that declares no identity.
        ValueError: If the budget is below what the build holds, ``directory`` is on a networked
            or FUSE filesystem or holds files no copy writes, or a record is unlike the first.
        OSError: If the copy's files cannot be reserved (a full disk).
        NotImplementedError: If a source holding a ragged field is asked into RAM.
    """
    _refuse_unbuildable(source)
    if unit_records < 1:
        raise ValueError(f"unit_records is at least 1; got {unit_records}")
    if directory is None:
        refuse_ragged_in_memory(source)
        return _into_memory(source, host_resources, unit_records)
    if not isinstance(source, SourceIdentified):
        raise TypeError(
            f"{type(source).__name__} declares no identity (source_identity()), so no decoded "
            "copy of it can be checked against it: an in-memory source has no origin a copy "
            "would stand for; materialize it into RAM (directory=None)"
        )
    return _onto_disk(source, Path(directory), host_resources, unit_records, rebuild=rebuild)


def _refuse_unbuildable(source: DataSourceModule) -> None:
    """Refuse a source ``materialize`` cannot read record by record.

    Args:
        source: The source asked of ``materialize``.

    Raises:
        TypeError: If the source is not ``INDEXED`` with the stateless host read, or its index
            space is not its length (a sharded source).
    """
    if source.record_identity is not RecordIdentity.INDEXED or not has_indexed_host_read(source):
        raise TypeError(
            f"materialize reads an INDEXED source by index with its stateless host read "
            f"(get_batch); {type(source).__name__} names its records "
            f"{source.record_identity.value}"
        )
    length = known_length(source)
    if source.index_space() != length:
        raise TypeError(
            f"{type(source).__name__} serves {length} of {source.index_space()} indices (a "
            "shard): materialize decodes a whole source, record r being index r"
        )


# ---------------------------------------------------------------------------------------------
# What workers run
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class _WriteUnit:
    """A worker's task for a copy: write a unit's fixed rows in place; send back the rest.

    Attributes:
        source: The source.
        files: The copy's reserved ``.partial`` files.
    """

    source: DataSourceModule
    files: CopyFiles

    def __call__(
        self, unit: int, read: UnitRead | None = None
    ) -> tuple[int, Any, bytes, dict[str, RaggedColumn]]:
        """Write ``unit`` (``read`` if given): its index, provenance, digest, ragged fields."""
        rows = self.files.unit_rows(unit)
        layout = self.files.layout
        read = read or read_unit(self.source, rows, with_provenance=True)
        with self.files.rows(unit) as out:
            provenance = write_unit(
                self.source, rows, out, layout=layout, with_provenance=True, read=read
            )
            ragged = ragged_parts(read, layout)
            digest = unit_digest(in_layout_order(layout, out, ragged))
        return unit, crossing_provenance(provenance, unit), digest, ragged


@dataclasses.dataclass(frozen=True, slots=True)
class _DecodeUnit:
    """A worker's task for RAM: decode a unit into fresh arrays the training process copies.

    Attributes:
        source: The source.
        layout: Its columns.
        unit_records: Records per unit.
    """

    source: DataSourceModule
    layout: ColumnLayout
    unit_records: int

    def __call__(
        self, unit: int, read: UnitRead | None = None
    ) -> tuple[int, dict[str, np.ndarray], Any]:
        """Decode ``unit`` (``read`` if given): its index, rows of every column, provenance."""
        rows = unit_rows(unit, self.layout.length, self.unit_records)
        out = {
            c.name: np.empty((len(rows), *c.shape), np.dtype(c.dtype)) for c in self.layout.fixed
        }
        provenance = write_unit(
            self.source, rows, out, layout=self.layout, with_provenance=True, read=read
        )
        return unit, out, crossing_provenance(provenance, unit)


@dataclasses.dataclass(frozen=True, slots=True)
class _FirstUnitOfBuild:
    """What the footprint probe worker runs once: decode one unit, writing nothing.

    Attributes:
        task: The build's worker task.
    """

    task: _WriteUnit | _DecodeUnit

    def __call__(self) -> None:
        """Decode the build's unit 1 (unit 0 is the training process's) into scratch arrays."""
        task = self.task
        layout = task.files.layout if isinstance(task, _WriteUnit) else task.layout
        units = task.files.unit_records if isinstance(task, _WriteUnit) else task.unit_records
        _DecodeUnit(task.source, layout, units)(min(1, unit_count(layout.length, units) - 1))


# ---------------------------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------------------------


def _workers(  # noqa: PLR0913 - what a build's plan is measured from, each given once
    source: DataSourceModule,
    resources: HostResources | None,
    *,
    remaining: int,
    unit_bytes: int,
    sink_bytes: int,
    task: _WriteUnit | _DecodeUnit,
) -> int:
    """The worker processes decoding the build's remaining units; 0 for the calling thread.

    Args:
        source: The source.
        resources: The caller's budget and cap, or ``None``.
        remaining: Units left after the training process's unit 0.
        unit_bytes: ``E``, what one unit sends back to the training process.
        sink_bytes: ``C``, the decoded columns the training process holds (0 for a copy).
        task: What a worker runs.

    Returns:
        The worker count.

    Raises:
        ValueError: If the budget is below what the build holds, naming the terms and the ways
            out.
    """
    gil_bound = source.host_read is HostRead.GIL_BOUND
    if resources is None:
        if gil_bound:
            warn_one_thread(source, "materialize")
        return 0
    in_processes = gil_bound and remaining >= _FEWEST_WORKERS
    path = HostReadPath.PROCESSES if in_processes else HostReadPath.ONE_THREAD
    settings = JaxSettings.current()
    terms: HostTerms = measure_terms(
        unit_bytes=unit_bytes,
        order_bytes=0,  # a build reads its source in stored order, whole units at a time
        depth=0,
        path=path,
        resources=resources,
        read=lambda: task,
        first_unit=_FirstUnitOfBuild,
        settings=settings,
        spans_processes=False,  # a build places on no mesh
    )
    sunk = dataclasses.replace(terms, sink_bytes=sink_bytes)
    admitted = 0
    try:
        if in_processes:
            admitted = budget_plan(resources, path, sunk).workers
        else:  # decoded on the calling thread, which reads nothing ahead
            refuse_below_fixed(resources, sunk)
    except ValueError as error:
        ways = (
            "read the source per batch instead (a Pipeline over it), or write a decoded copy "
            "(materialize(..., directory=...)), which holds no columns in RAM"
            if sink_bytes
            else "or omit host_resources to decode on the calling thread"
        )
        raise ValueError(f"{error}; for this build: {ways}") from error
    workers = min(admitted, remaining)
    if in_processes and workers < _FEWEST_WORKERS:
        logger.info(
            "the budget admits %d worker process(es) for %d unit(s) of %s; fewer than %d decode "
            "slower than the calling thread, which decodes them",
            admitted,
            remaining,
            type(source).__name__,
            _FEWEST_WORKERS,
        )
        return 0
    return workers


def _from_workers(
    task: _WriteUnit | _DecodeUnit, units: int, workers: int
) -> Iterator[tuple[Any, ...]]:
    """Units ``1 .. units - 1`` decoded by ``workers`` Grain worker processes, in unit order.

    Each arrives checked against the unit due (Grain's queue drops an element it cannot pickle
    without raising, google/grain#1415), and the run is refused if it ends early; the workers end
    when the iterator is closed, whatever ends the loop.

    Args:
        task: What each worker runs on a unit.
        units: The build's units, unit 0 included (the training process's).
        workers: Worker processes.

    Yields:
        Each worker's answer for a unit, in unit order.
    """
    dataset = (
        grain.MapDataset.range(1, units)
        .map(task)
        .to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
    )
    run = in_workers(
        dataset, workers=workers, worker_buffer=WORKER_BUFFER, settings=JaxSettings.current()
    )
    received = 0
    with contextlib.closing(iter(run)) as elements:
        for element in elements:
            check_arrival(received + 1, element[0])
            received += 1
            yield element
    check_all_arrived(received + 1, units)


# ---------------------------------------------------------------------------------------------
# The two sinks
# ---------------------------------------------------------------------------------------------


def _into_memory(
    source: DataSourceModule, resources: HostResources | None, unit_records: int
) -> MemorySource:
    """Decode every record into host columns; unit 0 on the calling thread, then the rest."""
    layout, unit_zero = first_read(source, unit_records=unit_records)
    columns = layout.empty()
    units = unit_count(layout.length, unit_records)
    provenance: list[Mapping[str, Any]] = []

    def take(unit: int, rows: Mapping[str, np.ndarray], records: Any) -> None:
        span = unit_rows(unit, layout.length, unit_records)
        for name, column in columns.items():
            column[span.start : span.stop] = rows[name]
        provenance.extend(records or ())

    task = _DecodeUnit(source, layout, unit_records)
    first = task(0, unit_zero)
    take(0, first[1], first[2])
    workers = _workers(
        source,
        resources,
        remaining=units - 1,
        unit_bytes=sum(rows.nbytes for rows in first[1].values()),
        sink_bytes=layout.nbytes,
        task=task,
    )
    if workers:
        for unit, rows, records in _from_workers(task, units, workers):
            take(unit, rows, records)
    else:
        for unit in range(1, units):
            _, rows, records = task(unit)
            take(unit, rows, records)
    return MemorySource.of_columns(
        layout.nested(columns), tuple(provenance) if any(provenance) else ()
    )


def _reusable(directory: Path, identity: Mapping[str, Any]) -> StreamingDiskSource | None:
    """The complete copy of the source in ``directory``, or ``None``, logging why not."""
    try:
        manifest = read_manifest(directory)
    except FileNotFoundError:
        logger.info("%s holds no complete decoded copy: writing it", directory)
        return None
    except ValueError as error:
        logger.info("%s: %s", directory, error)
        return None
    differing = identity_differences(manifest, identity)
    if differing:
        logger.info(
            "%s holds a decoded copy of another source (it differs in %s): rebuilding it",
            directory,
            differing,
        )
        return None
    try:
        return open_decoded_copy(directory, identity=identity)
    except (FileNotFoundError, ValueError) as error:
        logger.info("%s: %s: rebuilding it", directory, error)
        return None


def _onto_disk(
    source: DataSourceModule,
    directory: Path,
    resources: HostResources | None,
    unit_records: int,
    *,
    rebuild: bool,
) -> StreamingDiskSource:
    """Return the source's copy in ``directory``, written there unless a complete one is."""
    identity = cast(SourceIdentified, source).source_identity()
    if not rebuild:
        found = _reusable(directory, identity)
        if found is not None:
            return found
    refuse_unsupported_filesystem(directory)
    with CopyLock(directory):
        if not rebuild:
            found = _reusable(directory, identity)  # another process may have written it
            if found is not None:
                return found
        clear_copy(directory)
        _write_copy(source, directory, resources, unit_records, identity)
    return open_decoded_copy(directory, identity=identity)


def _write_copy(
    source: DataSourceModule,
    directory: Path,
    resources: HostResources | None,
    unit_records: int,
    identity: Mapping[str, Any],
) -> None:
    """Write the copy: reserve its files, unit 0 here, the rest here or in workers, finish."""
    layout, unit_zero = first_read(source, unit_records=unit_records)
    with begin_copy(directory, layout, unit_records=unit_records) as files:
        task = _WriteUnit(source, files)
        digests: list[bytes] = []
        with files.provenance() as provenance, files.ragged() as appended:
            first = task(0, unit_zero)
            digests.append(first[2])
            provenance.write(first[1] or ())
            appended.write(first[3])
            workers = _workers(
                source,
                resources,
                remaining=files.units - 1,
                unit_bytes=host_workers.pickled_bytes(first),
                sink_bytes=0,
                task=task,
            )
            answers: Callable[[], Iterator[tuple[Any, ...]]] = (
                (lambda: _from_workers(task, files.units, workers))
                if workers
                else (lambda: (task(unit) for unit in range(1, files.units)))
            )
            for _, records, digest, ragged in answers():
                digests.append(digest)
                provenance.write(records or ())
                appended.write(ragged)
        finish_copy(files, unit_digests=digests, identity=identity)


__all__ = ["materialize"]
