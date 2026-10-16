"""A run cut into units, and the read of an indexed source's unit: what each reader runs.

A run of a pipeline's epoch plan (:class:`~datarax.pipeline.epochs.Run`) is served in units
(:class:`RunUnits`): one batch, or a chunk of ``K`` batches read together, numbered from the run's
start. A unit as read on the host is a :class:`HostElement`: its batch, its records' provenance,
where the run stands once it is served, and its place in the run. An ``INDEXED`` source's unit is
read by :class:`IndexedRead`, which names the unit's records on the CPU device, several batches
per call (:class:`_NamedBatches`), and reads them with one host read under the run's JAX settings.
The read pickles with what it holds, so a Grain read thread and a worker process run the same
object; it never names the host stage, its cursor or its placement, which consume its units.
"""

from __future__ import annotations

import collections
import dataclasses
import threading
from typing import Any, cast, NamedTuple

import numpy as np

from datarax.core import batch_ops
from datarax.core.data_source import DataSourceModule, Provenance, read_records
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words, to_words
from datarax.pipeline import host_workers
from datarax.pipeline.epochs import HostNaming, Run
from datarax.pipeline.host_workers import JaxSettings


_NAMING_CALL_RECORDS = 16_384
"""Records the host naming names per call (:class:`_NamedBatches`): ``max(1, this // B)`` full
batches, so a call's names are at most this many records' (192 KiB), or one batch's for a larger
batch. Fixed, so the batches' program compiles once per source structure and plan. At B = 256 a
call names 64 batches."""
_CALLS_KEPT = 2
"""Calls' names a run holds at once: the read threads read consecutive units, so at most two
calls' names are in use; an evicted call's names asked for again are named again, identically."""


class HostElement(NamedTuple):
    """A unit as read on the host: its batch, its records' provenance, where the run stands after.

    Attributes:
        batch: The unit's batch, ``(B, ...)`` or a chunk ``(K, B, ...)``, on the host.
        provenance: One mapping per record, in row order, or ``None`` when not asked for.
        after: ``(epoch, position, arrived)`` once the unit is served.
        unit: The unit's place in the run, which the consumer checks units arrive by.
    """

    batch: Batch
    provenance: Provenance | None
    after: tuple[int, int, int]
    unit: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RunUnits:
    """A run cut into units: chunks of ``chunk`` full batches, then the rest one at a time.

    While ``chunk`` full batches remain, a unit holds that many; the full batches left over come
    singly, and the run's short final batch last, so a scan over chunks needs a ``chunk``-step
    body, a one-step body, and one more compile for the short batch. Units are numbered from the
    run's start. A :class:`~datarax.core.data_source.BatchSchedule`.

    Attributes:
        run: The run.
        chunk: Full batches per chunk, or ``None`` for single batches.
    """

    run: Run
    chunk: int | None = None

    @property
    def size(self) -> int:
        """Batches in a chunk; 1 when the run is served in single batches."""
        return 1 if self.chunk is None else self.chunk

    def is_chunk(self, ordinal: int) -> bool:
        """Whether unit ``ordinal`` is a chunk ``(K, B, ...)`` rather than a single batch."""
        if self.chunk is None:
            return False
        full = self._full_batches()
        return full is None or ordinal < full // self.chunk

    def _full_batches(self) -> int | None:
        """The run's full batches, or ``None`` for a run without an end."""
        total = self.run.batches()
        if total is None or total == 0:
            return total
        last = self.run.batch(total - 1)
        assert last is not None  # noqa: S101 - a batch below the run's count exists
        return total - (last[2] < self.run.plan.batch_size)

    def unit(self, ordinal: int) -> tuple[tuple[int, int, int], ...] | None:
        """The batches of unit ``ordinal``, or ``None`` past the run's end.

        Args:
            ordinal: The unit's place in the run.

        Returns:
            Its batches as ``(start, epoch, size)``.
        """
        first = self.first_batch(ordinal)
        count = self.size if self.is_chunk(ordinal) else 1
        batches = tuple(self.run.batch(first + k) for k in range(count))
        return None if None in batches else cast(tuple[tuple[int, int, int], ...], batches)

    def first_batch(self, ordinal: int) -> int:
        """The run's ordinal of unit ``ordinal``'s first batch.

        Args:
            ordinal: The unit's place in the run.

        Returns:
            Its first batch's place in the run.
        """
        if not self.is_chunk(ordinal):
            full = self._full_batches()
            if full is not None and self.chunk is not None:
                chunks = full // self.chunk
                return chunks * self.chunk + ordinal - chunks
            return ordinal
        return ordinal * self.size

    def units(self) -> int | None:
        """The run's units, or ``None`` for a run without an end."""
        full = self._full_batches()
        if full is None:
            return None
        total = self.run.batches()
        assert total is not None  # noqa: S101 - a run with full batches counted has an end
        size = self.size
        return full // size + total - full // size * size

    def after(self, ordinal: int) -> tuple[int, int]:
        """``(epoch, position)`` once unit ``ordinal`` is served.

        Args:
            ordinal: A unit of the run.

        Returns:
            Where the next unit starts.
        """
        batches = self.unit(ordinal)
        assert batches is not None  # noqa: S101 - only a served unit is asked for
        start, epoch, size = batches[-1]
        position, epoch = self.run.plan.advance(start, epoch, size)
        return epoch, position


class IndexedRead:
    """The read of an ``INDEXED`` source's run, unit by unit: what each Grain worker runs.

    A unit's batches are named on the CPU device (several per call, :class:`_NamedBatches`) and
    read with one host read (:func:`~datarax.core.data_source.read_records`, with provenance when
    asked); a chunk is split into its batches and stacked, ``(K, B, ...)``. A read marks a run of
    consecutive records as contiguous only when its names are (so a source reads it as views).
    Each unit is named and read under the run's JAX settings, which a read thread does not
    inherit. It holds the source, the run's units, the naming, the key as host words and the
    settings, and pickles with them, so a worker process reads with a copy of it; a worker's read
    sends provenance as plain dicts, which cross the process boundary
    (:func:`~datarax.pipeline.host_workers.crossing_provenance`).
    """

    def __init__(  # noqa: PLR0913 - what one unit's read needs, each given once
        self,
        source: DataSourceModule,
        units: RunUnits,
        naming: HostNaming,
        *,
        key: np.ndarray | None,
        with_provenance: bool,
        settings: JaxSettings,
        crossing: bool = False,
    ) -> None:
        """Hold what a unit's read needs.

        Args:
            source: The indexed source.
            units: The run's units.
            naming: The pipeline's host naming.
            key: The pipeline's key as host words, or ``None`` when it does not shuffle.
            with_provenance: Whether each unit's provenance is looked up.
            settings: The run's JAX settings, each unit named and read under them.
            crossing: Whether a worker process runs the read, sending its units across.
        """
        self.source = source
        self.units = units
        self.names = _NamedBatches(naming, units.run, key)
        self.key = key
        self.with_provenance = with_provenance
        self.settings = settings
        self.crossing = crossing

    def __call__(self, ordinal: int) -> HostElement:
        """Read unit ``ordinal``.

        Args:
            ordinal: The unit, from the run's start.

        Returns:
            The unit as read on the host.
        """
        batches = self.units.unit(ordinal)
        assert batches is not None  # noqa: S101 - Grain asks only for the run's units
        batch, provenance = self.settings.run(
            self._named_records, self.units.first_batch(ordinal), batches
        )
        if self.units.is_chunk(ordinal):
            batch = batch_ops.as_chunk(batch, len(batches))
        if self.crossing:
            provenance = host_workers.crossing_provenance(provenance, ordinal)
        epoch, position = self.units.after(ordinal)
        return HostElement(batch, provenance, (epoch, position, 0), ordinal)

    def _named_records(
        self, first: int, batches: tuple[tuple[int, int, int], ...]
    ) -> tuple[Batch, Provenance | None]:
        """Name a unit's batches and read their records, on a thread holding the run's settings.

        Args:
            first: The unit's first batch, from the run's start.
            batches: The unit's batches, as the run's schedule gives them.

        Returns:
            The unit's records as one batch, and their provenance when asked for.
        """
        named = [self.names(first + k, *batch) for k, batch in enumerate(batches)]
        indices = np.concatenate([indices for indices, _ in named])
        epochs = np.concatenate([epochs for _, epochs in named])
        values = from_words(indices)
        contiguous = len(values) > 1 and bool(np.all(np.diff(values.astype(np.int64)) == 1))
        return read_records(
            self.source,
            indices,
            epochs=epochs,
            contiguous=contiguous,
            with_provenance=self.with_provenance,
        )


class _NamedBatches:
    """A run's batch names, several full batches named per call of the host naming.

    Naming costs a fixed dispatch to the CPU device and back per call, which dominates a small
    batch's read, so the run's full batches are named :data:`_NAMING_CALL_RECORDS` records at a
    time: call ``c`` names the run's batches ``c * M .. c * M + M - 1`` (``M`` full batches),
    through :meth:`~datarax.pipeline.epochs.HostNaming.batches`, which names each exactly as alone
    whatever the global PRNG impl (naming pins threefry, exact under ``jax.vmap`` over keys).
    Calls count from the run's start, so a run resumed anywhere names the same records. The run's
    short final batch is named alone, by the program that serves its size; a call's slots past
    the run's full batches repeat a batch of the call and are never served. The read threads ask
    in any order: a lock guards the calls' names, of which the last :data:`_CALLS_KEPT` are held
    (12 bytes a record: its index words and epoch). It pickles as the naming, run and key.
    """

    def __init__(self, naming: HostNaming, run: Run, key: np.ndarray | None) -> None:
        """Hold what the run's names need; nothing is named until a batch asks.

        Args:
            naming: The pipeline's host naming.
            run: The run.
            key: The pipeline's key as host words, or ``None`` when it does not shuffle.
        """
        self._naming = naming
        self._run = run
        self._key = key
        self._per_call = max(1, _NAMING_CALL_RECORDS // run.plan.batch_size)
        self._lock = threading.Lock()
        self._calls: collections.OrderedDict[int, tuple[np.ndarray, np.ndarray]] = (
            collections.OrderedDict()
        )

    def __getstate__(self) -> dict[str, Any]:
        """What a copy needs (a worker process's): the naming, run and key, no names."""
        return {"naming": self._naming, "run": self._run, "key": self._key}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Hold what was copied; the copy names its own batches."""
        self.__init__(state["naming"], state["run"], state["key"])

    def __call__(
        self, ordinal: int, start: int, epoch: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """The records of the run's batch ``ordinal``, which starts at ``start`` of ``epoch``.

        Args:
            ordinal: The batch's place in the run.
            start: Where it starts in its epoch.
            epoch: Its first row's epoch.
            size: Its rows.

        Returns:
            Each row's record index, uint32 ``(size, 2)``, and epoch, int32 ``(size,)``.
        """
        if size != self._run.plan.batch_size:
            return self._naming(start, epoch, size, self._key)
        call, slot = divmod(ordinal, self._per_call)
        with self._lock:
            names = self._calls.get(call)
            if names is None:
                names = self._name(call, (start, epoch))
                self._calls[call] = names
                while len(self._calls) > _CALLS_KEPT:
                    self._calls.popitem(last=False)
            else:
                self._calls.move_to_end(call)
        indices, epochs = names
        return indices[slot], epochs[slot]

    def _name(self, call: int, filler: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Name call ``call``'s batches; ``filler``, a full batch, fills slots past the run's."""
        starts = np.empty((self._per_call, 2), np.uint32)
        epochs = np.empty(self._per_call, np.int32)
        size = self._run.plan.batch_size
        for slot in range(self._per_call):
            batch = self._run.batch(call * self._per_call + slot)
            start, epoch = filler if batch is None or batch[2] != size else batch[:2]
            starts[slot], epochs[slot] = to_words(start), epoch
        return self._naming.batches(starts, epochs, self._key)


__all__ = ["HostElement", "IndexedRead", "RunUnits"]
