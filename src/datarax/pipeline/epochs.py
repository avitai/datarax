"""How a pipeline's records divide into batches and epochs, and which records a batch holds.

:class:`EpochPlan` is the rule for a source the pipeline indexes, :func:`stream_batches` the same
rule over a stream's passes; this module is the one place the rule lives. :func:`batch_records`
names the records a batch of an indexed source holds by that rule, each row by its own epoch's
order; the compiled session traces it, and :class:`HostNaming` runs it on the CPU device for the
host stage. No batch holds padding.
Under
``drop_last`` the records short of a full batch are skipped and the next batch starts the
next epoch, tf.data's and Grain's ``batch(drop_remainder=True).repeat()``. Otherwise a batch
reaching the epoch's end is completed from the head of the next epoch's order, their
``repeat().batch()``, so every epoch serves each record once. A session serving a number of
epochs stops after exactly those records, so its final batch may be short.
:class:`~datarax.pipeline.pipeline.Pipeline` builds a plan from its source's current length,
and its length, ``batches_left``, iteration sessions and compiled step all read it. On the host
positions and epochs are Python integers, exact at every length up to ``2**64 - 1`` records.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from datarax.core import batch_ops
from datarax.core.data_source import DataSourceModule, Provenance
from datarax.core.element_batch import Batch
from datarax.core.index_words import is_word_start, split_constant, subtract, to_words
from datarax.core.prng import host_device
from datarax.pipeline.dag import Records


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class EpochPlan:
    """How records divide into batches and epochs.

    Attributes:
        length: Records per epoch, or ``None`` for a source without a length, whose
            epochs never end.
        batch_size: Records per batch.
        drop_last: Whether an epoch's records short of a full batch are skipped, rather
            than completed from the next epoch.
        num_epochs: Epochs a session serves, or ``None`` for a session that never stops.
    """

    length: int | None
    batch_size: int
    drop_last: bool
    num_epochs: int | None

    def __post_init__(self) -> None:
        """Refuse a plan no pipeline can serve.

        Raises:
            ValueError: If ``batch_size`` is not positive, ``num_epochs`` is neither
                ``None`` nor at least 1, the source has no records, or ``drop_last``
                skips every record because a batch is larger than an epoch.
        """
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {self.batch_size}")
        if self.num_epochs is not None and self.num_epochs < 1:
            raise ValueError(
                f"num_epochs must be at least 1, or None for a stream; got {self.num_epochs}"
            )
        if self.length == 0:
            raise ValueError("the source has no records, so no batch can be served")
        if self.drop_last and self.length is not None and self.batch_size > self.length:
            raise ValueError(
                f"drop_last needs batch_size <= len(source), or every record is skipped; got "
                f"batch_size {self.batch_size} over {self.length} records"
            )

    @property
    def crosses(self) -> bool:
        """Whether a batch reaching an epoch's end continues in the next epoch."""
        return not self.drop_last and self.length is not None

    def exhausted[T: (int, jax.Array)](self, position: T) -> bool | jax.Array:
        """Whether the epoch cannot start another batch at ``position``.

        It can while at least one record is left, or under ``drop_last`` a full batch.

        Args:
            position: Records of the epoch already served; a Python int or a traced array.

        Returns:
            A bool for an int position, a bool array for an array; never exhausted for a
            source without a length.
        """
        if self.length is None:
            return False
        needed = self.batch_size if self.drop_last else 1
        return position + needed > self.length

    def batch_start[T: (int, jax.Array)](self, position: T, epoch: T) -> tuple[T, T]:
        """Where the next batch starts: here, or at the next epoch's start if this one is done.

        Args:
            position: Records of the epoch already served.
            epoch: The epoch counter.

        Returns:
            ``(position, epoch)`` the batch starts at.
        """
        done = self.exhausted(position)
        first = position * 0  # position 0, as an int or an array of the position's dtype
        return _select(done, first, position), _select(done, epoch + 1, epoch)

    def advance[T: (int, jax.Array)](self, start: T, epoch: T, size: int) -> tuple[T, T]:
        """The position and epoch after a batch of ``size`` records starting at ``start``.

        A batch crossing the epoch's end leaves the position in the epoch it ends in (a
        batch larger than an epoch spans several); one ending exactly at an epoch's end
        leaves that epoch exhausted, so the next batch starts the next epoch.

        Args:
            start: Where the batch started, as :meth:`batch_start` gives it.
            epoch: The epoch the batch started in.
            size: Records in the batch.

        Returns:
            ``(position, epoch)`` after the batch.
        """
        end = start + size
        if self.length is None:
            return end, epoch
        crossed = (end - 1) // self.length  # epoch ends the batch passed, not reached
        return end - crossed * self.length, epoch + crossed

    def epochs_touched(self, size: int) -> int:
        """The most epochs a batch of ``size`` records starting inside an epoch holds.

        Args:
            size: Records in the batch.

        Returns:
            The count; 1 for a source without a length.
        """
        if self.length is None:
            return 1
        return (self.length + size - 2) // self.length + 1

    @staticmethod
    def next_epoch[T: (int, jax.Array)](epoch: T) -> tuple[int, T]:
        """Where the epoch after ``epoch`` starts: position 0.

        Args:
            epoch: The epoch just finished.

        Returns:
            ``(0, epoch + 1)``.
        """
        return 0, epoch + 1

    def run_extent(self, position: int) -> tuple[int, int] | None:
        """The batches a session starting at ``position`` serves, and its final batch's size.

        The session finishes the current epoch, an exhausted one counting as served, then
        serves the rest of ``num_epochs``.

        Args:
            position: Records of the current epoch already served.

        Returns:
            ``(batches, final_batch_size)``, or ``None`` for a session that never stops: a
            stream or a source without a length.
        """
        if self.num_epochs is None or self.length is None:
            return None
        left = 0 if self.exhausted(position) else self.length - position
        later = self.num_epochs - 1
        if self.drop_last:
            return left // self.batch_size + later * (self.length // self.batch_size), (
                self.batch_size
            )
        records = left + later * self.length
        batches = -(-records // self.batch_size)  # integer ceiling: exact past 2**53 records
        if batches == 0:
            return 0, 0
        return batches, records - (batches - 1) * self.batch_size


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Run:
    """The batches a run serves, each found from its ordinal without walking to it.

    A run starts at ``(position, epoch)`` and serves the plan's batches until epoch ``end_epoch``
    would start; a run without an end (``end_epoch`` ``None``) never stops. The end is recorded,
    not recomputed, so a run resumed anywhere ends where the uninterrupted one would. Batch
    ``ordinal`` is the one :meth:`EpochPlan.batch_start` and :meth:`EpochPlan.advance` reach
    after that many batches, the last one short when the records left do not fill it. Python
    integers throughout, exact at every length up to ``2**64 - 1`` records.

    Attributes:
        plan: How records divide into batches and epochs.
        position: Records of ``epoch`` served before the run's first batch.
        epoch: The epoch the run starts in.
        end_epoch: The epoch the run stops before, or ``None`` for a run without an end.
    """

    plan: EpochPlan
    position: int
    epoch: int
    end_epoch: int | None

    @classmethod
    def starting(cls, plan: EpochPlan, position: int, epoch: int) -> Run:
        """A run of the plan's ``num_epochs`` epochs from ``(position, epoch)``.

        It finishes the epoch it starts in, an exhausted one counting as served, then serves the
        rest: it stops before epoch ``epoch + num_epochs``, as
        :meth:`EpochPlan.run_extent` counts a session from ``position``.

        Args:
            plan: How records divide into batches and epochs.
            position: Records of ``epoch`` already served.
            epoch: The current epoch.

        Returns:
            The run.
        """
        start, first = plan.batch_start(position, epoch)
        end = None if plan.num_epochs is None or plan.length is None else epoch + plan.num_epochs
        return cls(plan=plan, position=start, epoch=first, end_epoch=end)

    def __len__(self) -> int:
        """The batches the run serves; see :meth:`batches`.

        Returns:
            The count.

        Raises:
            TypeError: If the run has no end.
        """
        count = self.batches()
        if count is None:
            raise TypeError("a run without an end has no length")
        return count

    def batches(self) -> int | None:
        """The batches the run serves, or ``None`` for a run without an end."""
        plan, length = self.plan, self.plan.length
        if self.end_epoch is None or length is None:
            return None
        position, epoch = plan.batch_start(self.position, self.epoch)
        if epoch >= self.end_epoch:
            return 0
        if plan.drop_last:
            per_epoch = length // plan.batch_size
            return (length - position) // plan.batch_size + (self.end_epoch - epoch - 1) * per_epoch
        records = (self.end_epoch - epoch) * length - position
        return -(-records // plan.batch_size)

    def batch(self, ordinal: int) -> tuple[int, int, int] | None:
        """Batch ``ordinal`` of the run: ``(start, epoch, size)``, or ``None`` past its end.

        Args:
            ordinal: The batch's place in the run, from 0.

        Returns:
            Where the batch starts, its first row's epoch and its size.
        """
        plan, length, size = self.plan, self.plan.length, self.plan.batch_size
        position, epoch = plan.batch_start(self.position, self.epoch)
        if length is None:
            return position + ordinal * size, epoch, size
        if plan.drop_last:
            first = (length - position) // size
            if ordinal < first:
                start, at = position + ordinal * size, epoch
            else:
                later, index = divmod(ordinal - first, length // size)
                start, at = index * size, epoch + 1 + later
            if self.end_epoch is not None and at >= self.end_epoch:
                return None
            return start, at, size
        served = position + ordinal * size  # records from the start of ``epoch``
        if self.end_epoch is not None:
            total = (self.end_epoch - epoch) * length
            if served >= total:
                return None
            size = min(size, total - served)
        return served % length, epoch + served // length, size


def _select[T: (int, jax.Array)](condition: bool | jax.Array, if_true: T, if_false: T) -> T:
    """``if_true`` where ``condition`` holds, else ``if_false``: on the host or traced alike."""
    if isinstance(condition, bool):
        return if_true if condition else if_false
    return cast(T, jnp.where(condition, if_true, if_false))


def batch_records(
    source: DataSourceModule,
    plan: EpochPlan,
    *,
    key_base: ArrayLike | None,
    start: ArrayLike,
    epoch: ArrayLike,
    size: int,
) -> Records:
    """The records of the batch of ``size`` rows starting at ``start`` of epoch ``epoch``.

    The rows are the plan's: the rest of the epoch ``start`` is in, then, when the plan crosses
    epochs, the heads of the following epochs' orders, so no row is padding. Every epoch the batch
    can touch is named by one ``source.record_indices_at`` vmapped over the epochs' keys (the
    first from ``start``, the rest from their heads), and each row takes its own epoch's name: no
    conditional, one batched index computation, the same program for every batch, and index
    arrays of O(``size``) per epoch touched. Epoch ``e`` is ordered by
    ``fold_in(wrap_key_data(key_base), e)``, or sequentially without a key. Traceable; the source
    is read for its order only, never its records.

    Args:
        source: The indexed source.
        plan: The pipeline's epoch plan.
        key_base: The pipeline's key data, or ``None`` when it does not shuffle.
        start: Where the batch starts in its epoch: a traced int32 (the compiled session's) or
            two uint32 words ``(hi, lo)`` (the host stage's, exact past ``2**31``).
        epoch: The epoch the first row belongs to, int32.
        size: Rows to serve (static).

    Returns:
        Each row's record index, uint32 ``(size, 2)``, and epoch, int32 ``(size,)``.
    """
    epoch = jnp.asarray(epoch, dtype=jnp.int32)

    def names(first: jax.Array, epoch_of: jax.Array) -> jax.Array:
        key = (
            None
            if key_base is None
            else jax.random.fold_in(jax.random.wrap_key_data(jnp.asarray(key_base)), epoch_of)
        )
        return jnp.asarray(source.record_indices_at(first, size, key), jnp.uint32)

    if not plan.crosses:
        return Records(names(jnp.asarray(start), epoch), jnp.full((size,), epoch, jnp.int32))
    length = plan.length
    assert length is not None  # noqa: S101 - a crossing plan has a length
    first = jnp.asarray(start)
    offsets = jnp.arange(plan.epochs_touched(size), dtype=jnp.int32)
    later_heads = jnp.zeros_like(first)
    starts = jnp.where((offsets == 0).reshape(-1, *([1] * first.ndim)), first, later_heads)
    named = jax.vmap(names)(starts, epoch + offsets)
    row = jnp.arange(size, dtype=jnp.int32)
    left = _rows_left(first, length, size)
    past = row - left  # rows after the epoch's end, counted from the next epoch's head
    if length > size:
        later = (row >= left).astype(jnp.int32)
        column = jnp.where(row < left, row, past)
    else:
        later = jnp.where(row < left, 0, 1 + past // length)
        column = jnp.where(row < left, row, past % length)
    return Records(named[later, column], epoch + later)


def _rows_left(start: jax.Array, length: int, size: int) -> jax.Array:
    """Rows of the batch inside the epoch it starts in: ``min(length - start, size)``, int32.

    Computed in two words, so a start past ``2**31`` of a longer epoch is exact.
    """
    words = (
        (start[0], start[1])
        if is_word_start(start)
        else (jnp.zeros((), jnp.uint32), start.astype(jnp.uint32))
    )
    length_high, length_low = split_constant(length)
    high, low = subtract((jnp.asarray(length_high), jnp.asarray(length_low)), words)
    inside = (high == 0) & (low < size)
    return jnp.where(inside, low, np.uint32(size)).astype(jnp.int32)


class HostNaming:
    """A pipeline's naming on the host: :func:`batch_records` as one ``jax.jit`` program on the CPU.

    The host stage names each batch's records here, before reading them: the start goes in as two
    uint32 words, so every position up to ``2**64 - 1`` is exact, and the program runs on the CPU
    device (:func:`~datarax.core.prng.host_device`), off the accelerator's queue. The source is
    closed over, never an argument, so none of its records is transferred; its order reads its
    lengths only. One program is compiled per batch shape (a run's short final batch is a second)
    and reused for every batch, run, reset and restore of the pipeline that holds this object.
    Every input is placed on the CPU device and every output read back explicitly, so naming
    needs no implicit transfer.
    """

    def __init__(self, source: DataSourceModule, plan: EpochPlan, *, shuffled: bool) -> None:
        """Hold the source and the plan; the program is built and compiled at first use.

        Args:
            source: The indexed source.
            plan: The pipeline's epoch plan.
            shuffled: Whether the pipeline shuffles, which a key then orders.
        """
        self._source = source
        self._plan = plan
        self._shuffled = shuffled
        self._device = host_device()
        self._lock = threading.Lock()
        self._program: Callable[..., tuple[jax.Array, jax.Array]] | None = None

    def __getstate__(self) -> dict[str, object]:
        """What a copy needs (a worker process's): the source, plan and order, not the program."""
        return {"source": self._source, "plan": self._plan, "shuffled": self._shuffled}

    def __setstate__(self, state: dict[str, object]) -> None:
        """Hold what was copied; the copy builds its own program at first use."""
        self.__init__(
            cast(DataSourceModule, state["source"]),
            cast(EpochPlan, state["plan"]),
            shuffled=bool(state["shuffled"]),
        )

    def _built(self) -> Callable[..., tuple[jax.Array, jax.Array]]:
        """The program, built once however many threads ask for it at once."""
        program = self._program
        if program is None:
            with self._lock:
                if self._program is None:
                    source, plan = self._source, self._plan

                    def _names(
                        start: jax.Array, epoch: jax.Array, key: jax.Array | None, size: int
                    ) -> tuple[jax.Array, jax.Array]:
                        records = batch_records(
                            source, plan, key_base=key, start=start, epoch=epoch, size=size
                        )
                        return records.indices, records.epochs

                    self._program = jax.jit(_names, static_argnames=("size",))
                program = self._program
        return program

    def __call__(
        self, start: int, epoch: int, size: int, key: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The records of the batch of ``size`` rows starting at ``start`` of epoch ``epoch``.

        Args:
            start: Where the batch starts in its epoch, any position up to ``2**64 - 2``.
            epoch: The epoch the first row belongs to.
            size: Rows to serve.
            key: The pipeline's key as uint32 words on the host
                (:func:`~datarax.core.prng.key_words`), or ``None`` when it does not shuffle.

        Returns:
            Each row's record index, uint32 ``(size, 2)``, and epoch, int32 ``(size,)``, on the
            host.

        Raises:
            ValueError: If a key is given to a pipeline that does not shuffle, or none to one
                that does.
        """
        if (key is None) == self._shuffled:
            raise ValueError(
                "a pipeline that shuffles orders each epoch by its key, which naming then takes"
                if self._shuffled
                else "this pipeline does not shuffle, so naming takes no key"
            )
        device = self._device
        placed_key = None if key is None else jax.device_put(np.asarray(key, np.uint32), device)
        indices, epochs = self._built()(
            jax.device_put(to_words(start), device),
            jax.device_put(np.int32(epoch), device),
            placed_key,
            size=size,
        )
        return jax.device_get(indices), jax.device_get(epochs)


def stream_batches(
    pull: Callable[[int], tuple[Batch, Provenance]],
    batch_size: int,
    *,
    drop_last: bool,
    num_epochs: int | None,
) -> Iterator[tuple[Batch, Provenance]]:
    """The batches of a stream's passes, by the rule :class:`EpochPlan` applies to an index.

    ``pull(n)`` returns up to ``n`` records of the stream's current pass as a ``Batch`` and their
    provenance (one mapping per record, or none), empty at the pass's end, after which it serves
    the next pass. Under ``drop_last`` a pass's records short of a full batch are skipped;
    otherwise the batch is completed from the head of the next pass, each row keeping its own
    epoch. After ``num_epochs`` passes the run ends, its last batch possibly short; with ``None``
    it never ends. A run may start where an earlier one stopped: mid-pass, or at a pass's end it
    has not yet read, which counts as a pass served, as an exhausted epoch does. A pass the run
    read from its start to its end holding no record means the stream holds none, and ends the
    run; whether a pass held records is known only for a pass the run saw begin, never from a
    count over the run.

    Args:
        pull: Reads up to the given number of records of the current pass, with provenance.
        batch_size: Records per batch.
        drop_last: Whether a pass's records short of a full batch are skipped.
        num_epochs: Passes served, or ``None`` for no end.

    Yields:
        Full batches with their records' provenance, and under ``drop_last=False`` the run's
        short final batch.
    """
    pending: list[tuple[Batch, Provenance]] = []
    held = passes = 0
    served: int | None = None  # records of the current pass, unknown until a pass starts in view
    while num_epochs is None or passes < num_epochs:
        batch, provenance = pull(batch_size - held)
        if batch.batch_size == 0:
            passes += 1
            if served == 0:  # a pass read whole held no record: the stream holds none
                break
            served = 0
            if drop_last:
                pending, held = [], 0
            continue
        pending.append((batch, provenance))
        held += batch.batch_size
        served = (served or 0) + batch.batch_size
        if held == batch_size:
            yield _joined(pending)
            pending, held = [], 0
    if pending and not drop_last:
        yield _joined(pending)


def _joined(parts: list[tuple[Batch, Provenance]]) -> tuple[Batch, Provenance]:
    if len(parts) == 1:
        return parts[0]
    batch = batch_ops.concatenate([batch for batch, _ in parts])
    return batch, tuple(record for _, provenance in parts for record in provenance)
