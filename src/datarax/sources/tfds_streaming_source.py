"""TFDSStreamingSource: streams a split TFDS has prepared as TFRecord, read without TensorFlow.

The stream is for datasets too large for host memory, in TFDS's default format. An index of each
shard file, built once from its frame headers, gives every record's byte offset; a pass's order is
computed over record ids, each record's payload is read at its offset and checked against its CRC,
and TFDS's NumPy decoder decodes it last. Records are named by the id TFDS reports for them,
``tfds_id``: the shard file and the record's offset in it (``STREAM_IDS``). Given the pipeline's
key, a pass is ordered as TFDS's training read orders it: the shard files in a keyed order, read 16
at a time in blocks of 16, through tf.data's buffer shuffle, the buffer holding record ids.

TensorFlow is never imported: TensorFlow in a JAX process breaks JAX's NCCL collectives. The source
does not prepare a dataset, because preparing imports TensorFlow: a split that is not prepared, or
is prepared only as ArrayRecord, is refused, naming the call that prepares it or the source that
reads it. Reading needs the ``data`` extra; preparing needs the ``tfds`` extra, in a process of its
own.

This module imports ``tensorflow_datasets`` at its top. ``datarax.sources`` exports the TFDS
sources' names lazily, so ``import datarax.sources`` imports no TFDS: the import happens on the line
that imports a TFDS source or ``from_tfds``. TFDS's first import enters
``etils.epy.lazy_imports()``, which replaces ``builtins.__import__`` for the whole process while it
runs. A garbage collection in that window runs finalizers whose imports fail, so the window must not
open inside a constructor in the middle of a run.
"""

from __future__ import annotations

import contextlib
import itertools
import sys
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

import grain
import numpy as np
import tensorflow_datasets as tfds
from jax.typing import ArrayLike

from datarax.core.data_source import (
    BatchSchedule,
    HostRead,
    Provenance,
    record_words,
    RecordIdentity,
    refuse_unnamed,
)
from datarax.core.host_values import HostValue
from datarax.core.prng import pass_generator
from datarax.sources._config_base import StreamingSourceConfigBase
from datarax.sources._source_base import DatasetSourceMixin, StreamingSourceBase
from datarax.sources._stream_order import buffer_shuffled, interleaved, RecordId, shard_ids
from datarax.sources._tfds_prepared import (
    kept_features,
    prepare_call,
    prepared_builder,
    supervised_keys,
    where,
)
from datarax.sources._tfrecord import (
    payload,
    record_index,
    remove,
    Shard,
    ShardIndex,
    share_index,
    split_shard,
)
from datarax.sources.eager_source import parts_of_records
from datarax.sources.stream_iterator import StreamChunk


@dataclass(frozen=True)
class TFDSStreamingConfig(StreamingSourceConfigBase):
    """Configuration for TFDSStreamingSource (streams a split prepared as TFRecord).

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to stream, slices included, e.g. ``"train[:50%]"`` (required)
        data_dir: Directory holding the dataset prepared as TFRecord; TFDS's default data
            directory (``TFDS_DATA_DIR``, else ``~/tensorflow_datasets``) when ``None``
        shuffle_buffer_size: Records the shuffle buffer holds when the pipeline shuffles
        as_supervised: If True, keeps only the dataset's supervised features
            (``info.supervised_keys``), under their own names
        include_keys: Optional set of keys to include in output (exclusive with exclude_keys)
        exclude_keys: Optional set of keys to exclude from output (exclusive with include_keys)

    Note:
        Whether a pass is shuffled, and by which key, is the pipeline's
        (``Pipeline(shuffle=...)``). The source never prepares a dataset; see
        :class:`TFDSStreamingSource`.
    """

    as_supervised: bool = False


class _Frame(NamedTuple):
    """One serialized record and the id it is named by."""

    shard: int
    offset: int
    raw: bytes


def _not_streamable(name: str, data_dir: str | None, found: str) -> FileNotFoundError:
    """The refusal of a copy the stream cannot read, naming what reads it or prepares one."""
    return FileNotFoundError(
        f"TFDS dataset {name!r} is not prepared as TFRecord in {where(data_dir)}: {found}. "
        "TFDSStreamingSource streams a copy prepared as TFRecord, TFDS's default format, and "
        "never prepares one, since preparing imports TensorFlow. A copy prepared as ArrayRecord "
        "(array_record) has random access: TFDSEagerSource reads it into host memory, and a copy "
        "larger than host memory is read by index once the pipeline's host stage serves indexed "
        "sources. To stream, prepare a TFRecord copy once, in a process of its own with the tfds "
        f"extra installed: {prepare_call(name, data_dir, 'tfrecord')}."
    )


def _features(name: str, data_dir: str | None) -> Any:
    """The dataset's TFDS features, which decode its serialized records with NumPy."""
    return tfds.builder(name, data_dir=data_dir).info.features


@dataclass(frozen=True, slots=True)
class _KeptFeatures:
    """The features a TFDS stream keeps of each record, as plain values that pickle.

    Attributes:
        keys: The supervised features kept, or ``None`` for every feature.
        include_keys: Features kept, when given.
        exclude_keys: Features dropped, when given.
    """

    keys: tuple[str, ...] | None
    include_keys: frozenset[str] | None
    exclude_keys: frozenset[str] | None

    @classmethod
    def of_config(  # noqa: DOC502 - supervised_keys raises
        cls, config: TFDSStreamingConfig, info: Any, dataset: str
    ) -> _KeptFeatures:
        """The features ``config`` keeps of ``dataset``'s records.

        Args:
            config: The stream's configuration.
            info: The dataset's ``DatasetInfo``.
            dataset: The dataset's name.

        Returns:
            The kept features.

        Raises:
            ValueError: If ``as_supervised`` is asked of a dataset without supervised keys.
        """
        include, exclude = config.include_keys, config.exclude_keys
        return cls(
            keys=tuple(supervised_keys(info, dataset)) if config.as_supervised else None,
            include_keys=None if include is None else frozenset(include),
            exclude_keys=None if exclude is None else frozenset(exclude),
        )

    def of(self, record: dict[str, Any]) -> dict[str, Any]:
        """The features of ``record`` kept."""
        return kept_features(
            record,
            self.keys,
            None if self.include_keys is None else set(self.include_keys),
            None if self.exclude_keys is None else set(self.exclude_keys),
        )


@dataclass(frozen=True, slots=True)
class _StreamRead:
    """What a run of a TFDS stream reads, as plain values that pickle to a worker process.

    Attributes:
        name: The TFDS dataset name.
        data_dir: The data directory holding it prepared as TFRecord.
        shards: The shard files and offsets the split reads, in file order.
        index: Each shard's record index, aligned with ``shards``; built once by the source and
            pickled with the read, so no worker reads a frame header.
        length: The records one pass serves.
        key: The pipeline's key as host words, each pass ordered by ``fold_in(key, pass)``, or
            ``None`` for file order.
        buffer_size: Records the shuffle buffer holds.
        schedule: The run's units, each a run of batches ``(start, pass, size)``.
        kept: The features kept of each record.
    """

    name: str
    data_dir: str | None
    shards: tuple[Shard, ...]
    index: tuple[ShardIndex, ...]
    length: int
    key: np.ndarray | None
    buffer_size: int
    schedule: BatchSchedule
    kept: _KeptFeatures


def _ordered_ids(read: _StreamRead, pass_index: int) -> Iterator[RecordId]:
    """Pass ``pass_index``'s records in its order, as ids: file order, or TFDS's training read's.

    The same order every worker of the run computes, from the read alone, with no file read.
    """
    streams = [shard_ids(position, shard) for position, shard in enumerate(read.shards)]
    if read.key is None:
        return itertools.chain.from_iterable(streams)
    generator = pass_generator(read.key, pass_index)
    order = generator.permutation(len(streams))
    return buffer_shuffled(interleaved([streams[i] for i in order]), read.buffer_size, generator)


def _decoded_batch(
    features: Any, frames: Sequence[_Frame], kept: _KeptFeatures
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray]:
    """``frames`` decoded with TFDS's NumPy decoder: host columns, provenance and ids.

    Args:
        features: The dataset's TFDS features.
        frames: The serialized records.
        kept: The features kept of each record.

    Returns:
        The array part as host columns, one provenance mapping per record and the records' ids.
    """
    records = [kept.of(features.deserialize_example_np(frame.raw)) for frame in frames]
    columns, provenance = parts_of_records(records)
    ids = np.asarray([(frame.shard << 32) | frame.offset for frame in frames], dtype=np.uint64)
    return columns, provenance or tuple({} for _ in frames), ids


@dataclass(frozen=True, slots=True)
class _PassUnits:
    """One pass from its start in units of ``size`` records, the last one short: a schedule."""

    pass_index: int
    length: int
    size: int

    def unit(self, ordinal: int) -> tuple[tuple[int, int, int], ...] | None:
        """Batch ``ordinal`` of the pass, or ``None`` past its end."""
        start = ordinal * self.size
        if start >= self.length:
            return None
        return ((start, self.pass_index, min(self.size, self.length - start)),)


class TFDSStreamDataset(grain.IterDataset):
    """A run of a TFDS stream's passes as a Grain dataset of decoded units.

    Its iterator computes each pass's order over record ids (no file read) and walks the run's
    schedule: for each unit it takes the ids at the unit's positions, a batch crossing a pass's
    end continuing at the next pass's head, reads those records' payloads by their offsets in the
    index and decodes them last. An element is the unit's host columns, provenance, ids, each
    record's pass and the unit's place in the run. Units are numbered from the run's first,
    wherever the run starts, and records before the run's start are skipped as ids, never read.
    It implements Grain's slicing hook:
    with ``set_slice(slice(i, None, k))`` the iterator walks the same schedule but reads and
    decodes only units ``j`` with ``j % k == i``, so ``k`` such slices interleaved round robin from
    the first give the unsliced run, the stream's order not depending on ``k``, and the slices
    read each record once between them. That is how Grain's process prefetch splits a dataset
    across workers. The dataset pickles (its read is plain values; the offset index goes as its
    arrays, or, for a run read by worker processes, as its place in the one copy they share), and
    each iterator opens its own files and decoder, so iterators in threads or processes share no
    lazy state. It reports what workers reading it share and hold
    (:class:`~datarax.core.host_resources.WorkerFootprint`).
    """

    def __init__(self, read: _StreamRead) -> None:
        """Hold the run's read; nothing is opened until iteration.

        The source builds the dataset (:meth:`TFDSStreamingSource.pass_dataset`,
        :meth:`TFDSStreamingSource.run_dataset`); its read is the source's own.

        Args:
            read: What the run reads.
        """
        super().__init__()
        self._read = read
        self._slice = slice(0, None, 1)

    def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
        """Keep units ``j`` with ``j % sl.step == sl.start``: Grain's slicing hook.

        Args:
            sl: The slice of units this dataset serves.
            sequential_slice: Must be ``False``: a stream of unknown length cannot be cut into
                contiguous blocks.

        Raises:
            ValueError: If ``sequential_slice`` is asked, or the slice has a stop.
        """
        if sequential_slice or sl.stop is not None:
            raise ValueError(
                "a stream is sliced by strides, slice(i, None, k), so its order does not depend "
                f"on the number of slices; got {sl} with sequential_slice={sequential_slice}"
            )
        self._slice = sl

    def __iter__(self) -> _TFDSStreamIterator:
        """A fresh iterator over the run, opening its own files and decoder."""
        return _TFDSStreamIterator(self._read, self._slice)

    def shared_bytes(self) -> int:
        """The offset index's bytes when every worker reads one shared copy of it, else 0."""
        index = self._read.index
        return sum(shard.nbytes for shard in index) if index and index[0].shared else 0

    def working_bytes(self) -> int:
        """A worker's shuffle buffer of record ids, when the run is shuffled."""
        if self._read.key is None:
            return 0
        held = RecordId(1 << 20, 1 << 20)
        return self._read.buffer_size * (sys.getsizeof(held) + 2 * sys.getsizeof(1 << 20) + 8)


type _Unit = tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray, np.ndarray, int]
"""A decoded unit: its host columns, provenance, ids, each record's pass, and its place in the
run."""


class _TFDSStreamIterator(grain.DatasetIterator):
    """Units of a run of a TFDS stream, those of its slice read and decoded."""

    def __init__(self, read: _StreamRead, sl: slice) -> None:
        super().__init__()
        self._read = read
        self._features = _features(read.name, read.data_dir)
        self._start, self._step = sl.start or 0, sl.step or 1
        self._unit = 0
        self._pass: int | None = None
        self._ids: Iterator[RecordId] = iter(())
        self._at = 0  # ids of the current pass taken
        self._files = contextlib.ExitStack()
        self._open: dict[int, Any] = {}
        weakref.finalize(self, self._files.close)

    def __next__(self) -> _Unit:
        while True:
            batches = self._read.schedule.unit(self._unit)
            if batches is None:
                self._files.close()
                raise StopIteration
            unit, self._unit = self._unit, self._unit + 1
            records = [record for batch in batches for record in self._take(*batch)]
            if unit % self._step == self._start:
                frames = [self._frame(record) for record, _ in records]
                columns, provenance, ids = _decoded_batch(self._features, frames, self._read.kept)
                epochs = np.asarray([pass_index for _, pass_index in records], dtype=np.int32)
                return columns, provenance, ids, epochs, unit

    def _take(self, start: int, pass_index: int, size: int) -> list[tuple[RecordId, int]]:
        """The ids of ``size`` positions from ``start`` of pass ``pass_index`` on, across passes."""
        taken: list[tuple[RecordId, int]] = []
        while len(taken) < size:
            self._seek(pass_index, start)
            count = min(size - len(taken), self._read.length - start)
            taken.extend((record, pass_index) for record in itertools.islice(self._ids, count))
            self._at += count
            pass_index, start = pass_index + 1, 0
        return taken

    def _seek(self, pass_index: int, position: int) -> None:
        """Move to ``position`` of pass ``pass_index``'s order, skipping ids, reading nothing."""
        if pass_index != self._pass or position < self._at:
            self._pass, self._ids, self._at = pass_index, _ordered_ids(self._read, pass_index), 0
        for _ in itertools.islice(self._ids, position - self._at):
            pass
        self._at = position

    def _frame(self, record: RecordId) -> _Frame:
        """The record's payload, read at its offset, and its id."""
        shard = self._read.shards[record.position]
        file = self._open.get(record.position)
        if file is None:
            opened = Path(shard.path).open("rb", buffering=0)  # noqa: SIM115 - the ExitStack closes it
            file = self._files.enter_context(opened)
            self._open[record.position] = file
        raw = payload(file, shard.path, self._read.index[record.position], record.offset)
        return _Frame(shard.shard, record.offset, raw)

    def get_state(self) -> dict[str, Any]:
        """The units of the run this iterator has passed."""
        return {"units": self._unit}

    def set_state(self, state: dict[str, Any]) -> None:
        """Not supported: a stream's exact resume is the host stage's.

        Args:
            state: The state to restore.

        Raises:
            NotImplementedError: Always.
        """
        del state
        raise NotImplementedError(
            "a TFDS stream resumes through the pipeline's host stage, which restores its order"
        )


class TFDSStreamingSource(DatasetSourceMixin, StreamingSourceBase):
    """A TFDS split prepared as TFRecord, streamed without TensorFlow, named by ``tfds_id``.

    An offset index of each shard file is built once from its frame headers. Each pass computes
    its order over record ids, reads each record's payload at its offset, checks the frame's CRCs
    as tf.data's TFRecord reader does, and decodes it last with TFDS's NumPy decoder: numeric
    features become host NumPy columns, text and other objects the record's provenance, beside
    the batch. A record is named by the id TFDS reports for it,
    ``tfds_id``: its shard file and its offset in the file, as the two words ``(shard, offset)``
    (``STREAM_IDS``), where the shard is the file's place among the dataset's files in name order
    (:attr:`shard_files`). A slice names its records as the full split does.

    The order is the pipeline's to choose. Without its key a pass reads the files in order. With
    it, pass ``p`` is ordered as TFDS's training read (``shuffle_files=True`` and
    ``shuffle(shuffle_buffer_size)``) orders an epoch: the files in a keyed order, interleaved 16 at
    a time in blocks of 16, through tf.data's buffer shuffle, every draw from a NumPy Philox
    generator keyed by ``fold_in(key, p)``. The buffer holds ``shuffle_buffer_size`` record ids
    (a shard position and an offset each), not records: a record is read only when served.

    A record's provenance is also looked up by its id, ``provenance(indices)``, reading the record
    again at its offset. TensorFlow is never imported. The source never prepares a dataset:
    a copy that is not prepared, or is prepared as ArrayRecord (which ``TFDSEagerSource`` reads),
    is refused, naming the call that prepares a TFRecord copy.

    Example:
        ```python
        source = TFDSStreamingSource(TFDSStreamingConfig(name="imagenet2012", split="train"))
        pipeline = Pipeline(source=source, stages=[], batch_size=256, rngs=nnx.Rngs(0),
                            shuffle=True, num_epochs=None)
        for batch in pipeline:
            train_step(batch)
        ```
    """

    config: TFDSStreamingConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def record_identity(self) -> RecordIdentity:
        """A TFDS record's id is its shard file and offset: the stream's own names."""
        return RecordIdentity.STREAM_IDS

    @property
    def host_read(self) -> HostRead:
        """TFDS's NumPy decode holds the GIL: under a budget, worker processes read the run."""
        return HostRead.GIL_BOUND

    def __init__(  # noqa: DOC502 - prepared_builder, supervised_keys and split_shard raise
        self, config: TFDSStreamingConfig, *, name: str | None = None
    ) -> None:
        """Open the prepared TFRecord split, reading no record.

        Args:
            config: Configuration for the source
            name: Optional name (defaults to TFDSStreamingSource(dataset:split))

        Raises:
            FileNotFoundError: If the dataset is not prepared as TFRecord in the data directory.
            ValueError: If ``as_supervised`` is asked of a dataset without supervised keys, or a
                record the split reads sits at an offset an id's low word cannot hold.
        """
        if name is None:
            name = f"TFDSStreamingSource({config.name}:{config.split})"
        super().__init__(config, name=name)
        dataset, split = config.name, config.split
        assert dataset is not None and split is not None  # noqa: S101 (invariant, not control flow)
        builder = prepared_builder(
            dataset,
            config.data_dir,
            "tfrecord",
            lambda found: _not_streamable(dataset, config.data_dir, found),
        )
        info = builder.info
        self.dataset_name = dataset
        self.split_name = split
        self._dataset = dataset
        self._dataset_info = HostValue(info)
        self._kept = HostValue(_KeptFeatures.of_config(config, info, dataset))
        files = {
            Path(instruction.filename).name: instruction
            for split_name in info.splits
            for instruction in info.splits[split_name].file_instructions
        }
        self._files = HostValue(tuple(sorted(files.items())))
        shard_of = {basename: shard for shard, (basename, _) in enumerate(self._files.value)}
        self._shards = HostValue(
            tuple(
                split_shard(
                    instruction.filename,
                    shard_of[Path(instruction.filename).name],
                    instruction.skip,
                    instruction.take,
                )
                for instruction in info.splits[split].file_instructions
            )
        )
        self._index = HostValue({})
        # The split's index as the one copy every worker of a run reads, made once it is needed.
        self._shared_index = HostValue(None)
        self.length = sum(shard.take for shard in self._shards.value)

    @property
    def shard_files(self) -> tuple[str, ...]:
        """The dataset's shard files by shard number: the file part of each record's ``tfds_id``."""
        return tuple(basename for basename, _ in self._files.value)

    def __len__(self) -> int:
        """The records one pass serves."""
        return self.length

    def _repr_extra_fields(self) -> dict[str, Any]:
        """The buffer size in the representation."""
        return {"shuffle_buffer_size": self.config.shuffle_buffer_size}

    def pass_dataset(
        self, pass_index: int, key: np.ndarray | None, batch_size: int
    ) -> TFDSStreamDataset:
        """Pass ``pass_index`` as a Grain dataset of decoded batches, a :class:`TFDSStreamDataset`.

        A run of one pass from its start, in batches of ``batch_size``, the last one short.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key as its uint32 words on the host
                (:func:`~datarax.core.prng.key_words`) when it shuffles, ``None`` for file
                order.
            batch_size: Records per decoded batch.

        Returns:
            The pass's dataset.
        """
        return TFDSStreamDataset(
            self._stream_read(_PassUnits(pass_index, self.length, batch_size), key)
        )

    def run_dataset(
        self, schedule: BatchSchedule, key: np.ndarray | None, *, for_workers: bool = False
    ) -> TFDSStreamDataset:
        """A run of passes as one Grain dataset of decoded units, numbered from the run's start.

        Args:
            schedule: The run's units, each a run of batches ``(start, pass, size)``.
            key: The pipeline's key as host words, or ``None`` for file order.
            for_workers: Whether worker processes read the run: its offset index then goes to
                them as one shared read-only copy, made once for the source, so a worker holds
                none of its own.

        Returns:
            The run's dataset (:class:`TFDSStreamDataset`).
        """
        return TFDSStreamDataset(self._stream_read(schedule, key, for_workers=for_workers))

    def _stream_read(
        self, schedule: BatchSchedule, key: np.ndarray | None, *, for_workers: bool = False
    ) -> _StreamRead:
        """What a run reads, as plain values, the shards' record index included."""
        config = self.config
        index = (
            self._shared()
            if for_workers
            else tuple(self._shard_index(shard.shard) for shard in self._shards.value)
        )
        return _StreamRead(
            name=self._dataset,
            data_dir=config.data_dir,
            shards=self._shards.value,
            index=index,
            length=self.length,
            key=None if key is None else np.asarray(key, np.uint32),
            buffer_size=config.shuffle_buffer_size,
            schedule=schedule,
            kept=self._kept.value,
        )

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index``: in file order, or as TFDS's training read under ``key``.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key as its host words when it shuffles, ``None`` for file order.
            read_size: Records decoded together.

        Yields:
            The pass's records as decoded chunks.
        """
        for columns, provenance, ids, _, _ in self.pass_dataset(pass_index, key, read_size):
            yield StreamChunk(columns, tuple(MappingProxyType(p) for p in provenance), ids)

    def provenance(  # noqa: DOC502 - record_words, refuse_unnamed and _frame_at raise
        self, indices: ArrayLike
    ) -> Provenance:
        """The provenance of the records ``indices`` names, read again by their ids.

        Each named record is read at its offset in its shard file (an offset index of the file
        is built on its first lookup) and decoded; its non-array features are its provenance.

        Args:
            indices: uint32 ``(n, 2)`` ids ``(shard, offset)``, as the stream's batches hold them.

        Returns:
            One mapping per index, in the order named.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            DamagedRecordError: If a named record's frame is damaged.
            IndexError: If an index is the padding index, or names no shard or no record in it.
        """
        words = record_words(indices)
        refuse_unnamed(words)
        frames = [self._frame_at(int(shard), int(offset)) for shard, offset in words]
        features = self._dataset_info.value.features
        return tuple(
            MappingProxyType(record)
            for record in _decoded_batch(features, frames, self._kept.value)[1]
        )

    def _shared(self) -> tuple[ShardIndex, ...]:
        """The split's index as one read-only copy in shared memory, the index this source reads.

        Made once: the indexes this process built are written to
        one file and replaced by views of it, so the host holds one copy, which every worker maps
        instead of receiving its own; the file goes with the source.

        Returns:
            Each of the split's shards' index, in shard order, as a view of the shared copy.
        """
        shared = self._shared_index.value
        if shared is None:
            indexes = [self._shard_index(shard.shard) for shard in self._shards.value]
            path, shared = share_index(indexes)
            for shard, index in zip(self._shards.value, shared, strict=True):
                self._index.value[shard.shard] = index
            self._shared_index.value = shared
            weakref.finalize(self, remove, path)
        return shared

    def _shard_index(self, shard: int) -> ShardIndex:
        """Shard ``shard``'s record index, built from its frame headers on first use, then kept."""
        index = self._index.value.get(shard)
        if index is None:
            index = self._index.value.setdefault(
                shard, record_index(self._files.value[shard][1].filename)
            )
        return index

    def _frame_at(self, shard: int, offset: int) -> _Frame:  # noqa: DOC503 - payload raises
        """The serialized record at ``offset`` of shard ``shard``.

        Args:
            shard: The shard number.
            offset: The record's offset in the shard file.

        Returns:
            The record and its id.

        Raises:
            IndexError: If the dataset has no such shard, or the shard no such record.
            DamagedRecordError: If the shard file is damaged at the record's frame.
        """
        files = self._files.value
        if shard >= len(files):
            raise IndexError(f"shard {shard} names no file: the dataset has {len(files)} shards")
        basename, instruction = files[shard]
        index = self._shard_index(shard)
        if offset >= len(index.lengths):
            raise IndexError(
                f"offset {offset} names no record of shard {shard} ({basename}), which holds "
                f"{len(index.lengths)}"
            )
        with Path(instruction.filename).open("rb", buffering=0) as file:
            raw = payload(file, instruction.filename, index, offset)
        return _Frame(shard, offset, raw)
