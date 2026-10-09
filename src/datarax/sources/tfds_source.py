"""TensorFlow Datasets (TFDS) sources for Datarax, both read without TensorFlow.

**TFDSEagerSource** reads a split that TFDS has prepared as ArrayRecord, whole, into host NumPy
columns and per-record provenance. TFDS's random-access reader (``builder.as_data_source``) reads
every record in one batched call and decodes it with NumPy and Pillow; the records then become
columns once, through the path every eager source shares.

- Holds no iteration state; the pipeline owns the order and position
- Strings a record carries (an ``id``, a caption) are kept as its provenance, never served
- Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**TFDSStreamingSource** streams a split TFDS has prepared as TFRecord (TFDS's default format), for
datasets too large for host memory. An index of each shard file, built once from its frame headers,
gives every record's byte offset; a pass's order is computed over record ids, each record's payload
is read at its offset and checked against its CRC, and TFDS's NumPy decoder decodes it last.
Records are named by the id TFDS reports for them, ``tfds_id``: the shard file and the record's
offset in it (``STREAM_IDS``). Given the pipeline's key, a pass is ordered as TFDS's training read
orders it: the shard files in a keyed order, read 16 at a time in blocks of 16, through tf.data's
buffer shuffle, the buffer holding record ids.

TensorFlow is never imported: TensorFlow in a JAX process breaks JAX's NCCL collectives. Neither
source prepares a dataset, because preparing imports TensorFlow: a split that is not prepared, or
is prepared only in the format the other source reads, is refused, naming the call that prepares
it or the source that reads it. Reading needs the ``data`` extra; preparing needs the ``tfds``
extra, in a process of its own.

``from_tfds`` picks between the two sources by the format the copy is prepared in.

This module is TFDS's integration module and imports ``tensorflow_datasets`` at its top, so
importing it without the ``data`` extra raises ``ImportError``. ``datarax.sources`` exports these
names lazily, so ``import datarax.sources`` imports no TFDS. The import happens on the line that
imports a TFDS source or ``from_tfds``. TFDS's first import enters ``etils.epy.lazy_imports()``,
which replaces ``builtins.__import__`` for the whole process while it runs. A garbage collection
in that window runs finalizers whose imports fail, so the window must not open inside a
constructor in the middle of a run.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import struct
import weakref
from collections.abc import Callable, Iterator, Sequence, Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol

import google_crc32c
import grain
import jax
import numpy as np
import tensorflow_datasets as tfds
from jax.typing import ArrayLike

from datarax.core.data_source import (
    BatchSchedule,
    DataSourceModule,
    Provenance,
    record_words,
    RecordIdentity,
    refuse_padding,
)
from datarax.sources._config_base import SourceConfigBase, StreamingSourceConfigBase
from datarax.sources._source_base import (
    DatasetSourceMixin,
    pass_generator,
    StreamChunk,
    StreamingSourceBase,
)
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule
from datarax.sources.eager_source import EagerSource, HostValue, parts_of_records
from datarax.sources.source_ops import filter_keys


# =============================================================================
# Opening a prepared copy
# =============================================================================


def _prepared_builder(  # noqa: DOC503 - the exception raised is the one ``refuse`` builds
    name: str, data_dir: str | None, file_format: str, refuse: Callable[[str], Exception]
) -> Any:
    """The builder of ``name`` as prepared in ``data_dir`` in ``file_format``, reading no record.

    Args:
        name: The TFDS dataset name, with its config when it has one.
        data_dir: The data directory; TFDS's default when ``None``.
        file_format: The format the copy must be prepared in (``"array_record"``, ``"tfrecord"``).
        refuse: Builds the error refusing a copy, from what was found instead.

    Returns:
        The TFDS builder.

    Raises:
        Exception: What ``refuse`` builds, if the copy is not prepared in ``file_format``.
    """
    try:
        builder = tfds.builder(name, data_dir=data_dir)
    except tfds.core.DatasetNotFoundError as error:
        raise refuse("it is not prepared there") from error
    if not builder.is_prepared():
        raise refuse(f"{builder.data_dir} is not prepared")
    formats = builder.info.available_file_formats()
    if tfds.core.FileFormat(file_format) not in formats:
        found = ", ".join(sorted(fmt.value for fmt in formats))
        raise refuse(f"{builder.data_dir} holds it prepared as {found}")
    return builder


def _tfrecord_only(name: str, data_dir: str | None) -> bool:
    """Whether ``name`` is prepared in ``data_dir`` as TFRecord and not as ArrayRecord.

    Such a copy has no random access, so it is streamed; any other (ArrayRecord, or nothing
    prepared) goes to the eager source, which reads ArrayRecord or names the call preparing it.

    Args:
        name: The TFDS dataset name.
        data_dir: The data directory; TFDS's default when ``None``.

    Returns:
        Whether the prepared copy is TFRecord only.
    """
    try:
        builder = tfds.builder(name, data_dir=data_dir)
    except tfds.core.DatasetNotFoundError:
        return False
    if not builder.is_prepared():
        return False
    formats = {fmt.value for fmt in builder.info.available_file_formats()}
    return "tfrecord" in formats and "array_record" not in formats


def _prepare_call(name: str, data_dir: str | None, file_format: str) -> str:
    return (
        f"tfds.builder({name!r}, data_dir={data_dir!r}, file_format={file_format!r})"
        ".download_and_prepare()"
    )


def _where(data_dir: str | None) -> str:
    return data_dir if data_dir is not None else "TFDS's default data directory"


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class TFDSEagerConfig(SourceConfigBase):
    """Configuration for TFDSEagerSource (reads a prepared ArrayRecord split into host columns).

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to load, e.g., "train", "test[:2000]" (required)
        data_dir: Directory holding the dataset prepared as ArrayRecord; TFDS's default data
            directory (``TFDS_DATA_DIR``, else ``~/tensorflow_datasets``) when ``None``
        as_supervised: If True, keeps only the dataset's supervised features
            (``info.supervised_keys``), under their own names
        include_keys: Optional set of keys to include in output (exclusive with exclude_keys)
        exclude_keys: Optional set of keys to exclude from output (exclusive with include_keys)

    Note:
        The order records are served in belongs to the pipeline (``Pipeline(shuffle=...)``).
        The source never prepares a dataset; see :class:`TFDSEagerSource`.
    """

    as_supervised: bool = False


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


# =============================================================================
# TFDSEagerSource - Reads a Prepared ArrayRecord Split into Host Columns at Init
# =============================================================================


class PreparedRecords(Protocol):
    """A prepared split's records, read by position: TFDS's random-access data source."""

    def __len__(self) -> int:
        """The number of records in the split."""
        ...

    def __getitems__(self, keys: Sequence[int]) -> Sequence[Any]:
        """Read and decode the records at ``keys``, in one batched read."""
        ...


def _not_prepared(name: str, data_dir: str | None, found: str) -> FileNotFoundError:
    """The refusal of a copy the eager source cannot read, naming the call that prepares one."""
    return FileNotFoundError(
        f"TFDS dataset {name!r} is not prepared as ArrayRecord in {_where(data_dir)}: {found}. "
        "TFDSEagerSource reads a copy prepared as ArrayRecord and never prepares one, since "
        "preparing imports TensorFlow; a copy prepared as TFRecord is streamed by "
        "TFDSStreamingSource. Prepare it once, in a process of its own with the tfds extra "
        f"installed: {_prepare_call(name, data_dir, 'array_record')}. A data directory holds one "
        "format per dataset version, so prepare it where no copy of that version in another "
        "format is."
    )


def _array_record_builder(name: str, data_dir: str | None) -> Any:  # noqa: DOC502
    """The builder of ``name`` prepared as ArrayRecord in ``data_dir``, refused otherwise.

    Args:
        name: The TFDS dataset name, with its config when it has one.
        data_dir: The data directory; TFDS's default when ``None``.

    Returns:
        The TFDS builder.

    Raises:
        FileNotFoundError: If the data directory holds no copy prepared as ArrayRecord.
    """
    return _prepared_builder(
        name, data_dir, "array_record", lambda found: _not_prepared(name, data_dir, found)
    )


def open_prepared_split(  # noqa: DOC502 - _prepared_builder raises the FileNotFoundError
    name: str, split: str, data_dir: str | None
) -> tuple[Any, PreparedRecords]:
    """Open ``split`` of the TFDS dataset ``name`` as prepared in ArrayRecord, reading no record.

    TFDS's own reader serves it (``builder.as_data_source``), and TensorFlow is not imported.

    Args:
        name: The TFDS dataset name, with its config when it has one (``"nsynth/full"``).
        split: The split, slices included (``"train[:5000]"``).
        data_dir: The data directory; TFDS's default when ``None``.

    Returns:
        The dataset's ``DatasetInfo`` and the split's records.

    Raises:
        FileNotFoundError: If the data directory holds no copy of the dataset prepared as
            ArrayRecord: nothing prepared, or a copy in another format such as TFRecord.
    """
    builder = _array_record_builder(name, data_dir)
    # tfds wraps as_data_source in a logging decorator whose type hides the parameters.
    records = builder.as_data_source(
        split=split,  # pyright: ignore[reportCallIssue]
        file_format="array_record",
    )
    return builder.info, records


def _kept_features(
    record: dict[str, Any],
    keys: Sequence[str] | None,
    include_keys: AbstractSet[str] | None,
    exclude_keys: AbstractSet[str] | None,
) -> dict[str, Any]:
    """The features of ``record`` a source keeps: the supervised ones if asked, then filtered."""
    if keys is not None:
        record = {key: record[key] for key in keys}
    return filter_keys(record, include_keys, exclude_keys)


def _supervised_keys(info: Any, dataset: str) -> list[str]:
    """The features ``as_supervised`` keeps: those the dataset declares supervised.

    Args:
        info: The dataset's ``DatasetInfo``.
        dataset: The dataset's name, for the refusal.

    Returns:
        The supervised features' names.

    Raises:
        ValueError: If the dataset declares no supervised keys.
    """
    declared = info.supervised_keys
    if declared is None:
        raise ValueError(f"{dataset} declares no supervised keys, which as_supervised keeps")
    return [str(key) for key in jax.tree.leaves(declared)]


@dataclass(frozen=True, slots=True)
class _ExampleDecoder:
    """Decodes a batch of a prepared split's serialized examples as TFDS's reader decodes one.

    Each record goes through the dataset's ``features.deserialize_example_np``, as
    ``builder.as_data_source`` does, then keeps the features the source keeps
    (:func:`_kept_features`), so a per-batch read holds what the eager source holds. It pickles
    with the features, for a worker process.
    """

    features: Any
    keys: tuple[str, ...] | None
    include_keys: AbstractSet[str] | None
    exclude_keys: AbstractSet[str] | None

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """One kept mapping of decoded features per record."""
        return [
            _kept_features(
                self.features.deserialize_example_np(record),
                self.keys,
                self.include_keys,
                self.exclude_keys,
            )
            for record in records
        ]


def _per_batch_split(  # noqa: DOC502 - _array_record_builder and _supervised_keys raise
    name: str,
    split: str,
    *,
    data_dir: str | None = None,
    as_supervised: bool = False,
    include_keys: AbstractSet[str] | None = None,
    exclude_keys: AbstractSet[str] | None = None,
) -> ArrayRecordSourceModule:
    """A split prepared as ArrayRecord, read and decoded per batch instead of held in memory.

    The source reads the split's records where TFDS's own reader would (the split's
    ``file_instructions``, slices included) and decodes each batch with TFDS's decoder, keeping
    the features ``TFDSEagerSource`` keeps; it holds no record between reads. TensorFlow is not
    imported.

    Args:
        name: The TFDS dataset name, with its config when it has one.
        split: The split, slices included (``"train[:5000]"``).
        data_dir: The data directory; TFDS's default when ``None``.
        as_supervised: If True, keeps only the supervised features, under their own names.
        include_keys: Optional set of keys to include.
        exclude_keys: Optional set of keys to exclude.

    Returns:
        An ``ArrayRecordSourceModule`` over the split.

    Raises:
        FileNotFoundError: If the dataset is not prepared as ArrayRecord in the data directory.
        ValueError: If ``as_supervised`` is asked of a dataset without supervised keys.
    """
    info = _array_record_builder(name, data_dir).info
    keys = tuple(_supervised_keys(info, name)) if as_supervised else None
    return ArrayRecordSourceModule(
        ArrayRecordSourceConfig(),
        info.splits[split].file_instructions,
        decode=_ExampleDecoder(info.features, keys, include_keys, exclude_keys),
    )


class TFDSEagerSource(DatasetSourceMixin, EagerSource):
    """Eager TFDS source: a split prepared as ArrayRecord, read whole into host columns.

    At construction it opens the split with TFDS's random-access reader, reads every record in
    one batched read and stores the numeric features as host NumPy columns and every other
    feature (text, such as CIFAR-10's ``id``) as each record's provenance; it then serves them as
    every eager source does (:class:`~datarax.sources.eager_source.EagerSource`): indexing,
    iteration in order and the stateless host read ``get_batch(indices, epochs=...)``. Values keep
    the dtype TFDS stores (a class label is int64 on the host; a device holds it as int32 while
    64-bit types are off).

    TensorFlow is never imported. The source never prepares a dataset: a split that is not
    prepared, or is prepared in another format, is refused with the call that prepares it, to be
    run once in a process of its own (preparing imports TensorFlow)::

        tfds.builder("mnist", data_dir=..., file_format="array_record").download_and_prepare()

    Example:
        ```python
        config = TFDSEagerConfig(name="mnist", split="train")
        source = TFDSEagerSource(config)

        for item in source:  # records in order
            process(item["image"])

        batch = source.get_batch(to_words(np.arange(32)))  # records 0..31 as a Batch
        ```
    """

    def __init__(  # noqa: DOC503 - open_prepared_split raises the FileNotFoundError
        self,
        config: TFDSEagerConfig,
        *,
        name: str | None = None,
    ) -> None:
        """Read the prepared split into host columns and provenance.

        Args:
            config: Configuration for the source
            name: Optional name (defaults to TFDSEagerSource(dataset:split))

        Raises:
            FileNotFoundError: If the dataset is not prepared as ArrayRecord in the data directory.
            ValueError: If ``as_supervised`` is asked of a dataset without supervised keys, the
                split and key filters leave nothing, or a feature's shape varies between records.
        """
        if name is None:
            name = f"TFDSEagerSource({config.name}:{config.split})"
        super().__init__(config, name=name)
        self.dataset_name = config.name
        self.split_name = config.split
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # name and split are validated non-None by the config's __post_init__
        dataset, split = config.name, config.split
        assert dataset is not None and split is not None  # noqa: S101 (invariant, not control flow)
        info, records = open_prepared_split(dataset, split, config.data_dir)
        self._dataset_info = HostValue(info)
        keys = self._supervised_keys(dataset) if config.as_supervised else None
        rows = [
            _kept_features(record, keys, config.include_keys, config.exclude_keys)
            for record in records.__getitems__(list(range(len(records))))
        ]
        if not rows or not rows[0]:
            raise ValueError(
                f"{dataset} {split} produced no records after loading and filtering; check the "
                "split and the include/exclude key filters"
            )
        self._store(*parts_of_records(rows))

    def _supervised_keys(self, dataset: str) -> list[str]:
        """The features ``as_supervised`` keeps: those the dataset declares supervised."""
        return _supervised_keys(self.get_dataset_info(), dataset)


# =============================================================================
# TFDSStreamingSource - Streams a Prepared TFRecord Split
# =============================================================================


_LARGEST_OFFSET = 2**32 - 1
"""The largest record offset in a shard file: an id keeps the offset in its low word."""

_CYCLE_LENGTH = 16
"""Shard files read at once when a pass is shuffled: TFDS's ``interleave_cycle_length``."""

_BLOCK_LENGTH = 16
"""Records read from one file before the next: TFDS's ``interleave_block_length``."""

_TFRECORD_HEADER = struct.Struct("<QI")
"""A TFRecord frame's header: the payload's length (uint64) and its masked CRC (uint32)."""

_TFRECORD_CRC = struct.Struct("<I")
"""A TFRecord frame's trailer, after the payload: the payload's masked CRC (uint32)."""

_CRC_MASK_DELTA = 0xA282EAD8
"""The constant TFRecord adds to a rotated CRC32C to mask it (TensorFlow's ``crc32c::Mask``)."""


@dataclass(frozen=True, slots=True)
class _Shard:
    """The records of one shard file a split reads: offsets ``skip .. skip + take``."""

    path: str
    shard: int
    skip: int
    take: int


class _Frame(NamedTuple):
    """One serialized record and the id it is named by."""

    shard: int
    offset: int
    raw: bytes


def _not_streamable(name: str, data_dir: str | None, found: str) -> FileNotFoundError:
    """The refusal of a copy the stream cannot read, naming what reads it or prepares one."""
    return FileNotFoundError(
        f"TFDS dataset {name!r} is not prepared as TFRecord in {_where(data_dir)}: {found}. "
        "TFDSStreamingSource streams a copy prepared as TFRecord, TFDS's default format, and "
        "never prepares one, since preparing imports TensorFlow. A copy prepared as ArrayRecord "
        "(array_record) has random access: TFDSEagerSource reads it into host memory, and a copy "
        "larger than host memory is read by index once the pipeline's host stage serves indexed "
        "sources. To stream, prepare a TFRecord copy once, in a process of its own with the tfds "
        f"extra installed: {_prepare_call(name, data_dir, 'tfrecord')}."
    )


class _RecordId(NamedTuple):
    """A record of a pass: the position of its shard in the pass's read, and its offset there."""

    position: int
    offset: int


@dataclass(frozen=True, slots=True, eq=False)
class ShardIndex:
    """Where each record of a TFRecord shard file sits: its payload's byte offset and length.

    Attributes:
        payloads: int64 byte offset of each record's payload in the file, by record offset.
        lengths: int64 byte length of each record's payload, by record offset.
    """

    payloads: np.ndarray
    lengths: np.ndarray


def _pread(file: Any, size: int, position: int) -> bytes:
    """Read ``size`` bytes of an open file from ``position``, with no buffering around it."""
    return os.pread(file.fileno(), size, position)


def _masked_crc(data: bytes) -> int:
    """TFRecord's masked CRC32C of ``data``: rotated right by 15 bits, plus a constant."""
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + _CRC_MASK_DELTA) & 0xFFFFFFFF


class DamagedRecordError(ValueError):
    """A TFRecord frame that fails tf.data's checks: its CRCs, or a file ending inside it."""

    def __init__(self, path: str, record: int, damage: str) -> None:
        """Name the shard file, the record and the damage.

        Args:
            path: The shard file.
            record: The record's offset in it.
            damage: What is wrong with the frame.
        """
        super().__init__(f"shard file {path}: record {record} {damage}; the file is damaged")


def _record_index(path: str) -> ShardIndex:
    """The payload offset and length of every record in a TFRecord file, from its frame headers.

    Neither TFDS, Grain nor ArrayRecord indexes a TFRecord file (searched: ``tfds.core.reader``,
    ``grain.experimental.TFRecordIterDataset``, ``array_record``), so the 12-byte frame headers
    are read here, each with one unbuffered positioned read, the payloads skipped. Each length is
    checked against its masked CRC32C and each frame against the file's size, as tf.data's
    TFRecord reader checks them.

    Args:
        path: The shard file.

    Returns:
        The shard's index.

    Raises:
        DamagedRecordError: If a frame's length fails its CRC check or the file ends inside it.
    """
    payloads, lengths, position = [], [], 0
    with Path(path).open("rb", buffering=0) as file:
        size = os.fstat(file.fileno()).st_size
        while header := _pread(file, _TFRECORD_HEADER.size, position):
            record = len(lengths)
            if len(header) < _TFRECORD_HEADER.size:
                raise DamagedRecordError(
                    path, record, f"is cut short: the file ends at byte {size}"
                )
            length, length_crc = _TFRECORD_HEADER.unpack(header)
            if _masked_crc(header[:8]) != length_crc:
                raise DamagedRecordError(path, record, "has a length that fails its CRC check")
            end = position + _TFRECORD_HEADER.size + length + _TFRECORD_CRC.size
            if end > size:
                raise DamagedRecordError(
                    path, record, f"is cut short: the file ends at byte {size}"
                )
            payloads.append(position + _TFRECORD_HEADER.size)
            lengths.append(length)
            position = end
    return ShardIndex(np.asarray(payloads, np.int64), np.asarray(lengths, np.int64))


def _payload(file: Any, path: str, index: ShardIndex, offset: int) -> bytes:
    """The payload of record ``offset`` of an open shard file, checked against its CRC.

    One positioned read takes the payload and the CRC after it.

    Args:
        file: The shard file, open unbuffered.
        path: Its path, for the refusal.
        index: Its record index.
        offset: The record's offset in the file.

    Returns:
        The serialized record.

    Raises:
        DamagedRecordError: If the payload fails its CRC check.
    """
    length = int(index.lengths[offset])
    frame = _pread(file, length + _TFRECORD_CRC.size, int(index.payloads[offset]))
    payload = frame[:length]
    if (
        len(frame) != length + _TFRECORD_CRC.size
        or _masked_crc(payload) != (_TFRECORD_CRC.unpack_from(frame, length)[0])
    ):
        raise DamagedRecordError(path, offset, "has a payload that fails its CRC check")
    return payload


def _split_shard(path: str, shard: int, skip: int, take: int) -> _Shard:
    """The records ``skip .. skip + take`` a split reads of a shard file, as a :class:`_Shard`.

    Args:
        path: The shard file.
        shard: Its number among the dataset's files.
        skip: The split's first record offset in the file.
        take: How many records the split reads of it.

    Returns:
        The shard.

    Raises:
        ValueError: If the split's last offset does not fit an id's low word.
    """
    last = skip + take - 1
    if last > _LARGEST_OFFSET:
        raise ValueError(
            f"shard file {path}: the record at offset {last} does not fit an id, whose low "
            f"word holds a record's offset in its shard, at most {_LARGEST_OFFSET}"
        )
    return _Shard(path, shard, skip, take)


def _shard_ids(position: int, shard: _Shard) -> Iterator[_RecordId]:
    """The records ``shard`` holds for the split, in file order, as ids."""
    return (_RecordId(position, offset) for offset in range(shard.skip, shard.skip + shard.take))


def _interleaved(streams: Sequence[Iterator[_RecordId]]) -> Iterator[_RecordId]:
    """The records of ``streams``, one per shard, as TFDS's training read interleaves its files.

    ``_CYCLE_LENGTH`` files are open at once and ``_BLOCK_LENGTH`` records are taken from each in
    turn; a file that ends is replaced by the next one in its place in the cycle (tf.data's
    ``interleave(cycle_length, block_length)``, which ``tfds.core.reader`` applies).
    """
    pending = iter(streams)
    cycle = list(itertools.islice(pending, _CYCLE_LENGTH))
    while cycle:
        slot = 0
        while slot < len(cycle):
            block = list(itertools.islice(cycle[slot], _BLOCK_LENGTH))
            yield from block
            if len(block) < _BLOCK_LENGTH:
                following = next(pending, None)
                if following is None:
                    del cycle[slot]
                    continue
                cycle[slot] = following
            slot += 1


def _buffer_shuffled(
    ids: Iterator[_RecordId], size: int, generator: np.random.Generator
) -> Iterator[_RecordId]:
    """``ids`` through tf.data's buffer shuffle of ``size`` records.

    The buffer fills with the first ``size`` records; then each record served is a uniform pick
    from the buffer, whose place the next record takes; at the end the buffer is served in
    uniform picks. The buffer holds record ids, so its state is the ids and the generator's.
    """
    buffer = list(itertools.islice(ids, size))
    for record in ids:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = record
    while buffer:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = buffer[-1]
        buffer.pop()


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
    def of_config(  # noqa: DOC502 - _supervised_keys raises
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
            keys=tuple(_supervised_keys(info, dataset)) if config.as_supervised else None,
            include_keys=None if include is None else frozenset(include),
            exclude_keys=None if exclude is None else frozenset(exclude),
        )

    def of(self, record: dict[str, Any]) -> dict[str, Any]:
        """The features of ``record`` kept."""
        return _kept_features(
            record,
            self.keys,
            None if self.include_keys is None else set(self.include_keys),
            None if self.exclude_keys is None else set(self.exclude_keys),
        )


@dataclass(frozen=True, slots=True)
class StreamRead:
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
    shards: tuple[_Shard, ...]
    index: tuple[ShardIndex, ...]
    length: int
    key: np.ndarray | None
    buffer_size: int
    schedule: BatchSchedule
    kept: _KeptFeatures


def _ordered_ids(read: StreamRead, pass_index: int) -> Iterator[_RecordId]:
    """Pass ``pass_index``'s records in its order, as ids: file order, or TFDS's training read's.

    The same order every worker of the run computes, from the read alone, with no file read.
    """
    streams = [_shard_ids(position, shard) for position, shard in enumerate(read.shards)]
    if read.key is None:
        return itertools.chain.from_iterable(streams)
    generator = pass_generator(read.key, pass_index)
    order = generator.permutation(len(streams))
    return _buffer_shuffled(_interleaved([streams[i] for i in order]), read.buffer_size, generator)


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
    index and decodes them last. An element is the unit's host columns, provenance, ids and each
    record's pass. Units are numbered from the run's first, wherever the run starts, and records
    before the run's start are skipped as ids, never read. It implements Grain's slicing hook:
    with ``set_slice(slice(i, None, k))`` the iterator walks the same schedule but reads and
    decodes only units ``j`` with ``j % k == i``, so ``k`` such slices interleaved round robin from
    the first give the unsliced run, the stream's order not depending on ``k``, and the slices
    read each record once between them. That is how Grain's process prefetch splits a dataset
    across workers. The dataset pickles (its read is plain values, the offset index included),
    and each iterator opens its own files and decoder, so iterators in threads or processes share
    no lazy state.
    """

    def __init__(self, read: StreamRead) -> None:
        """Hold the run's read; nothing is opened until iteration.

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


type _Unit = tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray, np.ndarray]


class _TFDSStreamIterator(grain.DatasetIterator):
    """Units of a run of a TFDS stream, those of its slice read and decoded."""

    def __init__(self, read: StreamRead, sl: slice) -> None:
        super().__init__()
        self._read = read
        self._features = _features(read.name, read.data_dir)
        self._start, self._step = sl.start or 0, sl.step or 1
        self._unit = 0
        self._pass: int | None = None
        self._ids: Iterator[_RecordId] = iter(())
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
                return columns, provenance, ids, epochs

    def _take(self, start: int, pass_index: int, size: int) -> list[tuple[_RecordId, int]]:
        """The ids of ``size`` positions from ``start`` of pass ``pass_index`` on, across passes."""
        taken: list[tuple[_RecordId, int]] = []
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

    def _frame(self, record: _RecordId) -> _Frame:
        """The record's payload, read at its offset, and its id."""
        shard = self._read.shards[record.position]
        file = self._open.get(record.position)
        if file is None:
            opened = Path(shard.path).open("rb", buffering=0)  # noqa: SIM115 - the ExitStack closes it
            file = self._files.enter_context(opened)
            self._open[record.position] = file
        raw = _payload(file, shard.path, self._read.index[record.position], record.offset)
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

    def __init__(  # noqa: DOC502 - _prepared_builder, _supervised_keys and _split_shard raise
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
        builder = _prepared_builder(
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
                _split_shard(
                    instruction.filename,
                    shard_of[Path(instruction.filename).name],
                    instruction.skip,
                    instruction.take,
                )
                for instruction in info.splits[split].file_instructions
            )
        )
        self._index = HostValue({})
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

    def run_dataset(self, schedule: BatchSchedule, key: np.ndarray | None) -> TFDSStreamDataset:
        """A run of passes as one Grain dataset of decoded units, numbered from the run's start.

        Args:
            schedule: The run's units, each a run of batches ``(start, pass, size)``.
            key: The pipeline's key as host words, or ``None`` for file order.

        Returns:
            The run's dataset (:class:`TFDSStreamDataset`).
        """
        return TFDSStreamDataset(self._stream_read(schedule, key))

    def _stream_read(self, schedule: BatchSchedule, key: np.ndarray | None) -> StreamRead:
        """What a run reads, as plain values, the shards' record index included."""
        config = self.config
        return StreamRead(
            name=self._dataset,
            data_dir=config.data_dir,
            shards=self._shards.value,
            index=tuple(self._shard_index(shard.shard) for shard in self._shards.value),
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
        for columns, provenance, ids, _ in self.pass_dataset(pass_index, key, read_size):
            yield StreamChunk(columns, tuple(MappingProxyType(p) for p in provenance), ids)

    def provenance(  # noqa: DOC502 - record_words, refuse_padding and _frame_at raise
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
        refuse_padding(words)
        frames = [self._frame_at(int(shard), int(offset)) for shard, offset in words]
        features = self._dataset_info.value.features
        return tuple(
            MappingProxyType(record)
            for record in _decoded_batch(features, frames, self._kept.value)[1]
        )

    def _shard_index(self, shard: int) -> ShardIndex:
        """Shard ``shard``'s record index, built from its frame headers on first use, then kept."""
        index = self._index.value.get(shard)
        if index is None:
            index = self._index.value.setdefault(
                shard, _record_index(self._files.value[shard][1].filename)
            )
        return index

    def _frame_at(self, shard: int, offset: int) -> _Frame:  # noqa: DOC503 - _payload raises
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
            raw = _payload(file, instruction.filename, index, offset)
        return _Frame(shard, offset, raw)


# =============================================================================
# Choosing the source by the prepared format
# =============================================================================


def from_tfds(
    name: str,
    split: str,
    *,
    data_dir: str | None = None,
    in_memory: bool = True,
    as_supervised: bool = False,
    include_keys: AbstractSet[str] | None = None,
    exclude_keys: AbstractSet[str] | None = None,
) -> DataSourceModule:
    """Create the TFDS source that reads the copy prepared in ``data_dir``, by its format.

    - A copy prepared as ArrayRecord has random access. With ``in_memory=True`` (the default)
      ``TFDSEagerSource`` decodes it into host columns at init; with ``in_memory=False`` an
      ``ArrayRecordSourceModule`` reads and decodes each batch's records when the pipeline reads
      them, for a split larger than RAM, at the cost of decoding every record every epoch.
    - A copy prepared as TFRecord (TFDS's default format) is streamed by
      ``TFDSStreamingSource``, which holds no split in memory, whatever ``in_memory`` says.

    Both read without TensorFlow.

    Neither prepares a dataset; a split that is not prepared is refused, naming the call that
    prepares it as ArrayRecord. The order records are served in belongs to the pipeline
    (``Pipeline(shuffle=...)``).

    Args:
        name: TFDS dataset name (e.g., "mnist", "cifar10", "imagenet2012")
        split: Dataset split (e.g., "train", "test", "train[:1000]")
        data_dir: Optional directory where the dataset is prepared
        in_memory: Whether an ArrayRecord copy is decoded into memory at init (True) or read
            per batch (False)
        as_supervised: If True, keeps only the supervised features, under their own names
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude

    Returns:
        TFDSEagerSource or ArrayRecordSourceModule for an ArrayRecord copy, TFDSStreamingSource
        for a TFRecord copy.

    Example:
        ```python
        from datarax.sources import from_tfds

        source = from_tfds("mnist", "train")  # prepared as ArrayRecord: the eager source
        ```
    """
    if _tfrecord_only(name, data_dir):
        return TFDSStreamingSource(
            TFDSStreamingConfig(
                name=name,
                split=split,
                data_dir=data_dir,
                as_supervised=as_supervised,
                include_keys=include_keys,
                exclude_keys=exclude_keys,
            )
        )
    if not in_memory:
        return _per_batch_split(
            name,
            split,
            data_dir=data_dir,
            as_supervised=as_supervised,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    return TFDSEagerSource(
        TFDSEagerConfig(
            name=name,
            split=split,
            data_dir=data_dir,
            as_supervised=as_supervised,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    )
