"""TensorFlow Datasets (TFDS) sources for Datarax, both read without TensorFlow.

**TFDSEagerSource** reads a split that TFDS has prepared as ArrayRecord, whole, into host NumPy
columns and per-record provenance. TFDS's random-access reader (``builder.as_data_source``) reads
every record in one batched call and decodes it with NumPy and Pillow; the records then become
columns once, through the path every eager source shares.

- Holds no iteration state; the pipeline owns the order and position
- Strings a record carries (an ``id``, a caption) are kept as its provenance, never served
- Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**TFDSStreamingSource** streams a split TFDS has prepared as TFRecord (TFDS's default format), for
datasets too large for host memory: Grain's TFRecord reader frames the records and TFDS's NumPy
decoder decodes them. Records are named by the id TFDS reports for them, ``tfds_id``: the shard
file and the record's offset in it (``STREAM_IDS``). Given the pipeline's key, a pass is ordered as
TFDS's training read orders it: the shard files in a keyed order, read 16 at a time in blocks of
16, through tf.data's buffer shuffle.

TensorFlow is never imported: TensorFlow in a JAX process breaks JAX's NCCL collectives. Neither
source prepares a dataset, because preparing imports TensorFlow: a split that is not prepared, or
is prepared only in the format the other source reads, is refused, naming the call that prepares
it or the source that reads it. Reading needs the ``data`` extra; preparing needs the ``tfds``
extra, in a process of its own.
"""

from __future__ import annotations

import itertools
import logging
import os
import struct
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol

import grain
import jax
import numpy as np
from jax.typing import ArrayLike

from datarax.core.data_source import record_words, RecordIdentity, refuse_padding
from datarax.sources._config_base import SourceConfigBase
from datarax.sources._source_base import (
    DatasetSourceMixin,
    pass_seed,
    Provenance,
    StreamChunk,
    StreamingSourceBase,
)
from datarax.sources.eager_source import EagerSource, HostValue, parts_of_records
from datarax.sources.source_ops import filter_keys, validate_source_settings


logger = logging.getLogger(__name__)


# =============================================================================
# Opening a prepared copy
# =============================================================================


def _configure_protobuf_runtime() -> None:
    """Configure protobuf runtime before importing TensorFlow ecosystem modules."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


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
    _configure_protobuf_runtime()
    import tensorflow_datasets as tfds

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


def tfrecord_only(name: str, data_dir: str | None) -> bool:
    """Whether ``name`` is prepared in ``data_dir`` as TFRecord and not as ArrayRecord.

    Such a copy has no random access, so it is streamed; any other (ArrayRecord, or nothing
    prepared) goes to the eager source, which reads ArrayRecord or names the call preparing it.

    Args:
        name: The TFDS dataset name.
        data_dir: The data directory; TFDS's default when ``None``.

    Returns:
        Whether the prepared copy is TFRecord only.
    """
    _configure_protobuf_runtime()
    import tensorflow_datasets as tfds

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

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_source_settings(self, "TFDSEagerConfig")


@dataclass(frozen=True)
class TFDSStreamingConfig(SourceConfigBase):
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

    shuffle_buffer_size: int = 1000
    as_supervised: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ValueError: If the name, split or key filters are invalid, or the buffer holds no
                record.
        """
        validate_source_settings(self, "TFDSStreamingConfig")
        if self.shuffle_buffer_size < 1:
            raise ValueError(
                f"shuffle_buffer_size must be at least 1; got {self.shuffle_buffer_size}"
            )


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
    builder = _prepared_builder(
        name, data_dir, "array_record", lambda found: _not_prepared(name, data_dir, found)
    )
    # tfds wraps as_data_source in a logging decorator whose type hides the parameters.
    records = builder.as_data_source(
        split=split,  # pyright: ignore[reportCallIssue]
        file_format="array_record",
    )
    return builder.info, records


def _kept_features(
    record: dict[str, Any],
    keys: Sequence[str] | None,
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
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


def _frames(shard: _Shard) -> Iterator[_Frame]:
    """The records ``shard`` reads, in file order, with Grain's TFRecord reader.

    Args:
        shard: The shard file and the offsets the split reads of it.

    Yields:
        Each record and its id.

    Raises:
        ValueError: If a record's offset does not fit the id's low word.
    """
    for offset, raw in enumerate(grain.experimental.TFRecordIterDataset(shard.path)):
        if offset >= shard.skip + shard.take:
            return
        if offset < shard.skip:
            continue
        if offset > _LARGEST_OFFSET:
            raise ValueError(
                f"shard file {shard.path}: the record at offset {offset} does not fit an id, "
                f"whose low word holds a record's offset in its shard, at most {_LARGEST_OFFSET}"
            )
        yield _Frame(shard.shard, offset, raw)


def _interleaved(shards: Sequence[_Shard]) -> Iterator[_Frame]:
    """The records of ``shards`` read as TFDS's training read interleaves its files.

    ``_CYCLE_LENGTH`` files are open at once and ``_BLOCK_LENGTH`` records are read from each in
    turn; a file that ends is replaced by the next one in its place in the cycle (tf.data's
    ``interleave(cycle_length, block_length)``, which ``tfds.core.reader`` applies).
    """
    pending = iter(shards)
    cycle = [_frames(shard) for shard in itertools.islice(pending, _CYCLE_LENGTH)]
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
                cycle[slot] = _frames(following)
            slot += 1


def _buffer_shuffled(
    frames: Iterator[_Frame], size: int, generator: np.random.Generator
) -> Iterator[_Frame]:
    """``frames`` through tf.data's buffer shuffle of ``size`` records.

    The buffer fills with the first ``size`` records; then each record served is a uniform pick
    from the buffer, whose place the next record takes; at the end the buffer is served in
    uniform picks. The buffer holds serialized records, so decoding follows the order. Its state
    is the records' ids and the generator's state.
    """
    buffer = list(itertools.islice(frames, size))
    for frame in frames:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = frame
    while buffer:
        pick = int(generator.integers(len(buffer)))
        yield buffer[pick]
        buffer[pick] = buffer[-1]
        buffer.pop()


def _record_offsets(path: str) -> np.ndarray:
    """The byte offset of every record in a TFRecord file, by reading its frame headers.

    Neither TFDS, Grain nor ArrayRecord indexes a TFRecord file (searched: ``tfds.core.reader``,
    ``grain.experimental.TFRecordIterDataset``, ``array_record``), so the headers are read here:
    12 bytes per record, the payload skipped.
    """
    offsets = []
    with Path(path).open("rb") as file:
        position = 0
        while header := file.read(_TFRECORD_HEADER.size):
            length, _ = _TFRECORD_HEADER.unpack(header)
            offsets.append(position)
            position += _TFRECORD_HEADER.size + length + 4
            file.seek(position)
    return np.asarray(offsets, dtype=np.int64)


def _features(name: str, data_dir: str | None) -> Any:
    """The dataset's TFDS features, which decode its serialized records with NumPy."""
    import tensorflow_datasets as tfds  # noqa: PLC0415 - opened where the records are decoded

    return tfds.builder(name, data_dir=data_dir).info.features


@dataclass(frozen=True, slots=True)
class StreamRead:
    """What one pass of a TFDS stream reads, as plain values that pickle to a worker process.

    Attributes:
        name: The TFDS dataset name.
        data_dir: The data directory holding it prepared as TFRecord.
        shards: The shard files and offsets the split reads, in file order.
        seed: The pass's order as an integer key (``fold_in(key, pass)``'s data), or ``None`` for
            file order.
        buffer_size: Records the shuffle buffer holds.
        batch_size: Records decoded together, one element of the dataset.
        keys: The supervised features kept, or ``None`` for every feature.
        include_keys: Features kept, when given.
        exclude_keys: Features dropped, when given.
    """

    name: str
    data_dir: str | None
    shards: tuple[_Shard, ...]
    seed: int | None
    buffer_size: int
    batch_size: int
    keys: tuple[str, ...] | None
    include_keys: frozenset[str] | None
    exclude_keys: frozenset[str] | None


def _ordered_frames(read: StreamRead) -> Iterator[_Frame]:
    """The pass's serialized records in its order: file order, or TFDS's training read's."""
    if read.seed is None:
        return itertools.chain.from_iterable(map(_frames, read.shards))
    generator = np.random.Generator(np.random.Philox(key=read.seed))
    order = generator.permutation(len(read.shards))
    return _buffer_shuffled(
        _interleaved([read.shards[i] for i in order]), read.buffer_size, generator
    )


def _decoded_batch(
    features: Any, frames: Sequence[_Frame], read: StreamRead
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray]:
    """``frames`` decoded with TFDS's NumPy decoder: host columns, provenance and ids.

    Args:
        features: The dataset's TFDS features.
        frames: The serialized records.
        read: The pass's read, which names the features kept.

    Returns:
        The array part as host columns, one provenance mapping per record and the records' ids.
    """
    records = [
        _kept_features(
            features.deserialize_example_np(frame.raw),
            read.keys,
            set(read.include_keys) if read.include_keys is not None else None,
            set(read.exclude_keys) if read.exclude_keys is not None else None,
        )
        for frame in frames
    ]
    columns, provenance = parts_of_records(records)
    ids = np.asarray([(frame.shard << 32) | frame.offset for frame in frames], dtype=np.uint64)
    return columns, provenance or tuple({} for _ in frames), ids


class TFDSStreamDataset(grain.IterDataset):
    """One pass of a TFDS stream as a Grain dataset of decoded batches.

    Its iterator reads the pass's serialized records in the pass's order, cuts them into batches of
    ``batch_size`` and decodes each batch last, so the order runs on raw bytes and only the decode
    is per batch. It implements Grain's slicing hook: with ``set_slice(slice(i, None, k))`` the
    iterator still orders every raw record but decodes and yields only batches ``j`` with
    ``j % k == i``, so ``k`` such slices interleaved round robin give the unsliced batches, the
    stream's order not depending on ``k``. That is how Grain's process prefetch splits a dataset
    across workers. The dataset pickles (its read is plain values), and each iterator opens its
    own files and decoder, so iterators in threads or processes share no lazy state.
    """

    def __init__(self, read: StreamRead) -> None:
        """Hold the pass's read; nothing is opened until iteration.

        Args:
            read: What the pass reads.
        """
        super().__init__()
        self._read = read
        self._slice = slice(0, None, 1)

    def set_slice(self, sl: slice, sequential_slice: bool = False) -> None:
        """Keep batches ``j`` with ``j % sl.step == sl.start``: Grain's slicing hook.

        Args:
            sl: The slice of batches this dataset serves.
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
        """A fresh iterator over the pass, opening its own files and decoder."""
        return _TFDSStreamIterator(self._read, self._slice)


class _TFDSStreamIterator(grain.DatasetIterator):
    """Batches of one pass of a TFDS stream, those of its slice decoded."""

    def __init__(self, read: StreamRead, sl: slice) -> None:
        super().__init__()
        self._read = read
        self._frames = _ordered_frames(read)
        self._features = _features(read.name, read.data_dir)
        self._start, self._step = sl.start or 0, sl.step or 1
        self._batch = 0

    def __next__(self) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], np.ndarray]:
        while True:
            frames = list(itertools.islice(self._frames, self._read.batch_size))
            if not frames:
                raise StopIteration
            batch, self._batch = self._batch, self._batch + 1
            if batch % self._step == self._start:
                return _decoded_batch(self._features, frames, self._read)

    def get_state(self) -> dict[str, Any]:
        """The batches of the pass this iterator has passed."""
        return {"batches": self._batch}

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

    Each pass reads the split's shard files with Grain's TFRecord reader and decodes each record
    with TFDS's NumPy decoder: numeric features become host NumPy columns, text and other objects
    the record's provenance, beside the batch. A record is named by the id TFDS reports for it,
    ``tfds_id``: its shard file and its offset in the file, as the two words ``(shard, offset)``
    (``STREAM_IDS``), where the shard is the file's place among the dataset's files in name order
    (:attr:`shard_files`). A slice names its records as the full split does.

    The order is the pipeline's to choose. Without its key a pass reads the files in order. With
    it, pass ``p`` is ordered as TFDS's training read (``shuffle_files=True`` and
    ``shuffle(shuffle_buffer_size)``) orders an epoch: the files in a keyed order, interleaved 16 at
    a time in blocks of 16, through tf.data's buffer shuffle on the serialized records, every draw
    from a NumPy Philox generator keyed by ``fold_in(key, p)``. The buffer holds
    ``shuffle_buffer_size`` serialized records in host memory.

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

    def __init__(  # noqa: DOC502 - _prepared_builder and _supervised_keys raise
        self, config: TFDSStreamingConfig, *, name: str | None = None
    ) -> None:
        """Open the prepared TFRecord split, reading no record.

        Args:
            config: Configuration for the source
            name: Optional name (defaults to TFDSStreamingSource(dataset:split))

        Raises:
            FileNotFoundError: If the dataset is not prepared as TFRecord in the data directory.
            ValueError: If ``as_supervised`` is asked of a dataset without supervised keys.
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
        self._keys = HostValue(
            tuple(_supervised_keys(info, dataset)) if config.as_supervised else None
        )
        files = {
            Path(instruction.filename).name: instruction
            for split_name in info.splits
            for instruction in info.splits[split_name].file_instructions
        }
        self._files = HostValue(tuple(sorted(files.items())))
        shard_of = {basename: shard for shard, (basename, _) in enumerate(self._files.value)}
        self._shards = HostValue(
            tuple(
                _Shard(
                    instruction.filename,
                    shard_of[Path(instruction.filename).name],
                    instruction.skip,
                    instruction.take,
                )
                for instruction in info.splits[split].file_instructions
            )
        )
        self._offsets = HostValue({})
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
        self, pass_index: int, key: jax.Array | None, batch_size: int
    ) -> TFDSStreamDataset:
        """Pass ``pass_index`` as a Grain dataset of decoded batches, a :class:`TFDSStreamDataset`.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key when it shuffles, ``None`` for file order.
            batch_size: Records per decoded batch.

        Returns:
            The pass's dataset.
        """
        return TFDSStreamDataset(self._stream_read(pass_index, key, batch_size))

    def _stream_read(self, pass_index: int, key: jax.Array | None, batch_size: int) -> StreamRead:
        """What pass ``pass_index`` reads, as plain values."""
        config = self.config
        include, exclude = config.include_keys, config.exclude_keys
        return StreamRead(
            name=self._dataset,
            data_dir=config.data_dir,
            shards=self._shards.value,
            seed=None if key is None else pass_seed(key, pass_index),
            buffer_size=config.shuffle_buffer_size,
            batch_size=batch_size,
            keys=self._keys.value,
            include_keys=None if include is None else frozenset(include),
            exclude_keys=None if exclude is None else frozenset(exclude),
        )

    def _open_pass(
        self, pass_index: int, key: jax.Array | None, size_hint: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index``: in file order, or as TFDS's training read under ``key``.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key when it shuffles, ``None`` for file order.
            size_hint: Records decoded together, the first pull's size.

        Yields:
            The pass's records as decoded chunks.
        """
        for columns, provenance, ids in self.pass_dataset(pass_index, key, size_hint):
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
            IndexError: If an index is the padding index, or names no shard or no record in it.
        """
        words = record_words(indices)
        refuse_padding(words)
        frames = [self._frame_at(int(shard), int(offset)) for shard, offset in words]
        read = self._stream_read(0, None, len(frames))
        features = self._dataset_info.value.features
        return tuple(
            MappingProxyType(record) for record in _decoded_batch(features, frames, read)[1]
        )

    def _frame_at(self, shard: int, offset: int) -> _Frame:
        """The serialized record at ``offset`` of shard ``shard``.

        Args:
            shard: The shard number.
            offset: The record's offset in the shard file.

        Returns:
            The record and its id.

        Raises:
            IndexError: If the dataset has no such shard, or the shard no such record.
        """
        files = self._files.value
        if shard >= len(files):
            raise IndexError(f"shard {shard} names no file: the dataset has {len(files)} shards")
        basename, instruction = files[shard]
        offsets = self._offsets.value.get(shard)
        if offsets is None:
            offsets = self._offsets.value.setdefault(shard, _record_offsets(instruction.filename))
        if offset >= len(offsets):
            raise IndexError(
                f"offset {offset} names no record of shard {shard} ({basename}), which holds "
                f"{len(offsets)}"
            )
        reader = iter(grain.experimental.TFRecordIterDataset(instruction.filename))
        reader.set_state({"reader_offset": int(offsets[offset])})
        return _Frame(shard, offset, next(reader))
