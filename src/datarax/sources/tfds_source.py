"""TensorFlow Datasets (TFDS) sources for Datarax.

**TFDSEagerSource** reads a split that TFDS has prepared as ArrayRecord, whole, into host NumPy
columns and per-record provenance. TFDS's random-access reader (``builder.as_data_source``) reads
every record in one batched call and decodes it with NumPy and Pillow; the records then become
columns once, through the path every eager source shares. TensorFlow is never imported, so a
training process that reads TFDS data this way holds none of it (TensorFlow in a JAX process
breaks JAX's NCCL collectives). The source never prepares a dataset, because preparing imports
TensorFlow: a split that is not prepared, or is prepared only in another format such as TFRecord,
is refused, naming the call that prepares it as ArrayRecord. Reading needs the ``data`` extra;
preparing needs the ``tfds`` extra, in a process of its own.

- Holds no iteration state; the pipeline owns the order and position
- Strings a record carries (an ``id``, a caption) are kept as its provenance, never served
- Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**TFDSStreamingSource** streams a copy prepared as TFRecord through ``tf.data``, for datasets too
large for host memory; it imports TensorFlow into the process.

- DLPack zero-copy conversion for each record
- Fixed prefetch buffer (no AUTOTUNE thread storms)
- Trade-offs: external iterator state, can't checkpoint mid-epoch
- Ideal for: ImageNet, large-scale datasets, memory-constrained environments
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import jax
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.sources._config_base import SourceConfigBase
from datarax.sources._conversion import tf_to_jax
from datarax.sources._source_base import DatasetSourceMixin, StreamingSourceBase
from datarax.sources.eager_source import EagerSource, HostValue, parts_of_records
from datarax.sources.source_ops import (
    converted_filtered_record,
    filter_keys,
    validate_eager_source_settings,
    validate_positive_optional_int,
    validate_streaming_source_settings,
)


logger = logging.getLogger(__name__)


# =============================================================================
# TFDS Builder Helpers
# =============================================================================


def _is_read_only_tfds_source(builder: Any) -> bool:
    """Check if a TFDS builder is a ReadOnlyBuilder (from try_gcs or GCS cache).

    ReadOnlyBuilder reads pre-built TFRecords from GCS and needs no
    download_and_prepare() call. We use a string-based check to avoid
    importing the ReadOnlyBuilder class at module level (heavy import).
    """
    return type(builder).__name__ == "ReadOnlyBuilder"


def _configure_protobuf_runtime() -> None:
    """Configure protobuf runtime before importing TensorFlow ecosystem modules."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _prepare_tfds_builder(
    name: str,
    data_dir: str | None,
    try_gcs: bool,
    download_and_prepare_kwargs: dict[str, Any] | None,
    beam_num_workers: int | None = None,
    local_files_only: bool = False,
) -> Any:
    """Create and prepare a TFDS builder, handling ReadOnlyBuilder from GCS.

    Args:
        name: TFDS dataset name
        data_dir: Optional local data directory
        try_gcs: Whether to try loading from GCS
        download_and_prepare_kwargs: Optional kwargs for download_and_prepare
        beam_num_workers: Optional number of Beam DirectRunner workers.
            When set, enables multi-processing mode for parallel dataset
            generation. Useful for large datasets that use Apache Beam
            (e.g., NSynth). None means single-threaded (Beam default).
        local_files_only: If True, skip ``download_and_prepare`` entirely and
            assume the dataset is already prepared in ``data_dir``. The user
            is responsible for ensuring the cache is populated; otherwise
            ``tfds.load`` will surface its own error downstream.

    Returns:
        A prepared TFDS builder with dataset info available.
    """
    _configure_protobuf_runtime()
    import tensorflow_datasets as tfds

    builder = tfds.builder(name, data_dir=data_dir, try_gcs=try_gcs)

    # In local_files_only mode the user guarantees the cache is populated;
    # ReadOnlyBuilder (from try_gcs when dataset is on GCS) also needs no
    # preparation.
    if not local_files_only and not _is_read_only_tfds_source(builder):
        download_kwargs = download_and_prepare_kwargs or {}

        if beam_num_workers is not None:
            import apache_beam as beam

            beam_options = beam.options.pipeline_options.PipelineOptions(
                direct_num_workers=beam_num_workers,
                direct_running_mode="multi_processing",
            )
            download_config = tfds.download.DownloadConfig(
                beam_options=beam_options,
            )
            builder.download_and_prepare(download_config=download_config, **download_kwargs)
        else:
            builder.download_and_prepare(**download_kwargs)

    return builder


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
        validate_eager_source_settings(self, "TFDSEagerConfig")


@dataclass(frozen=True)
class TFDSStreamingConfig(SourceConfigBase):
    """Configuration for TFDSStreamingSource (streams data from TF dataset).

    Use this for datasets too large to fit in memory. The streaming source
    uses fixed prefetch buffers to avoid AUTOTUNE thread storms.

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: TF shuffle buffer size (default: 1000)
        as_supervised: If True, returns 'image'/'label' keys
        download_and_prepare_kwargs: Optional keyword arguments for download_and_prepare
        include_keys: Optional set of keys to include in output
        exclude_keys: Optional set of keys to exclude from output
        prefetch_buffer: Fixed prefetch buffer size (default: 2, NOT AUTOTUNE)

    Note:
        The prefetch_buffer uses a fixed size instead of TF AUTOTUNE to prevent
        background thread storms that cause delays during epoch transitions.
    """

    try_gcs: bool = False
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    beam_num_workers: int | None = None
    prefetch_buffer: int = 2  # Fixed, NOT AUTOTUNE
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_streaming_source_settings(self, "TFDSStreamingConfig")

        if self.try_gcs and self.data_dir is not None:
            raise ValueError(
                f"Cannot specify both try_gcs=True and data_dir='{self.data_dir}' "
                "in TFDSStreamingConfig. "
                "try_gcs overrides data_dir to the public GCS bucket (gs://tfds-data/datasets/)."
            )

        validate_positive_optional_int(self.beam_num_workers, "beam_num_workers")


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
    where = data_dir if data_dir is not None else "TFDS's default data directory"
    return FileNotFoundError(
        f"TFDS dataset {name!r} is not prepared as ArrayRecord in {where}: {found}. "
        "TFDSEagerSource reads a copy prepared as ArrayRecord and never prepares one, since "
        "preparing imports TensorFlow. Prepare it once, in a process of its own with the tfds "
        f"extra installed: tfds.builder({name!r}, data_dir={data_dir!r}, "
        "file_format='array_record').download_and_prepare(). A data directory holds one format "
        "per dataset version, so prepare it where no copy of that version in another format is."
    )


def open_prepared_split(  # noqa: DOC503 - _not_prepared builds the FileNotFoundError
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
    _configure_protobuf_runtime()
    import tensorflow_datasets as tfds

    try:
        builder = tfds.builder(name, data_dir=data_dir)
    except tfds.core.DatasetNotFoundError as error:
        raise _not_prepared(name, data_dir, "it is not prepared there") from error
    if not builder.is_prepared():
        raise _not_prepared(name, data_dir, f"{builder.data_dir} is not prepared")
    formats = builder.info.available_file_formats()
    if tfds.core.FileFormat.ARRAY_RECORD not in formats:
        found = ", ".join(sorted(fmt.value for fmt in formats))
        raise _not_prepared(name, data_dir, f"{builder.data_dir} holds it prepared as {found}")
    # tfds wraps as_data_source in a logging decorator whose type hides the parameters.
    records = builder.as_data_source(
        split=split,  # pyright: ignore[reportCallIssue]
        file_format=tfds.core.FileFormat.ARRAY_RECORD,
    )
    return builder.info, records


def _kept_features(
    record: dict[str, Any], keys: Sequence[str] | None, config: TFDSEagerConfig
) -> dict[str, Any]:
    """The features of ``record`` the source keeps: the supervised ones if asked, then filtered."""
    if keys is not None:
        record = {key: record[key] for key in keys}
    return filter_keys(record, config.include_keys, config.exclude_keys)


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
            _kept_features(record, keys, config)
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
        declared = self.get_dataset_info().supervised_keys
        if declared is None:
            raise ValueError(f"{dataset} declares no supervised keys, which as_supervised keeps")
        return [str(key) for key in jax.tree.leaves(declared)]


# =============================================================================
# TFDSStreamingSource - Thin Wrapper for Large Datasets
# =============================================================================


class TFDSStreamingSource(StreamingSourceBase):
    """Streaming TFDS source for large datasets.

    Thin wrapper around TF dataset for data that can't fit in memory.
    Uses DLPack for efficient conversion and fixed prefetch buffer.

    Key Features:
        - DLPack zero-copy for TF→JAX conversion
        - Fixed prefetch buffer (no AUTOTUNE thread storms)
        - Supports all TFDS datasets
        - include_keys/exclude_keys filtering

    Trade-offs vs Eager:
        - Cannot checkpoint mid-epoch (external iterator state)
        - Some TF thread overhead (minimized with fixed prefetch)
        - Use with Artifex train_epoch_streaming() for best results

    Example:
        ```python
        # Create streaming source for large dataset
        config = TFDSStreamingConfig(name="imagenet2012", split="train")
        source = TFDSStreamingSource(config, rngs=nnx.Rngs(0))

        # Iterate with prefetching
        for batch in prefetch_to_device(source, size=2):
            train_step(batch)
        ```
    """

    @property
    def record_identity(self) -> RecordIdentity:
        """TFDS can report a record's id (its file shard and offset): the stream's own names."""
        return RecordIdentity.STREAM_IDS

    def __init__(
        self,
        config: TFDSStreamingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize TFDSStreamingSource.

        Args:
            config: Configuration for the source
            rngs: Optional RNG state
            name: Optional name (defaults to TFDSStreamingSource(dataset:split))
        """
        if name is None:
            name = f"TFDSStreamingSource({config.name}:{config.split})"
        super().__init__(config, rngs=rngs, name=name)

        self.dataset_name = config.name
        self.split_name = config.split
        self._is_random_order = config.shuffle
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # name/split validated non-None by config __post_init__
        name = config.name
        assert name is not None  # noqa: S101 (invariant, not control flow)

        # Load builder and info
        builder = _prepare_tfds_builder(
            name,
            config.data_dir,
            config.try_gcs,
            config.download_and_prepare_kwargs,
            beam_num_workers=config.beam_num_workers,
            local_files_only=config.local_files_only,
        )
        self._dataset_info = builder.info

        # Build TF dataset with optimizations
        self._tf_dataset = builder.as_dataset(
            split=config.split,
            as_supervised=config.as_supervised,
        )

        if config.shuffle:
            self._tf_dataset = self._tf_dataset.shuffle(
                buffer_size=config.shuffle_buffer_size,
                reshuffle_each_iteration=True,
            )

        # CRITICAL: Fixed prefetch, NOT AUTOTUNE
        # This prevents thread storms during epoch transitions
        self._tf_dataset = self._tf_dataset.prefetch(config.prefetch_buffer)

        # Try to get length from split info
        split = config.split
        try:
            assert split is not None  # noqa: S101 (invariant, not control flow)
            split_base = split.split("[")[0]  # Handle splits like "train[:1000]"
            self.length: int | None = self._dataset_info.splits[split_base].num_examples
        except (AttributeError, KeyError):
            self.length = None

        self._iterator: Iterator | None = None
        self.epoch = nnx.Variable(0)

    def __len__(self) -> int:
        """Return the total number of data elements if known.

        Returns:
            Total number of elements or raises NotImplementedError if unknown

        Raises:
            NotImplementedError: If the length of the dataset split is unknown.
        """
        if self.length is None:
            raise NotImplementedError("Length unknown for this dataset split")
        return self.length

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Start iteration over the dataset."""
        self.epoch.set_value(self.epoch.get_value() + 1)
        self._iterator = iter(self._tf_dataset)
        return self

    def __next__(self) -> dict[str, jax.Array]:  # noqa: DOC502
        """Get next element from the dataset.

        Returns:
            Dictionary of JAX arrays

        Raises:
            StopIteration: When dataset is exhausted
        """
        iterator = self._iterator
        if iterator is None:
            iterator = iter(self._tf_dataset)
            self._iterator = iterator

        return self._convert_record(next(iterator))

    def _convert_record(self, tf_element: Any) -> dict[str, Any]:
        """Filter and convert one raw TFDS element into the record this source emits."""
        # Handle as_supervised tuple format
        if self.as_supervised and isinstance(tf_element, tuple):
            tf_element = {"image": tf_element[0], "label": tf_element[1]}

        return converted_filtered_record(
            tf_element,
            self.include_keys,
            self.exclude_keys,
            tf_to_jax,
        )

    def element_spec(self) -> Any:
        """Return the spec of the records this source emits, derived from the first one.

        TFDS streams yield single-element dicts (or tuples in ``as_supervised``
        mode). The first element of a fresh iterator on the cached
        ``self._tf_dataset`` (built in ``__init__``, so no download is
        re-triggered) is filtered and converted exactly as iteration does it,
        so ``include_keys``/``exclude_keys`` apply and the spec describes the JAX
        arrays batches carry. Top-level dict values are described as single
        leaves so vector features become 1-D ``ShapeDtypeStruct`` instead of
        per-scalar leaves.
        """
        from datarax.core.spec import array_to_spec  # noqa: PLC0415

        record = self._convert_record(next(iter(self._tf_dataset)))
        return {key: array_to_spec(value) for key, value in record.items()}
