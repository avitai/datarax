"""TFDSEagerSource and ``from_tfds``: a split TFDS has prepared, read without TensorFlow.

**TFDSEagerSource** reads a split that TFDS has prepared as ArrayRecord, whole, into host NumPy
columns and per-record provenance. TFDS's random-access reader (``builder.as_data_source``) reads
every record in one batched call and decodes it with NumPy and Pillow; the records then become
columns once, through the path every eager source shares.

- Holds no iteration state; the pipeline owns the order and position
- Strings a record carries (an ``id``, a caption) are kept as its provenance, never served
- Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

A split prepared as TFRecord, TFDS's default format, is streamed by
:class:`~datarax.sources.tfds_streaming_source.TFDSStreamingSource`, for datasets too large for host
memory. ``from_tfds`` picks between the two sources by the format the copy is prepared in.

TensorFlow is never imported: TensorFlow in a JAX process breaks JAX's NCCL collectives. Neither
source prepares a dataset, because preparing imports TensorFlow: a split that is not prepared, or
is prepared only in the format the other source reads, is refused, naming the call that prepares
it or the source that reads it. Reading needs the ``data`` extra; preparing needs the ``tfds``
extra, in a process of its own.

This module imports ``tensorflow_datasets`` at its top, so importing it without the ``data`` extra
raises ``ImportError``. ``datarax.sources`` exports the TFDS sources' names lazily, so ``import
datarax.sources`` imports no TFDS: the import happens on the line that imports a TFDS source or
``from_tfds``. TFDS's first import enters ``etils.epy.lazy_imports()``, which replaces
``builtins.__import__`` for the whole process while it runs. A garbage collection in that window
runs finalizers whose imports fail, so the window must not open inside a constructor in the middle
of a run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence, Set as AbstractSet
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import tensorflow_datasets as tfds

from datarax.core.data_source import DataSourceModule
from datarax.core.host_resources import HostResources
from datarax.core.host_values import HostValue
from datarax.core.spec import JAX_ARRAY_KINDS
from datarax.sources._config_base import SourceConfigBase
from datarax.sources._source_base import DatasetSourceMixin
from datarax.sources._tfds_prepared import (
    kept_features,
    prepare_call,
    prepared_builder,
    supervised_keys,
    where,
)
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule
from datarax.sources.eager_source import EagerSource, parts_of_records
from datarax.sources.tfds_streaming_source import TFDSStreamingConfig, TFDSStreamingSource


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
        f"TFDS dataset {name!r} is not prepared as ArrayRecord in {where(data_dir)}: {found}. "
        "TFDSEagerSource reads a copy prepared as ArrayRecord and never prepares one, since "
        "preparing imports TensorFlow; a copy prepared as TFRecord is streamed by "
        "TFDSStreamingSource. Prepare it once, in a process of its own with the tfds extra "
        f"installed: {prepare_call(name, data_dir, 'array_record')}. A data directory holds one "
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
    return prepared_builder(
        name, data_dir, "array_record", lambda found: _not_prepared(name, data_dir, found)
    )


def open_prepared_split(  # noqa: DOC502 - prepared_builder raises the FileNotFoundError
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


@dataclass(frozen=True, slots=True)
class _ExampleDecoder:
    """Decodes a batch of a prepared split's serialized examples as TFDS's reader decodes one.

    Each record goes through the dataset's ``features.deserialize_example_np``, as
    ``builder.as_data_source`` does, then keeps the features the source keeps
    (:func:`kept_features`), so a per-batch read holds what the eager source holds. It pickles
    with the features, for a worker process.
    """

    features: Any
    keys: tuple[str, ...] | None
    include_keys: AbstractSet[str] | None
    exclude_keys: AbstractSet[str] | None

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """One kept mapping of decoded features per record."""
        return [
            kept_features(
                self.features.deserialize_example_np(record),
                self.keys,
                self.include_keys,
                self.exclude_keys,
            )
            for record in records
        ]


def _per_batch_split(  # noqa: DOC502 - _array_record_builder and supervised_keys raise
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
    keys = tuple(supervised_keys(info, name)) if as_supervised else None
    return ArrayRecordSourceModule(
        ArrayRecordSourceConfig(),
        info.splits[split].file_instructions,
        decode=_ExampleDecoder(info.features, keys, include_keys, exclude_keys),
    )


def _decoded_bytes(features: Any, path: str = "") -> int:
    """The bytes one record's numeric features decode to, at their stored dtypes.

    Strings and other non-numeric features are a record's provenance, not columns, and their
    size is not known before reading; they are not counted.

    Args:
        features: A TFDS feature, or a mapping of them (a record's kept features, or a
            ``FeaturesDict``, which maps its names to features without being a ``Mapping``).
        path: The feature's path, for the refusal.

    Returns:
        The bytes.

    Raises:
        ValueError: If a numeric feature has no fixed shape, naming it and its shape.
    """
    if callable(getattr(features, "items", None)):
        return sum(
            _decoded_bytes(feature, f"{path}/{key}" if path else str(key))
            for key, feature in features.items()
        )
    dtype = np.dtype(features.np_dtype)
    if dtype.kind not in JAX_ARRAY_KINDS:
        return 0
    shape = tuple(features.shape)
    if any(size is None for size in shape):
        raise ValueError(
            f"feature {path!r} has shape {shape}, so its decoded size is not known before "
            "reading: an in-memory build under a RAM budget holds features of a fixed shape, as "
            "a batch's shapes are static"
        )
    return math.prod(shape) * dtype.itemsize


def _refuse_over_budget(  # noqa: PLR0913 - what the decoded size is computed from
    info: Any,
    dataset: str,
    split: str,
    keys: Sequence[str] | None,
    include_keys: AbstractSet[str] | None,
    exclude_keys: AbstractSet[str] | None,
    resources: HostResources,
) -> None:
    """Refuse an in-memory build of ``split`` whose decoded size is above the caller's budget.

    The size is exact from TFDS's fixed-shape features: each kept numeric feature's values at its
    stored dtype, times the split's records, computed before any record is read.

    Args:
        info: The dataset's ``DatasetInfo``.
        dataset: The dataset's name.
        split: The split, slices included.
        keys: The supervised features ``as_supervised`` keeps, or ``None``.
        include_keys: Features to include, or ``None``.
        exclude_keys: Features to exclude, or ``None``.
        resources: The caller's budget.

    Raises:
        ValueError: If the decoded size is above ``resources.ram_budget_bytes``, naming the size,
            the budget and the per-batch read; or a kept numeric feature has no fixed shape.
    """
    kept = kept_features(dict(info.features), keys, include_keys, exclude_keys)
    size = info.splits[split].num_examples * _decoded_bytes(kept)
    if size > resources.ram_budget_bytes:
        raise ValueError(
            f"{dataset} {split} decodes to {size} bytes in memory (its numeric features; strings "
            f"kept as provenance come on top), above ram_budget_bytes="
            f"{resources.ram_budget_bytes}: read it per batch with from_tfds(..., "
            "in_memory=False), or raise the budget"
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
        host_resources: HostResources | None = None,
        name: str | None = None,
    ) -> None:
        """Read the prepared split into host columns and provenance.

        Args:
            config: Configuration for the source
            host_resources: The caller's RAM budget, or ``None``; with one, a split whose
                decoded size is above it is refused before any record is read.
            name: Optional name (defaults to TFDSEagerSource(dataset:split))

        Raises:
            FileNotFoundError: If the dataset is not prepared as ArrayRecord in the data directory.
            ValueError: If ``as_supervised`` is asked of a dataset without supervised keys, the
                split and key filters leave nothing, a feature's shape varies between records, or
                the decoded split is above ``host_resources``'s budget.
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
        keys = self.supervised_keys(dataset) if config.as_supervised else None
        if host_resources is not None:
            _refuse_over_budget(
                info,
                dataset,
                split,
                keys,
                config.include_keys,
                config.exclude_keys,
                host_resources,
            )
        rows = [
            kept_features(record, keys, config.include_keys, config.exclude_keys)
            for record in records.__getitems__(list(range(len(records))))
        ]
        if not rows or not rows[0]:
            raise ValueError(
                f"{dataset} {split} produced no records after loading and filtering; check the "
                "split and the include/exclude key filters"
            )
        self._store(*parts_of_records(rows))

    def supervised_keys(self, dataset: str) -> list[str]:
        """The features ``as_supervised`` keeps: those the dataset declares supervised."""
        return supervised_keys(self.get_dataset_info(), dataset)


def from_tfds(
    name: str,
    split: str,
    *,
    data_dir: str | None = None,
    in_memory: bool = True,
    host_resources: HostResources | None = None,
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

    With ``host_resources`` the in-memory build refuses a split whose decoded size is above the
    RAM budget, before reading a record. A source read per batch takes its budget from the
    pipeline that reads it, ``Pipeline(host_resources=...)``, and refuses one here.

    Neither prepares a dataset; a split that is not prepared is refused, naming the call that
    prepares it as ArrayRecord. The order records are served in belongs to the pipeline
    (``Pipeline(shuffle=...)``).

    Args:
        name: TFDS dataset name (e.g., "mnist", "cifar10", "imagenet2012")
        split: Dataset split (e.g., "train", "test", "train[:1000]")
        data_dir: Optional directory where the dataset is prepared
        in_memory: Whether an ArrayRecord copy is decoded into memory at init (True) or read
            per batch (False)
        host_resources: The RAM budget the in-memory build is checked against, or ``None``
        as_supervised: If True, keeps only the supervised features, under their own names
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude

    Returns:
        TFDSEagerSource or ArrayRecordSourceModule for an ArrayRecord copy, TFDSStreamingSource
        for a TFRecord copy.

    Raises:
        ValueError: If ``host_resources`` is given for a source read per batch, or the in-memory
            build is over its budget.

    Example:
        ```python
        from datarax.sources import from_tfds

        source = from_tfds("mnist", "train")  # prepared as ArrayRecord: the eager source
        ```
    """
    stream = _tfrecord_only(name, data_dir)
    if host_resources is not None and (stream or not in_memory):
        raise ValueError(
            f"{name} {split} is read per batch, so its RAM budget is the pipeline's that reads "
            "it: pass Pipeline(host_resources=...) instead of from_tfds(host_resources=...)"
        )
    if stream:
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
        ),
        host_resources=host_resources,
    )
