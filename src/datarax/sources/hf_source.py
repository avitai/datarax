"""HuggingFace Datasets sources for Datarax.

**HFEagerSource** reads a map-style dataset whole, for datasets that fit in host memory:
- Loads ALL data at initialization: array columns as host NumPy, text and objects as provenance
- No HuggingFace work during training: reads are NumPy gathers
- Holds no iteration state; the pipeline owns the order and position
- Ideal for: MNIST, CIFAR-10, sentiment datasets, small custom datasets

**HFStreamingSource** streams a dataset with HuggingFace's streaming mode
(``load_dataset(..., streaming=True)``), for datasets too large to hold:
- Batched NumPy reads; array columns in their features' dtypes, text and objects as provenance
  beside each batch
- Records named by arrival; the pipeline's seed orders each pass through HF's buffer shuffle
- Ideal for: The Pile, C4, large-scale datasets, memory-constrained environments
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Collection, Iterator, Mapping, Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np


try:
    import datasets
except ImportError as error:
    msg = 'datarax.sources.hf_source needs datasets: uv pip install "datarax[data]"'
    raise ImportError(msg) from error

from datarax.core.data_source import DataSourceModule, NO_PROVENANCE, RecordIdentity
from datarax.core.spec import JAX_ARRAY_KINDS
from datarax.sources._config_base import SourceConfigBase, StreamingSourceConfigBase
from datarax.sources._source_base import (
    DatasetSourceMixin,
    key_integer,
    StreamChunk,
    StreamingSourceBase,
)
from datarax.sources.eager_source import EagerSource, HostValue, is_array_leaf, stack_records


# =============================================================================
# Configuration Classes
# =============================================================================


def _resolve_hf_download_options(
    download_kwargs: dict[str, Any] | None,
    *,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Prepare kwargs for ``datasets.load_dataset``.

    The ``datasets`` 4.x API no longer accepts ``local_files_only`` as a
    top-level kwarg (it is forwarded to ``BuilderConfig`` and rejected).
    The flag now lives on ``DownloadConfig`` and is passed via
    ``download_config=``.
    """
    resolved = dict(download_kwargs or {})
    # datasets.load_dataset no longer accepts this transformers-only flag.
    resolved.pop("trust_remote_code", None)
    resolved.setdefault("revision", "main")
    if local_files_only:
        existing = resolved.get("download_config")
        if isinstance(existing, datasets.DownloadConfig):
            existing.local_files_only = True
        else:
            resolved["download_config"] = datasets.DownloadConfig(local_files_only=True)
    return resolved


def _load_hf_dataset(config: HFEagerConfig | HFStreamingConfig, **extra: Any) -> Any:
    """Call ``datasets.load_dataset`` with a config's name, split, folders and download options.

    Args:
        config: The source's configuration.
        **extra: Further ``load_dataset`` arguments (``streaming``).

    Returns:
        What ``load_dataset`` returns.
    """
    # name/split validated non-None by config __post_init__
    assert config.name is not None  # noqa: S101 (invariant, not control flow)
    assert config.split is not None  # noqa: S101 (invariant, not control flow)
    return datasets.load_dataset(  # nosec B615
        config.name,
        split=config.split,
        data_dir=config.data_dir,
        cache_dir=config.cache_dir,
        **_resolve_hf_download_options(
            config.download_kwargs, local_files_only=config.local_files_only
        ),
        **extra,
    )


def _selected_hf_columns(
    column_names: list[str],
    include_keys: Collection[str] | None,
    exclude_keys: Collection[str] | None,
) -> list[str]:
    """Return the dataset columns the include and exclude filters keep, in dataset order."""
    included = [name for name in column_names if not include_keys or name in include_keys]
    return [name for name in included if not exclude_keys or name not in exclude_keys]


def _hf_value_on_host(value: Any) -> Any:
    """One HuggingFace value as the host holds it: a PIL image as its NumPy array."""
    return np.asarray(value) if hasattr(value, "mode") else value


def _hf_column_parts(key: str, values: Any) -> tuple[np.ndarray | None, list[Any] | None]:
    """Split one numpy-formatted HF column into a host NumPy column or provenance values.

    A numeric column is already one NumPy array. A column of images or numeric sequences is
    stacked once (the shapes must agree); a column of text or other objects is provenance.

    Args:
        key: The column's name.
        values: The column as the numpy format gives it.

    Returns:
        ``(column, None)`` for an array column, ``(None, values)`` for provenance.
    """
    if isinstance(values, np.ndarray) and is_array_leaf(values):
        return values, None
    host = [_hf_value_on_host(value) for value in values]
    if host and all(is_array_leaf(value) for value in host):
        return stack_records([{key: np.asarray(value)} for value in host])[key], None
    return None, list(values)


def _hf_parts(
    loaded: Mapping[str, Any], size: int
) -> tuple[dict[str, np.ndarray], tuple[dict[str, Any], ...]]:
    """Numpy-formatted HF columns as host array columns and one provenance mapping per record.

    Args:
        loaded: Column name to the column as HF's numpy format gives it.
        size: The records the columns hold.

    Returns:
        The array columns, and one mapping per record of its non-array columns (empty when
        every column is an array).
    """
    columns: dict[str, np.ndarray] = {}
    provenance_columns: dict[str, list[Any]] = {}
    for key, values in loaded.items():
        column, provenance = _hf_column_parts(key, values)
        if column is not None:
            columns[key] = column
        else:
            provenance_columns[key] = provenance or []
    provenance_records = tuple(
        {key: values[row] for key, values in provenance_columns.items()} for row in range(size)
    )
    return columns, provenance_records if provenance_columns else ()


@dataclass(frozen=True)
class HFEagerConfig(SourceConfigBase):
    """Configuration for HFEagerSource (loads all data into host columns at init).

    Configuration for eager-loading HuggingFace datasets into host NumPy columns.

    Args:
        name: Name of the dataset in HuggingFace Hub (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional folder inside the dataset's repository whose data files are
            loaded (``datasets.load_dataset``'s ``data_dir``), not a storage location
        cache_dir: Optional folder where downloaded files are cached (``load_dataset``'s
            ``cache_dir``; the Hugging Face default when ``None``)
        download_kwargs: Optional keyword arguments for load_dataset
        include_keys: Optional set of keys to include in output (exclusive with exclude_keys)
        exclude_keys: Optional set of keys to exclude from output (exclusive with include_keys)

    Note:
        The order records are served in belongs to the pipeline (``Pipeline(shuffle=...)``).
    """

    cache_dir: str | None = None
    download_kwargs: dict[str, Any] | None = None
    local_files_only: bool = False


@dataclass(frozen=True)
class HFStreamingConfig(StreamingSourceConfigBase):
    """Configuration for HFStreamingSource (streams a dataset with HF's streaming mode).

    Args:
        name: Name of the dataset in HuggingFace Hub (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional folder inside the dataset's repository whose data files are
            loaded (``datasets.load_dataset``'s ``data_dir``), not a storage location
        cache_dir: Optional folder where downloaded files are cached (``load_dataset``'s
            ``cache_dir``; the Hugging Face default when ``None``)
        shuffle_buffer_size: Records HF's shuffle buffer holds when the pipeline shuffles
        download_kwargs: Optional keyword arguments for load_dataset
        include_keys: Optional set of keys to include in output
        exclude_keys: Optional set of keys to exclude from output

    Note:
        Whether a pass is shuffled, and by which seed, is the pipeline's
        (``Pipeline(shuffle=...)``). A map-style dataset is read by ``HFEagerSource``.
    """

    cache_dir: str | None = None
    download_kwargs: dict[str, Any] | None = None
    local_files_only: bool = False


# =============================================================================
# HFEagerSource - Loads All Data into Host Columns at Init
# =============================================================================


class HFEagerSource(DatasetSourceMixin, EagerSource):
    """Eager-loading HuggingFace source for small/medium datasets.

    Loads ALL data at initialization: every array column (numbers, images, numeric sequences)
    becomes one host NumPy column, and every other column (text, objects) the records'
    provenance, kept beside the columns and never refused. It then serves its records as every
    eager source does (:class:`~datarax.sources.eager_source.EagerSource`).

    Key Features:
        - One-time conversion at init (PIL images to NumPy arrays)
        - No HuggingFace work during training
        - Holds no iteration state: the pipeline owns the order and the position

    Example:
        ```python
        # Create eager source for MNIST from HuggingFace
        config = HFEagerConfig(name="mnist", split="train")
        source = HFEagerSource(config)

        # Iterate in order
        for item in source:
            process(item["image"])

        # Read records 0..31 as a Batch
        batch = source.get_batch(to_words(np.arange(32)))
        ```
    """

    def __init__(
        self,
        config: HFEagerConfig,
        *,
        name: str | None = None,
    ) -> None:
        """Initialize HFEagerSource by loading all data into host columns and provenance.

        Args:
            config: Configuration for the source
            name: Optional name (defaults to HFEagerSource(dataset:split))
        """
        if name is None:
            name = f"HFEagerSource({config.name}:{config.split})"
        super().__init__(config, name=name)

        # Store config for feature access
        self.dataset_name = config.name
        self.split_name = config.split
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load the dataset once: its info and ALL its data, at init
        dataset = _load_hf_dataset(config)
        # Only the single-split Dataset variants of load_dataset's return union carry ``info``.
        self._dataset_info = HostValue(getattr(dataset, "info", None))
        self._store(*self._load_columns(dataset, config))

        # Clean up resources
        gc.collect()

    def _load_columns(
        self, dataset: Any, config: HFEagerConfig
    ) -> tuple[dict[str, np.ndarray], tuple[dict[str, Any], ...]]:
        """The loaded dataset as host NumPy columns and per-record provenance.

        All HuggingFace work happens here, at init time.

        Args:
            dataset: The dataset ``load_dataset`` returned
            config: Source configuration

        Returns:
            The array columns, and one mapping per record of its non-array columns (empty when
            every column is an array).

        Raises:
            ValueError: If the dataset yields no elements after loading and key filtering.
        """
        keys = _selected_hf_columns(dataset.column_names, config.include_keys, config.exclude_keys)
        if not keys or len(dataset) == 0:
            raise ValueError(
                "Dataset produced no elements after loading/filtering. "
                "Check split selection and include/exclude key filters."
            )

        # The numpy format decodes and stacks each column once, so a numeric column is one
        # NumPy array, read without per-row work.
        return _hf_parts(dataset.select_columns(keys).with_format("numpy")[:], len(dataset))


# =============================================================================
# HFStreamingSource - Streams with HuggingFace's Streaming Mode
# =============================================================================


def _feature_dtypes(features: Any, keys: Collection[str]) -> dict[str, np.dtype]:
    """The NumPy dtype each array feature declares, which HF's numpy format may widen.

    HF's numpy formatter returns an ``Array3D(dtype="uint8")`` column as int64; the declared dtype
    is the column's.
    """
    dtypes: dict[str, np.dtype] = {}
    for key in keys:
        dtype = getattr(features.get(key) if features else None, "dtype", None)
        try:
            resolved = np.dtype(dtype) if dtype is not None else None
        except TypeError:  # a feature dtype NumPy does not name, such as "string"
            resolved = None
        if resolved is not None and resolved.kind in JAX_ARRAY_KINDS:
            dtypes[key] = resolved
    return dtypes


class HFStreamingSource(StreamingSourceBase):
    """A HuggingFace dataset streamed with HF's streaming mode, records named by arrival.

    The dataset is loaded with ``load_dataset(..., streaming=True)``; a map-style dataset is
    ``HFEagerSource``'s. Each pass reads batched NumPy columns (``with_format("numpy").iter``):
    an array column becomes a host NumPy column in its feature's dtype, text and other objects
    the records' provenance, beside the batch. A HuggingFace stream reports no record ids, so
    records are named by their arrival ordinal, never reset (``ARRIVAL``), and lookups by record
    (``provenance(indices)``, ``record_keys``) are refused.

    The order is the pipeline's to choose: without its key a pass serves the dataset's order;
    with it, HF's buffer shuffle (``IterableDataset.shuffle(seed, buffer_size)``) is seeded from
    the key, and ``set_epoch(pass)`` gives each pass its own order.

    Example:
        ```python
        source = HFStreamingSource(HFStreamingConfig(name="allenai/c4", split="train",
                                                     download_kwargs={"name": "en"}))
        pipeline = Pipeline(source=source, stages=[tokenize], batch_size=64, rngs=nnx.Rngs(0),
                            shuffle=True, num_epochs=None)
        for batch in pipeline:
            train_step(batch)
        ```
    """

    config: HFStreamingConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def record_identity(self) -> RecordIdentity:
        """A HuggingFace stream reports no record ids, so a record is named by its arrival."""
        return RecordIdentity.ARRIVAL

    def __init__(self, config: HFStreamingConfig, *, name: str | None = None) -> None:
        """Load the dataset in HuggingFace's streaming mode.

        Args:
            config: Configuration for the source
            name: Optional name (defaults to HFStreamingSource(dataset:split))
        """
        if name is None:
            name = f"HFStreamingSource({config.name}:{config.split})"
        super().__init__(config, name=name)
        self._dataset = HostValue(_load_hf_dataset(config, streaming=True))

    @property
    def dataset_name(self) -> str | None:
        """Dataset name from source config."""
        return self.config.name

    @property
    def split_name(self) -> str | None:
        """Dataset split from source config."""
        return self.config.split

    def get_dataset_info(self) -> Any:
        """The streamed dataset's ``DatasetInfo``."""
        return getattr(self._dataset.value, "info", None)

    def __len__(self) -> int:  # noqa: DOC201 - it only raises
        """A stream's length is unknown.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Length unknown for streaming dataset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HFStreamingSource(dataset={self.dataset_name}:{self.split_name}, "
            f"shuffle_buffer_size={self.config.shuffle_buffer_size})"
        )

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index``: the dataset's order, or HF's seeded shuffle under ``key``.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key as its host words when it shuffles, ``None`` for the
                dataset's order.
            read_size: Records per batched read.

        Yields:
            The pass's records as chunks.
        """
        dataset = self._dataset.value
        features = getattr(dataset, "features", None)
        dtypes = _feature_dtypes(features, list(features or ()))
        if key is not None:
            dataset = dataset.shuffle(
                seed=key_integer(key), buffer_size=self.config.shuffle_buffer_size
            )
        dataset = dataset.with_format("numpy")
        if key is not None:
            # After with_format, which copies the dataset without its epoch (datasets 5.0.1).
            dataset.set_epoch(pass_index)
        # HF's batched reader is a generator over Parquet generators; closing it here, when this
        # pass is closed, finalizes them while the interpreter runs (left to module teardown,
        # HF's Parquet reader hangs).
        with contextlib.closing(dataset.iter(batch_size=read_size)) as batches:
            for rows in batches:
                size = len(next(iter(rows.values())))
                kept = _selected_hf_columns(
                    list(rows), self.config.include_keys, self.config.exclude_keys
                )
                cast = {
                    column: np.asarray(rows[column]).astype(dtypes[column], copy=False)
                    if column in dtypes
                    else rows[column]
                    for column in kept
                }
                columns, provenance = _hf_parts(cast, size)
                yield StreamChunk(
                    columns,
                    tuple(MappingProxyType(record) for record in provenance)
                    if provenance
                    else (NO_PROVENANCE,) * size,
                    None,
                )


# =============================================================================
# Choosing the source
# =============================================================================


def from_hf(
    name: str,
    split: str,
    *,
    streaming: bool = False,
    data_dir: str | None = None,
    cache_dir: str | None = None,
    include_keys: AbstractSet[str] | None = None,
    exclude_keys: AbstractSet[str] | None = None,
    download_kwargs: dict[str, Any] | None = None,
) -> DataSourceModule:
    """Create a HuggingFace source: the eager source, or the stream when asked.

    ``HFEagerSource`` loads the dataset whole into host columns at init; with ``streaming=True``,
    ``HFStreamingSource`` streams it with HuggingFace's streaming mode, for datasets too large to
    hold. The order records are served in belongs to the pipeline (``Pipeline(shuffle=...)``).

    Args:
        name: HuggingFace dataset name (e.g., "mnist", "imdb", "allenai/c4")
        split: Dataset split (e.g., "train", "test")
        streaming: Whether to stream the dataset rather than load it whole
        data_dir: Optional folder inside the dataset's repository whose data files are
            loaded (``datasets.load_dataset``'s ``data_dir``), not a storage location
        cache_dir: Optional folder where downloaded files are cached
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude
        download_kwargs: Optional kwargs for datasets.load_dataset

    Returns:
        HFEagerSource, or HFStreamingSource with ``streaming=True``.

    Example:
        ```python
        from datarax.sources import from_hf

        source = from_hf("ylecun/mnist", "train")  # loaded whole
        stream = from_hf("allenai/c4", "train", streaming=True, download_kwargs={"name": "en"})
        ```
    """
    if streaming:
        return HFStreamingSource(
            HFStreamingConfig(
                name=name,
                split=split,
                data_dir=data_dir,
                cache_dir=cache_dir,
                include_keys=include_keys,
                exclude_keys=exclude_keys,
                download_kwargs=download_kwargs,
            )
        )
    return HFEagerSource(
        HFEagerConfig(
            name=name,
            split=split,
            data_dir=data_dir,
            cache_dir=cache_dir,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            download_kwargs=download_kwargs,
        )
    )


__all__ = [
    "HFEagerConfig",
    "HFEagerSource",
    "HFStreamingConfig",
    "HFStreamingSource",
    "from_hf",
]
