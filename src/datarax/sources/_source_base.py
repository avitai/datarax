"""Shared source pieces: the mixin of sources reading a named dataset, and the base of streams.

A stream reads its records forward, one pass after another, in an order it applies itself. The
:class:`StreamingSourceBase` owns everything but the reading: a subclass opens a pass
(``_open_pass``), a generator of :class:`~datarax.sources.stream_iterator.StreamChunk` (host
columns, provenance and record ids), and :meth:`StreamingSourceBase.iter_batches` reads the passes
as host ``Batch``es named by the stream, through one resumable iterator
(:class:`~datarax.sources.stream_iterator.StreamBatches`). The stream holds no position: where a
read stands is its iterator's, so no Python value sits in a Variable and the module's graph
definition never changes as the stream is read.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, overload

import jax
import numpy as np
from jax.typing import ArrayLike

from datarax.core.data_source import BatchSchedule, DataSourceModule, Provenance
from datarax.core.element_batch import Batch
from datarax.core.host_values import HostValue
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.sources.source_ops import format_source_repr
from datarax.sources.stream_iterator import chunk_size, StreamBatches, StreamChunk


class DatasetSourceMixin:
    """What a source reading a named dataset reports: its name, split and info.

    A subclass sets ``dataset_name``, ``split_name`` and ``_dataset_info`` in its ``__init__``.
    The info is held in a :class:`~datarax.core.host_values.HostValue`: a backend's info
    object compares by identity (TFDS's ``DatasetInfo``), so held as an attribute it would make
    every source a graphdef of its own and every jitted step over a new source compile again.
    """

    dataset_name: str | None
    split_name: str | None
    _dataset_info: HostValue
    length: int

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info.value

    def _repr_extra_fields(self) -> dict[str, Any]:
        """Optional additional repr fields for subclasses."""
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return format_source_repr(
            type(self).__name__,
            self.dataset_name,
            self.split_name,
            self.length,
            self._repr_extra_fields(),
        )


class StreamingSourceBase(DataSourceModule):
    """The base of every stream: a source read forward, one pass after another.

    A subclass declares its kind (``STREAM_IDS`` or ``ARRIVAL``) and implements
    :meth:`_open_pass`, which reads one pass in its order. The base serves it:

    - :meth:`iter_batches` reads the passes as host ``Batch``es: NumPy columns, ``indices`` the
      stream's ids as two words (``STREAM_IDS``) or the records' arrival ordinals (``ARRIVAL``),
      ``epochs`` each row's pass, draws 0. Strings and objects never enter a batch:
      ``with_provenance=True`` returns them beside it.
    - The order is the stream's to apply and the pipeline's to choose: the pipeline passes its key
      when it shuffles and ``None`` otherwise, and pass ``p`` is ordered by that key and ``p``.
    - :meth:`element_spec` is the array part of the first record, as the device holds it.
    """

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index`` in its order, as chunks of records: a subclass's generator.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key as its uint32 words on the host
                (:func:`~datarax.core.prng.key_words`) when it shuffles, ``None`` for the
                stream's own order. A pass's order is drawn from ``fold_in(key, pass_index)``
                (:func:`~datarax.core.prng.pass_seed`), folded on the CPU device.
            read_size: Records the pass reads at a time (a backend's batched read or decode);
                chunks may hold any number.

        Returns:
            The pass's records, in the order served.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _open_pass")

    @overload
    def iter_batches(  # noqa: PLR0913 - the read's batch rule, order and reads
        self,
        batch_size: int,
        *,
        num_passes: int | None = 1,
        key: ArrayLike | None = None,
        with_provenance: Literal[False] = False,
        read_size: int | None = None,
        drop_last: bool = False,
    ) -> StreamBatches[Batch]: ...

    @overload
    def iter_batches(  # noqa: PLR0913 - the read's batch rule, order and reads
        self,
        batch_size: int,
        *,
        num_passes: int | None = 1,
        key: ArrayLike | None = None,
        with_provenance: Literal[True],
        read_size: int | None = None,
        drop_last: bool = False,
    ) -> StreamBatches[tuple[Batch, Provenance]]: ...

    def iter_batches(  # noqa: PLR0913, DOC502 - the iterator refuses its arguments
        self,
        batch_size: int,
        *,
        num_passes: int | None = 1,
        key: ArrayLike | None = None,
        with_provenance: bool = False,
        read_size: int | None = None,
        drop_last: bool = False,
    ) -> StreamBatches[Any]:
        """Read the stream's passes as host ``Batch``es, through one resumable iterator.

        Args:
            batch_size: Records per batch.
            num_passes: Passes served, or ``None`` for no end.
            key: The key each pass's order is drawn from (typed, raw or its host words), or
                ``None`` for the stream's own order.
            with_provenance: Whether each batch comes with its records' provenance, one mapping
                per row.
            read_size: Records a pass reads at a time (the pipeline's batch size); ``None`` reads
                it ``batch_size`` at a time.
            drop_last: Whether a pass's records short of a full batch are skipped; otherwise they
                are completed from the next pass's head, each row keeping its pass.

        Returns:
            The iterator (:class:`~datarax.sources.stream_iterator.StreamBatches`), with
            ``get_state``/``set_state`` and ``close``.

        Raises:
            ValueError: If ``batch_size``, ``read_size`` or ``num_passes`` is below one.
        """
        return StreamBatches(
            self,
            batch_size,
            num_passes=num_passes,
            key=key,
            with_provenance=with_provenance,
            read_size=read_size,
            drop_last=drop_last,
        )

    def run_dataset(
        self, schedule: BatchSchedule, key: np.ndarray | None, *, for_workers: bool = False
    ) -> Any:
        """A run of this stream's passes as one sliceable Grain dataset, if the stream has one.

        A stream whose passes' orders are computed without reading its records (TFDS) returns
        a dataset serving the schedule's units, numbered from the run's start (each element's
        last item is its unit's place in the run), which Grain can slice across workers. This
        base has none, and the host stage reads it pass by pass.

        Args:
            schedule: The run's units, each a run of batches ``(start, pass, size)``.
            key: The pipeline's key as host words, or ``None`` for the stream's own order.
            for_workers: Whether worker processes read the run, so what every worker reads
                alike is shared among them rather than sent to each.

        Returns:
            ``None``: the base reads no run as one dataset.
        """
        del schedule, key, for_workers
        return None

    def element_spec(self) -> Any:
        """The spec of one record's array part as the device holds it, from the first record.

        A fresh first pass in the stream's own order is opened and its first record read.

        Returns:
            A pytree of ``jax.ShapeDtypeStruct``.

        Raises:
            ValueError: If the stream holds no records.
        """
        for chunk in self._open_pass(0, None, 1):
            if chunk_size(chunk):
                return device_spec(jax.tree.map(array_to_spec_strip_leading, chunk.columns))
        raise ValueError(f"{type(self).__name__} holds no records, so it declares no spec")
