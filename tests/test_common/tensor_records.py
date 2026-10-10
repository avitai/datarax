"""TFDS-serialized tensor records, read in Grain worker processes by the tests.

:func:`write_tensor_records` writes ArrayRecord files of TFDS-serialized examples holding a
16,384-value float ``Tensor`` (a non-image kind); :class:`TensorDecoder` decodes them with TFDS's
NumPy decoder, as ``from_tfds(..., in_memory=False)`` does.

This module imports ``tensorflow_datasets`` at its top, so only a test that reads TFDS records
imports it (and TFDS with it); a worker process unpickling :class:`TensorDecoder` imports TFDS
when it imports this module, before its first read, never inside one. TFDS's first import of a
process swaps ``builtins.__import__`` while it runs (``datarax.sources.tfds_source`` says why
that must not happen mid-run).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow_datasets as tfds

from tests.test_common.worker_reads import write_records


TENSOR_WIDTH = 16_384


@functools.cache
def _tensor_features(width: int) -> Any:
    return tfds.features.FeaturesDict(
        {
            "x": tfds.features.Tensor(shape=(width,), dtype=np.float32),
            "label": tfds.features.Scalar(dtype=np.int64),
            "name": tfds.features.Text(),
        }
    )


def tensor_values(index: int, width: int = TENSOR_WIDTH) -> np.ndarray:
    """Record ``index``'s ``x``."""
    return np.random.default_rng(index).standard_normal(width).astype(np.float32)


def write_tensor_records(
    directory: Path, count: int, *, width: int = TENSOR_WIDTH, shards: int = 2
) -> list[str]:
    """ArrayRecord files of ``count`` TFDS-serialized examples with a ``width``-value float tensor.

    Args:
        directory: Where the files go.
        count: The records.
        width: Values in each record's ``x``.
        shards: The files they are split into, in order.

    Returns:
        The files' paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    features = _tensor_features(width)
    records = [
        features.serialize_example(
            {"x": tensor_values(index, width), "label": index, "name": f"example-{index}"}
        )
        for index in range(count)
    ]
    return write_records(directory, "tensor", records, shards)


@dataclass(frozen=True, slots=True)
class TensorDecoder:
    """Decodes a batch of :func:`write_tensor_records` examples with TFDS's NumPy decoder.

    Attributes:
        width: Values in each record's ``x``.
    """

    width: int = TENSOR_WIDTH

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """One mapping per record: ``x``, ``label`` and the ``name`` (provenance)."""
        features = _tensor_features(self.width)
        return [features.deserialize_example_np(record) for record in records]
