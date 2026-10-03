"""Grain's own mix, read through its public API, as the reference a mixed source is held to.

``grain.MapDataset.mix`` over ``MapDataset.range(offset, offset + length)`` children serves, at
mix position ``k``, the value ``offset_c + j``: the child ``c`` Grain selects for ``k`` and the
child position ``j`` it reads there. With each child's offset the sum of the lengths before it,
that value is the mixed record's index ``MixDataSourcesNode`` names when its children serve
their records in order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import grain
import jax
import numpy as np

from datarax.core import batch_ops
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, record_words, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


def offsets_of(lengths: Sequence[int]) -> list[int]:
    """Where each child's records start in the mix's index space."""
    return [sum(lengths[:position]) for position in range(len(lengths))]


def grain_mix(lengths: Sequence[int], weights: Sequence[float]) -> grain.MapDataset:
    """Grain's mix of children holding the index-space ranges of ``lengths``."""
    return grain.MapDataset.mix(
        [
            grain.MapDataset.range(offset, offset + length)
            for offset, length in zip(offsets_of(lengths), lengths, strict=True)
        ],
        weights=list(weights),
    )


def grain_mix_indices(
    lengths: Sequence[int], weights: Sequence[float], positions: Iterable[int]
) -> list[int]:
    """The mixed record index Grain's mix serves at each of ``positions``."""
    mixed = grain_mix(lengths, weights)
    return [int(cast(int, mixed[position])) for position in positions]


@dataclass(frozen=True)
class SizedConfig(StructuralConfig):
    """Configuration of :class:`Sized` (nothing to configure)."""


class Sized(DataSourceModule):
    """An indexed source with a length, a one-float record spec and the default naming.

    It names position ``p`` of an unshuffled pass ``p`` itself, so a mix over such children
    names, without a key, exactly the indices :func:`grain_mix_indices` gives. Its host read
    serves each record's index as its value, so any length costs no memory.
    """

    def __init__(self, length: int) -> None:
        super().__init__(SizedConfig())
        self.rows = length

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __len__(self) -> int:
        return self.rows

    def element_spec(self) -> Any:
        """One float32 scalar per record."""
        return {"x": jax.ShapeDtypeStruct((), jax.numpy.float32)}

    def get_batch(self, indices: Any, *, epochs: Any = 0, contiguous: bool = False) -> Batch:
        """The records ``indices`` names, each record's value its index."""
        del contiguous
        words = record_words(indices)
        values = from_words(words).astype(np.float32)
        epoch_of_each = np.broadcast_to(np.asarray(epochs, np.int32), (len(values),)).copy()
        return batch_ops.from_arrays({"x": values}).replace(indices=words, epochs=epoch_of_each)


_SHAPES = {"image": (4, 4, 3), "text": (6,)}
"""The image and text shapes of C4d's fill fixture (``tests/core/test_maybe.py``)."""

_PRESENCE_CASES = (("image", "text", "label"), ("image", "label"), ("text", "label"), ("label",))
"""Both present, image only, text only, neither."""

_RECORDS = 6
"""Records per child of :func:`four_presence_cases`."""


def child_columns(child: int, names: Sequence[str]) -> dict[str, np.ndarray]:
    """Host columns whose values name their child, so a wrong row is a wrong value."""
    rng = np.random.default_rng(child)
    columns = {
        "image": rng.random((_RECORDS, *_SHAPES["image"])).astype(np.float32) + child + 1,
        "text": rng.random((_RECORDS, *_SHAPES["text"])).astype(np.float32) + child + 1,
        "label": np.full(_RECORDS, child, np.int32),
    }
    return {name: columns[name] for name in names}


def four_presence_cases() -> MixDataSourcesNode:
    """A 1:1:1:1 mix of one child per presence case of ``image`` and ``text``, all with ``label``.

    Unshuffled, position ``k`` is child ``k % 4``'s record ``k // 4``.
    """
    children: list[DataSourceModule] = [
        MemorySource(MemorySourceConfig(), child_columns(child, names))
        for child, names in enumerate(_PRESENCE_CASES)
    ]
    return MixDataSourcesNode(MixDataSourcesConfig(weights=(0.25,) * 4), children)
