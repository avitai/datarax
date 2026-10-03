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

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity


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
    names, without a key, exactly the indices :func:`grain_mix_indices` gives.
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
