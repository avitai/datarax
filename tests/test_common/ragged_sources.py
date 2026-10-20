"""Ragged records in a source a decoded copy can be written from, for the format 2 tests.

No shipped source both declares an identity (which a decoded copy needs) and serves the ragged
read (MemorySource and HFEagerSource do; ArrayRecord and TFDS sources do not read ragged records
yet). :class:`IdentifiedRaggedSource` is a ``MemorySource`` that declares an identity and a
GIL-bound read, so a budget decodes it in worker processes, and logs every ragged read it serves
with the reading process's id (a read counter that crosses processes). It is defined at module
level, so it pickles by reference into a worker process.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from jax.typing import ArrayLike

from datarax.core.data_source import HostRead
from datarax.sources import MemorySource, MemorySourceConfig


RAGGED = ("meta/mask", "tokens")
"""The ragged fields of :func:`ragged_records`."""

FIELDS = (*RAGGED, "label", "pair")
"""Every numeric field of :func:`ragged_records`."""


def ragged_records(count: int, *, seed: int = 0, longest: int = 9) -> list[dict[str, Any]]:
    """``count`` records whose ``tokens`` and ``meta/mask`` differ in length, zero included.

    Args:
        count: The records.
        seed: The lengths' and values' seed.
        longest: One more than the longest record.

    Returns:
        One mapping per record: ``tokens`` int32 ``(n,)``, ``meta/mask`` float32 ``(n, 2)``,
        ``label`` int64, ``pair`` float32 ``(3,)`` and ``name``, a string (its provenance).
    """
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, longest, count)
    return [
        {
            "tokens": rng.integers(0, 1000, length).astype(np.int32),
            "meta": {"mask": rng.random((length, 2)).astype(np.float32)},
            "label": np.int64(index),
            "pair": rng.random(3).astype(np.float32),
            "name": f"record-{index}",
        }
        for index, length in enumerate(lengths)
    ]


class IdentifiedRaggedSource(MemorySource):
    """In-memory ragged records declaring an identity and a GIL-bound read, logging its reads."""

    def __init__(
        self, records: list[dict[str, Any]], *, seed: int = 0, log: Path | None = None
    ) -> None:
        """Hold ``records``.

        Args:
            records: The records (:func:`ragged_records`).
            seed: Part of the declared identity, so two record sets are two sources.
            log: A file every ragged read appends ``<pid> <field> <records>`` to, or ``None``.
        """
        super().__init__(MemorySourceConfig(), records)
        self._seed = seed
        self._log = None if log is None else str(log)

    @property
    def host_read(self) -> HostRead:
        """Declared GIL-bound, so a budget reads it in worker processes."""
        return HostRead.GIL_BOUND

    def source_identity(self) -> dict[str, Any]:
        """The records' seed and count; nothing decodes, so no decoder."""
        return {
            "source": {"kind": "test_ragged", "seed": self._seed, "records": len(self)},
            "decoder": {},
        }

    def get_ragged(self, indices: ArrayLike, field: str) -> tuple[np.ndarray, np.ndarray]:
        """The ragged read, logged."""
        values, offsets = super().get_ragged(indices, field)
        if self._log is not None:
            with Path(self._log).open("a", encoding="utf-8") as log:
                log.write(f"{os.getpid()} {field} {len(offsets) - 1}\n")
        return values, offsets


class MisdeclaredRaggedSource(IdentifiedRaggedSource):
    """Names ``meta/mask`` fixed although its records differ in length: a source's defect."""

    @property
    def ragged_fields(self) -> tuple[str, ...]:
        """``tokens`` only."""
        return ("tokens",)


def reads(log: Path) -> list[tuple[int, str, int]]:
    """``(process id, field, records)`` of every ragged read an :class:`IdentifiedRaggedSource`
    logged."""
    if not log.exists():
        return []
    return [
        (int(pid), field, int(records))
        for pid, field, records in (line.split() for line in log.read_text().splitlines() if line)
    ]
