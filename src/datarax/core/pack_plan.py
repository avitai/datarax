"""The pack plan: which records share each packed row, decided once from their lengths.

Packing (design D9) puts several variable-length records side by side in one row of a static
length, so a ragged dataset reaches the step at one shape. The plan is first fit, the default of
the field's packers: each record, in the order given, goes into the leftmost row with room for it,
and a row holding ``max_segments`` records takes no more. It reproduces Megatron-Bridge's
``first_fit`` (``megatron/bridge/data/packing/algorithms.py``) and NeMo-RL's ``FirstFitPacker``
(``nemo_rl/data/packing/algorithms.py``) on the same order, with the segment cap applied as Grain's
``BestFitPackedBatch`` applies it (a full row admits no record). A record has one length per packed
field and goes only where every field fits, as Grain's ``length_struct`` packing decides.

Records are planned a window of ``window`` consecutive records of the order at a time, and a
row's records always come from one window, so planning holds one window's lengths and rows,
never the dataset's: host memory grows with the window, not with the records. The window and
the segment cap are the caller's choice, with no default. A wider window fills rows better, but
reorders more records: WLB-LLM (arXiv 2503.17924, section 3.3) finds that final training loss
rises as the packing window spans more global batches. How much fill a window buys depends on
the lengths and the row length. Rows of 2048 over one lognormal length distribution (a probe,
not a rule) fill 0.9812, 0.9916, 0.9951, 0.9962 and 0.9967 at windows of 256, 1024, 4096,
16384 and 65536 records, and 0.9969 unwindowed.

The plan is a directory read through a memory map, whatever the source keeps its records in:

- ``records.bin``: every record's index, uint64, rows one after another;
- ``row_starts.bin``: int64, row ``r`` holding ``records[row_starts[r]:row_starts[r + 1]]``;
- ``plan.json``: the layout and counts, written last, so a directory without it holds no plan.

Planning a source of ``N`` records into ``R`` rows writes ``8 N + 8 (R + 1)`` bytes.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np


PLANNER_VERSION = 1
"""The plan layout and algorithm this module writes and reads; a plan of another is refused."""

PLAN_META = "plan.json"

RECORDS = "records.bin"

ROW_STARTS = "row_starts.bin"


def _check_layout(fields: set[str], row_lengths: Mapping[str, int], max_segments: int) -> None:
    """Refuse a layout first fit cannot plan, naming what is wrong.

    Args:
        fields: The fields whose lengths are given.
        row_lengths: Each packed field's row length.
        max_segments: The most records a row holds.

    Raises:
        ValueError: If no field is packed, a row length or ``max_segments`` is below 1, or the
            lengths and the row lengths name different fields.
    """
    if not row_lengths:
        raise ValueError("packing needs at least one field and its row length")
    short = {field: length for field, length in row_lengths.items() if length < 1}
    if short:
        raise ValueError(f"a row length must be at least 1; got {short}")
    if max_segments < 1:
        raise ValueError(f"max_segments must be at least 1; got {max_segments}")
    if fields != set(row_lengths):
        raise ValueError(
            f"lengths are given for {sorted(fields)} and row lengths for {sorted(row_lengths)}"
        )


def first_fit(
    records: np.ndarray,
    lengths: Mapping[str, np.ndarray],
    *,
    row_lengths: Mapping[str, int],
    max_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack ``records``, in their order, into rows by first fit.

    Args:
        records: The records' indices, in the order they are placed.
        lengths: Each packed field's length for every record, in the same order.
        row_lengths: Each packed field's row length.
        max_segments: The most records a row holds.

    Returns:
        The records grouped by row, each row's records in the order placed (uint64), and the row
        starts into them, ``R + 1`` of them for ``R`` rows (int64).

    Raises:
        ValueError: If the layout is invalid, or a record is longer than its row in a field,
            naming the record, the field, its length and the row length.
    """
    _check_layout(set(lengths), row_lengths, max_segments)
    fields = sorted(row_lengths)
    need = np.stack([np.asarray(lengths[field], np.int64) for field in fields], axis=1)
    capacity = np.asarray([row_lengths[field] for field in fields], np.int64)
    records = np.asarray(records, np.uint64)
    too_long = np.nonzero(need > capacity)
    if too_long[0].size:
        record, column = int(too_long[0][0]), int(too_long[1][0])
        raise ValueError(
            f"record {int(records[record])}'s {fields[column]!r} holds {int(need[record, column])} "
            f"cells, longer than its row length {int(capacity[column])}: raise the row length or "
            "shorten the record before packing"
        )
    count = len(records)
    remaining = np.empty((count, len(fields)), np.int64)
    held = np.zeros(count, np.int64)
    row_of = np.empty(count, np.int64)
    rows = 0
    for position in range(count):
        row = rows
        if rows:
            room = np.all(remaining[:rows] >= need[position], axis=1) & (held[:rows] < max_segments)
            first = int(np.argmax(room))
            if room[first]:
                row = first
        if row == rows:
            remaining[row] = capacity
            rows += 1
        remaining[row] -= need[position]
        held[row] += 1
        row_of[position] = row
    order = np.argsort(row_of, kind="stable")
    row_starts = np.concatenate([[0], np.cumsum(held[:rows])]).astype(np.int64)
    return records[order], row_starts


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class PackPlan:
    """A written plan, read through a memory map.

    Attributes:
        directory: Where the plan's files are.
        length: The records planned.
        rows: The rows they fill.
        row_lengths: Each packed field's row length.
        max_segments: The most records a row holds.
        window: The records planned together.
        records: Every record's index, rows one after another (uint64, memory-mapped).
        row_starts: Each row's start into ``records``, ``rows + 1`` of them (int64,
            memory-mapped).
    """

    directory: Path
    length: int
    rows: int
    row_lengths: Mapping[str, int]
    max_segments: int
    window: int
    records: np.ndarray
    row_starts: np.ndarray

    @classmethod
    def open(cls, directory: str | os.PathLike[str]) -> PackPlan:  # noqa: DOC503 - read_text raises the FileNotFoundError
        """Read the plan written in ``directory``.

        Args:
            directory: A directory :func:`plan_rows` wrote.

        Returns:
            The plan, its arrays memory-mapped.

        Raises:
            FileNotFoundError: If ``directory`` holds no finished plan (no ``plan.json``).
            ValueError: If the plan is of another version, naming both.
        """
        directory = Path(directory)
        meta = json.loads((directory / PLAN_META).read_text())
        if meta["version"] != PLANNER_VERSION:
            raise ValueError(
                f"{directory} holds a plan of version {meta['version']}; this datarax reads "
                f"version {PLANNER_VERSION}: plan again"
            )
        return cls(
            directory=directory,
            length=meta["length"],
            rows=meta["rows"],
            row_lengths=MappingProxyType(dict(meta["row_lengths"])),
            max_segments=meta["max_segments"],
            window=meta["window"],
            records=_mapped(directory / RECORDS, np.uint64, meta["length"]),
            row_starts=_mapped(directory / ROW_STARTS, np.int64, meta["rows"] + 1),
        )

    def take(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The records of ``rows``, in the order asked.

        Args:
            rows: Row indices, any order, repeats allowed.

        Returns:
            Their records, row after row (uint64), and each asked row's start into them,
            ``len(rows) + 1`` of them (int64).
        """
        rows = np.asarray(rows, np.int64)
        starts = np.asarray(self.row_starts[rows])
        held = np.asarray(self.row_starts[rows + 1]) - starts
        out_starts = np.concatenate([[0], np.cumsum(held)]).astype(np.int64)
        positions = np.arange(out_starts[-1]) - np.repeat(out_starts[:-1] - starts, held)
        return np.asarray(self.records[positions], np.uint64), out_starts


def _mapped(path: Path, dtype: type[np.generic], count: int) -> np.ndarray:
    """``count`` values of ``dtype`` in ``path``, read through a memory map (empty when 0)."""
    if count == 0:
        return np.zeros(0, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=(count,))


def plan_rows(  # noqa: PLR0913 - the plan's inputs, all keyword-only
    *,
    length: int,
    records_at: Callable[[int, int], np.ndarray],
    lengths_of: Callable[[np.ndarray], Mapping[str, np.ndarray]],
    row_lengths: Mapping[str, int],
    max_segments: int,
    window: int,
    directory: str | os.PathLike[str],
) -> PackPlan:
    """Plan ``length`` records into rows a window at a time and write the plan to ``directory``.

    Args:
        length: The records in the planning order.
        records_at: The records at positions ``[start, stop)`` of the order.
        lengths_of: Each packed field's length for the given records.
        row_lengths: Each packed field's row length.
        max_segments: The most records a row holds.
        window: The records planned together; no row spans two windows. No default: see the
            module docstring for what a window trades.
        directory: Where the plan is written (created if absent; a plan there is replaced).

    Returns:
        The written plan, read back through a memory map.

    Raises:
        ValueError: If ``window`` is below 1, or as :func:`first_fit` raises.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1; got {window}")
    _check_layout(set(row_lengths), row_lengths, max_segments)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLAN_META).unlink(missing_ok=True)
    rows = 0
    placed = 0
    with (
        (directory / RECORDS).open("wb") as records_file,
        (directory / ROW_STARTS).open("wb") as starts_file,
    ):
        starts_file.write(np.zeros(1, np.int64).tobytes())
        for start in range(0, length, window):
            records = np.asarray(records_at(start, min(start + window, length)), np.uint64)
            members, row_starts = first_fit(
                records, lengths_of(records), row_lengths=row_lengths, max_segments=max_segments
            )
            records_file.write(members.tobytes())
            starts_file.write((row_starts[1:] + placed).tobytes())
            rows += len(row_starts) - 1
            placed += len(members)
        for written in (records_file, starts_file):
            written.flush()
            os.fsync(written.fileno())
    meta = {
        "version": PLANNER_VERSION,
        "length": length,
        "rows": rows,
        "row_lengths": dict(row_lengths),
        "max_segments": max_segments,
        "window": window,
    }
    (directory / PLAN_META).write_text(json.dumps(meta, sort_keys=True))
    return PackPlan.open(directory)


__all__ = [
    "PLAN_META",
    "PLANNER_VERSION",
    "PackPlan",
    "first_fit",
    "plan_rows",
]
