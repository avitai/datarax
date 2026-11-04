"""What one steady host read of a disk source costs in Python calls: counts, never times.

The record's nesting is static, so the source derives it once from its columns' key tuples and
a steady ``get_batch`` only places its arrays: it rebuilds no nesting and splits no string,
whether its columns are named by one key each or by key tuples, and each column adds only its
map's accessor and its gather. Counted with substrax's ``counted_calls`` over a short and a long
pass (``per_iteration``), so work done once per pass, the first read's map opening included,
cancels (OWN-1009-TESTS-HARDWARE-INDEPENDENT).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from substrax.testing import by_package, counted_calls, per_iteration

from datarax.core.field_paths import FieldName
from datarax.core.index_words import to_words
from datarax.sources.streaming_disk_source import (
    DiskColumn,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


_ROWS = 64
_BATCH = 8
_KEYS = by_package("datarax", named=["datarax"])


def _string_splits(callee: object, first: object) -> int:
    """One for each call of a string's ``split``, the old spelling rebuild's step.

    A method call reports the unbound ``str.split`` with the string as its first argument; a
    bound method called later reports the builtin bound to its string.
    """
    del first
    if callee is str.split:
        return 1
    bound = getattr(callee, "__self__", None)
    return int(isinstance(bound, str) and getattr(callee, "__name__", "") == "split")


def _one_key(column: int) -> FieldName:
    return f"c{column}"


def _key_tuple(column: int) -> FieldName:
    """A nested field and a literal key holding the separator, in turn."""
    return ("meta", f"c{column}") if column % 2 else (f"a/b{column}",)


_NAMES: dict[str, Callable[[int], FieldName]] = {"one-key": _one_key, "key-tuples": _key_tuple}


def _source(directory: Path, naming: str, columns: int) -> StreamingDiskSource:
    name = _NAMES[naming]
    for column in range(columns):
        np.save(directory / f"c{column}.npy", np.zeros((_ROWS, 2), np.float32))
    return StreamingDiskSource(
        StreamingDiskSourceConfig(
            columns=tuple(
                DiskColumn.from_npy(name(column), directory / f"c{column}.npy")
                for column in range(columns)
            )
        )
    )


def _per_batch(source: StreamingDiskSource) -> dict[str, int]:
    words = to_words(np.arange(_BATCH, dtype=np.uint64))

    def count(batches: int) -> dict[str, int]:
        source.get_batch(words)  # warm: the maps are open before the counted pass
        with counted_calls(_KEYS, weights={"str.split": _string_splits}) as counts:
            for _ in range(batches):
                source.get_batch(words)
        return dict(counts)

    return per_iteration(count, short=2, long=6)


@pytest.mark.parametrize("naming", list(_NAMES))
def test_a_steady_read_rebuilds_no_nesting_and_splits_no_string(
    naming: str, tmp_path: Path
) -> None:
    work = _per_batch(_source(tmp_path, naming, columns=2))

    assert work.get("datarax:StreamingDiskSource.get_batch") == 1, f"positive control: {work}"
    found = {key: work.get(key, 0) for key in ("datarax:fields_nested", "str.split")}
    assert found == {"datarax:fields_nested": 0, "str.split": 0}, work


@pytest.mark.parametrize("naming", list(_NAMES))
def test_each_column_adds_only_its_map_and_its_gather(naming: str, tmp_path: Path) -> None:
    (tmp_path / "two").mkdir()
    (tmp_path / "six").mkdir()
    two = _per_batch(_source(tmp_path / "two", naming, columns=2))
    six = _per_batch(_source(tmp_path / "six", naming, columns=6))

    grown = {
        key: six.get(key, 0) - two.get(key, 0)
        for key in {*two, *six}
        if key.startswith("datarax") and six.get(key, 0) != two.get(key, 0)
    }
    added = 6 - 2
    each = ("datarax:HostArray.array", "datarax:take_rows.<locals>.<lambda>")  # map, gather
    assert grown == {"datarax": len(each) * added, **dict.fromkeys(each, added)}, (two, six)


def test_positive_control_the_counter_sees_a_string_split() -> None:
    def count(batches: int) -> dict[str, int]:
        text = "a/b"
        with counted_calls(_KEYS, weights={"str.split": _string_splits}) as counts:
            for _ in range(batches):
                text.split("/")
                split = text.split
                split("/")
        return dict(counts)

    assert per_iteration(count, short=2, long=6).get("str.split") == 2, "both call forms"
