"""The pack plan: records packed into rows by first fit, a window of the planning order at a time.

The plan is the field's first fit (each record, in the given order, into the leftmost row with
room), checked against two published implementations kept below as oracles, with Grain's segment
cap (a row holding ``max_segments`` records takes no more). Rows never span two windows of
``window`` consecutive records, so planning holds a window's lengths and rows, never the whole
dataset's. The plan is a file read through a memory map: the records in row order and each row's
start, 8 bytes per record and 8 per row boundary.
"""

from __future__ import annotations

import json
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest

from datarax.core.pack_plan import first_fit, PackPlan, plan_rows, PLANNER_VERSION


# ---------------------------------------------------------------------------------------------
# Oracles: published first-fit implementations, copied verbatim apart from formatting.
#
# Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the
# License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing permissions and
# limitations under the License.
#
# _MegatronSegmentTree and megatron_first_fit: NVIDIA-NeMo/Megatron-Bridge at 2b1c8ccd,
# src/megatron/bridge/data/packing/algorithms.py:34-131.
# nemo_rl_first_fit: NVIDIA-NeMo/RL at f9b4a98a, nemo_rl/data/packing/algorithms.py:471-514
# (FirstFitPacker._pack_implementation, in the given order).
# ---------------------------------------------------------------------------------------------


class _MegatronSegmentTree:
    def __init__(self, capacity: int) -> None:
        self._n = capacity
        self._tree = [0] * (4 * capacity)

    def _push_up(self, node: int) -> None:
        self._tree[node] = max(self._tree[2 * node], self._tree[2 * node + 1])

    def _update(self, node: int, start: int, end: int, idx: int, val: int) -> None:
        if start == end:
            self._tree[node] = val
            return
        mid = (start + end) // 2
        if idx <= mid:
            self._update(2 * node, start, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, end, idx, val)
        self._push_up(node)

    def _query(self, node: int, start: int, end: int, need: int) -> int:
        if self._tree[node] < need:
            return -1
        if start == end:
            return start
        mid = (start + end) // 2
        left = self._query(2 * node, start, mid, need)
        if left != -1:
            return left
        return self._query(2 * node + 1, mid + 1, end, need)

    def update(self, idx: int, val: int) -> None:
        self._update(1, 0, self._n - 1, idx, val)

    def query_first_fit(self, need: int) -> int:
        return self._query(1, 0, self._n - 1, need)


def megatron_first_fit(seqlens: Sequence[int], pack_size: int) -> list[list[int]]:
    """Megatron-Bridge's ``first_fit``, returning each bin's item positions."""
    if not seqlens:
        return []
    n = len(seqlens)
    tree = _MegatronSegmentTree(n)
    res: list[list[int]] = []
    remaining: list[int] = []
    for item, length in enumerate(seqlens):
        first_bin = tree.query_first_fit(length)
        if first_bin == -1 or first_bin >= len(res):
            new_idx = len(res)
            res.append([item])
            remaining.append(pack_size - length)
            tree.update(new_idx, remaining[new_idx])
        else:
            res[first_bin].append(item)
            remaining[first_bin] -= length
            tree.update(first_bin, remaining[first_bin])
    return res


def nemo_rl_first_fit(
    sequence_lengths: Sequence[int], bin_capacity: int, max_sequences_per_bin: int | None
) -> list[list[int]]:
    """NeMo-RL's ``FirstFitPacker`` over the lengths in their given order."""
    indexed_lengths = [(length, i) for i, length in enumerate(sequence_lengths)]
    bins: list[list[int]] = []
    bin_remaining: list[int] = []
    for length, idx in indexed_lengths:
        if length > bin_capacity:
            raise ValueError(f"Sequence length {length} exceeds bin capacity {bin_capacity}")
        bin_found = False
        for i, remaining in enumerate(bin_remaining):
            has_sequence_room = (
                max_sequences_per_bin is None or len(bins[i]) < max_sequences_per_bin
            )
            if remaining >= length and has_sequence_room:
                bins[i].append(idx)
                bin_remaining[i] -= length
                bin_found = True
                break
        if not bin_found:
            bins.append([idx])
            bin_remaining.append(bin_capacity - length)
    return bins


# ---------------------------------------------------------------------------------------------
# A linear scan over several fields: a record goes where every field fits (Grain's rule).
# ---------------------------------------------------------------------------------------------


def linear_first_fit(
    lengths: Mapping[str, Sequence[int]], row_lengths: Mapping[str, int], max_segments: int
) -> list[list[int]]:
    """First fit over several fields, each with its own row length, at most ``max_segments``."""
    count = len(next(iter(lengths.values())))
    rows: list[list[int]] = []
    used: list[dict[str, int]] = []
    for record in range(count):
        for row, taken in zip(rows, used, strict=True):
            fits = all(taken[f] + lengths[f][record] <= row_lengths[f] for f in row_lengths)
            if fits and len(row) < max_segments:
                row.append(record)
                for f in row_lengths:
                    taken[f] += lengths[f][record]
                break
        else:
            rows.append([record])
            used.append({f: lengths[f][record] for f in row_lengths})
    return rows


def _as_rows(members: np.ndarray, row_starts: np.ndarray) -> list[list[int]]:
    """A plan's rows as lists of records."""
    return [members[a:b].tolist() for a, b in zip(row_starts[:-1], row_starts[1:], strict=True)]


def _lognormal(seed: int, count: int, row_length: int) -> np.ndarray:
    """Lengths like a text corpus's: lognormal, clipped to [1, row_length]."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.lognormal(4.0, 1.0, count).astype(np.int64), 1, row_length)


def _plan_in(
    directory: Path,
    lengths: Mapping[str, np.ndarray],
    *,
    row_lengths: Mapping[str, int],
    max_segments: int,
    window: int,
    order: np.ndarray | None = None,
) -> PackPlan:
    """Plan records ``order`` (default ``0..N-1``) whose lengths are ``lengths``."""
    count = len(next(iter(lengths.values())))
    order = np.arange(count, dtype=np.uint64) if order is None else order

    def records_at(start: int, stop: int) -> np.ndarray:
        return order[start:stop]

    def lengths_of(records: np.ndarray) -> dict[str, np.ndarray]:
        return {f: values[records.astype(np.int64)] for f, values in lengths.items()}

    return plan_rows(
        length=count,
        records_at=records_at,
        lengths_of=lengths_of,
        row_lengths=row_lengths,
        max_segments=max_segments,
        window=window,
        directory=directory,
    )


# ---------------------------------------------------------------------------------------------
# One window: the field's first fit.
# ---------------------------------------------------------------------------------------------


class TestFirstFitIsTheFieldsFirstFit:
    @pytest.mark.parametrize(("seed", "count"), [(0, 1), (1, 7), (2, 200), (3, 2000)])
    def test_equals_megatron_bridge_without_a_cap(self, seed: int, count: int) -> None:
        lengths = _lognormal(seed, count, 512)
        records = np.arange(count, dtype=np.uint64)
        members, starts = first_fit(
            records, {"tokens": lengths}, row_lengths={"tokens": 512}, max_segments=count
        )
        assert _as_rows(members, starts) == megatron_first_fit(lengths.tolist(), 512)

    @pytest.mark.parametrize("max_segments", [1, 2, 4, 8, 32])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equals_nemo_rl_with_the_segment_cap(self, max_segments: int, seed: int) -> None:
        lengths = _lognormal(seed, 1500, 512)
        records = np.arange(1500, dtype=np.uint64)
        members, starts = first_fit(
            records, {"tokens": lengths}, row_lengths={"tokens": 512}, max_segments=max_segments
        )
        assert _as_rows(members, starts) == nemo_rl_first_fit(lengths.tolist(), 512, max_segments)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_several_fields_go_where_every_field_fits(self, seed: int) -> None:
        rows_needed = {"tokens": 256, "targets": 96}
        lengths = {"tokens": _lognormal(seed, 800, 256), "targets": _lognormal(seed + 7, 800, 96)}
        records = np.arange(800, dtype=np.uint64)
        members, starts = first_fit(records, lengths, row_lengths=rows_needed, max_segments=6)
        assert _as_rows(members, starts) == linear_first_fit(
            {f: v.tolist() for f, v in lengths.items()}, rows_needed, 6
        )

    def test_returns_the_records_given_not_their_positions(self) -> None:
        records = np.array([90, 13, 7, 2**40 + 5], dtype=np.uint64)
        lengths = np.array([300, 300, 200, 100])
        members, starts = first_fit(
            records, {"tokens": lengths}, row_lengths={"tokens": 512}, max_segments=4
        )
        assert _as_rows(members, starts) == [[90, 7], [13, 2**40 + 5]]
        assert members.dtype == np.uint64
        assert starts.dtype == np.int64

    def test_every_record_appears_once_and_no_row_overflows(self) -> None:
        lengths = _lognormal(5, 3000, 300)
        records = np.arange(3000, dtype=np.uint64)
        members, starts = first_fit(
            records, {"tokens": lengths}, row_lengths={"tokens": 300}, max_segments=5
        )
        assert sorted(members.tolist()) == list(range(3000))
        for row in _as_rows(members, starts):
            assert 1 <= len(row) <= 5
            assert int(lengths[row].sum()) <= 300

    def test_a_zero_length_record_takes_a_segment_slot(self) -> None:
        records = np.arange(4, dtype=np.uint64)
        lengths = np.array([5, 0, 0, 5])
        members, starts = first_fit(
            records, {"tokens": lengths}, row_lengths={"tokens": 10}, max_segments=2
        )
        assert _as_rows(members, starts) == [[0, 1], [2, 3]]

    def test_a_record_longer_than_its_row_is_refused_naming_it(self) -> None:
        records = np.array([4, 2**33 + 1], dtype=np.uint64)
        lengths = {"tokens": np.array([10, 600]), "targets": np.array([1, 1])}
        with pytest.raises(ValueError, match=r"record 8589934593.*'tokens'.*600.*row length 512"):
            first_fit(records, lengths, row_lengths={"tokens": 512, "targets": 4}, max_segments=8)

    @pytest.mark.parametrize(
        ("row_lengths", "max_segments", "message"),
        [
            ({"tokens": 0}, 4, "row length"),
            ({"tokens": 8}, 0, "max_segments"),
            ({}, 4, "at least one"),
            ({"other": 8}, 4, "'tokens'"),
        ],
    )
    def test_invalid_layouts_are_refused(
        self, row_lengths: dict[str, int], max_segments: int, message: str
    ) -> None:
        records = np.arange(3, dtype=np.uint64)
        with pytest.raises(ValueError, match=message):
            first_fit(
                records,
                {"tokens": np.array([1, 2, 3])},
                row_lengths=row_lengths,
                max_segments=max_segments,
            )


# ---------------------------------------------------------------------------------------------
# Windows and the plan file.
# ---------------------------------------------------------------------------------------------


class TestWindows:
    @pytest.mark.parametrize(
        ("count", "window"), [(1000, 64), (1000, 1000), (1000, 3000), (999, 100)]
    )
    def test_each_window_is_first_fit_on_its_own_records(
        self, tmp_path: Path, count: int, window: int
    ) -> None:
        lengths = _lognormal(11, count, 256)
        plan = _plan_in(
            tmp_path, {"tokens": lengths}, row_lengths={"tokens": 256}, max_segments=8,
            window=window,
        )  # fmt: skip
        expected: list[list[int]] = []
        for start in range(0, count, window):
            rows = nemo_rl_first_fit(lengths[start : start + window].tolist(), 256, 8)
            expected += [[start + i for i in row] for row in rows]
        assert _as_rows(np.asarray(plan.records), np.asarray(plan.row_starts)) == expected

    def test_no_row_spans_two_windows(self, tmp_path: Path) -> None:
        order = np.random.default_rng(3).permutation(5000).astype(np.uint64)
        lengths = _lognormal(4, 5000, 128)
        plan = _plan_in(
            tmp_path, {"tokens": lengths}, row_lengths={"tokens": 128}, max_segments=16,
            window=500, order=order,
        )  # fmt: skip
        window_of = {int(record): position // 500 for position, record in enumerate(order)}
        for row in _as_rows(np.asarray(plan.records), np.asarray(plan.row_starts)):
            assert len({window_of[record] for record in row}) == 1

    def test_the_same_inputs_give_the_same_plan_files(self, tmp_path: Path) -> None:
        lengths = {"tokens": _lognormal(6, 2000, 200)}
        first = _plan_in(
            tmp_path / "a", lengths, row_lengths={"tokens": 200}, max_segments=4, window=300
        )
        second = _plan_in(
            tmp_path / "b", lengths, row_lengths={"tokens": 200}, max_segments=4, window=300
        )
        for name in ("records.bin", "row_starts.bin"):
            assert (first.directory / name).read_bytes() == (second.directory / name).read_bytes()


class TestThePlanFile:
    def test_open_reads_back_the_plan_written(self, tmp_path: Path) -> None:
        lengths = {"tokens": _lognormal(8, 1200, 100)}
        written = _plan_in(
            tmp_path, lengths, row_lengths={"tokens": 100}, max_segments=4, window=256
        )
        opened = PackPlan.open(tmp_path)
        np.testing.assert_array_equal(opened.records, written.records)
        np.testing.assert_array_equal(opened.row_starts, written.row_starts)
        assert (opened.length, opened.rows) == (1200, written.rows)
        assert opened.row_lengths == {"tokens": 100}
        assert (opened.max_segments, opened.window) == (4, 256)

    def test_take_gathers_the_records_of_the_rows_asked_for(self, tmp_path: Path) -> None:
        lengths = {"tokens": _lognormal(9, 600, 64)}
        plan = _plan_in(tmp_path, lengths, row_lengths={"tokens": 64}, max_segments=4, window=600)
        all_rows = _as_rows(np.asarray(plan.records), np.asarray(plan.row_starts))
        asked = np.array([5, 0, plan.rows - 1, 5], dtype=np.int64)
        members, starts = plan.take(asked)
        assert _as_rows(members, starts) == [all_rows[r] for r in asked.tolist()]
        assert starts[0] == 0

    def test_the_plan_holds_eight_bytes_a_record_and_a_row_boundary(self, tmp_path: Path) -> None:
        lengths = {"tokens": _lognormal(10, 4321, 128)}
        plan = _plan_in(tmp_path, lengths, row_lengths={"tokens": 128}, max_segments=8, window=97)
        size = sum((tmp_path / name).stat().st_size for name in ("records.bin", "row_starts.bin"))
        assert size == 8 * 4321 + 8 * (plan.rows + 1)

    def test_a_plan_of_another_version_is_refused_naming_both(self, tmp_path: Path) -> None:
        _plan_in(tmp_path, {"tokens": np.ones(10, np.int64)}, row_lengths={"tokens": 4},
                 max_segments=2, window=10)  # fmt: skip
        meta = json.loads((tmp_path / "plan.json").read_text())
        meta["version"] = PLANNER_VERSION + 1
        (tmp_path / "plan.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=rf"version {PLANNER_VERSION + 1}.*{PLANNER_VERSION}"):
            PackPlan.open(tmp_path)

    def test_a_directory_without_a_finished_plan_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="plan.json"):
            PackPlan.open(tmp_path)


class TestPlanningMemory:
    def _peak(self, tmp_path: Path, count: int, window: int) -> int:
        lengths = _lognormal(12, count, 64)
        order = np.arange(count, dtype=np.uint64)

        def records_at(start: int, stop: int) -> np.ndarray:
            return order[start:stop].copy()

        def lengths_of(records: np.ndarray) -> dict[str, np.ndarray]:
            return {"tokens": lengths[records.astype(np.int64)]}

        directory = tmp_path / str(count)
        tracemalloc.start()
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        plan_rows(
            length=count, records_at=records_at, lengths_of=lengths_of,
            row_lengths={"tokens": 64}, max_segments=8, window=window, directory=directory,
        )  # fmt: skip
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
        return peak

    def test_planning_holds_a_window_whatever_the_record_count(self, tmp_path: Path) -> None:
        """A table of every record (8 bytes each) would add 8 x 9N bytes from N to 10N."""
        small, large = 20_000, 200_000
        growth = self._peak(tmp_path, large, 512) - self._peak(tmp_path, small, 512)
        assert growth < 8 * small, f"planning memory grew {growth} bytes from {small} to {large}"


def test_planner_version_is_recorded_in_the_plan(tmp_path: Path) -> None:
    _plan_in(tmp_path, {"tokens": np.ones(5, np.int64)}, row_lengths={"tokens": 2},
             max_segments=2, window=5)  # fmt: skip
    meta = json.loads((tmp_path / "plan.json").read_text())
    assert meta["version"] == PLANNER_VERSION
    assert (meta["length"], meta["rows"]) == (5, 3)
