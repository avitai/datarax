"""Packed rows: a unit of planned rows assembled from its records' segments (design D9).

Each packed field ``f`` becomes ``f`` (B, row length, ...) with the records' values side by side
and zeros after them, ``f_segment_ids`` numbering a row's records from 1 with 0 in unused cells,
``f_positions`` counting from 0 inside each record, and ``f_segment_indices`` (B, S, 2) holding the
records' identities in slot order with ``PADDING_INDEX`` in unused slots. Host NumPy in, host NumPy
out, as every host batch construction (design D11).
"""

from __future__ import annotations

import numpy as np
import pytest

from datarax.core.batch_ops import pack_rows
from datarax.core.element_batch import PADDING_INDEX
from datarax.core.index_words import to_words


def _ragged(records: list[list[int]] | list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Records as one values array and their int64 offsets."""
    arrays = [np.asarray(r) for r in records]
    lengths = [len(a) for a in arrays]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    values = np.concatenate(arrays) if arrays else np.zeros(0, np.int32)
    return values, offsets


def _pack(
    records: list[list[int]],
    identities: list[int],
    row_starts: list[int],
    *,
    row_length: int = 8,
    max_segments: int = 3,
) -> dict[str, np.ndarray]:
    values, offsets = _ragged([np.asarray(r, np.int32) for r in records])
    return pack_rows(
        {"tokens": values},
        {"tokens": offsets},
        to_words(np.asarray(identities, np.uint64)),
        np.asarray(row_starts, np.int64),
        row_lengths={"tokens": row_length},
        max_segments=max_segments,
    )


class TestOneField:
    # Two rows: [10, 11, 12 | 20] and [30, 31].
    RECORDS = [[10, 11, 12], [20], [30, 31]]
    IDS = [7, 2**40 + 3, 9]
    STARTS = [0, 2, 3]

    def test_values_side_by_side_then_zeros(self) -> None:
        out = _pack(self.RECORDS, self.IDS, self.STARTS)
        np.testing.assert_array_equal(
            out["tokens"],
            [[10, 11, 12, 20, 0, 0, 0, 0], [30, 31, 0, 0, 0, 0, 0, 0]],
        )

    def test_segment_ids_count_from_one_with_zero_in_unused_cells(self) -> None:
        out = _pack(self.RECORDS, self.IDS, self.STARTS)
        np.testing.assert_array_equal(
            out["tokens_segment_ids"], [[1, 1, 1, 2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]
        )

    def test_positions_restart_in_each_record(self) -> None:
        out = _pack(self.RECORDS, self.IDS, self.STARTS)
        np.testing.assert_array_equal(
            out["tokens_positions"], [[0, 1, 2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]]
        )

    def test_identities_in_slot_order_with_padding_index_in_unused_slots(self) -> None:
        out = _pack(self.RECORDS, self.IDS, self.STARTS)
        slots = out["tokens_segment_indices"]
        assert slots.shape == (2, 3, 2)
        np.testing.assert_array_equal(slots[0, 0], to_words(np.uint64(7)))
        np.testing.assert_array_equal(slots[0, 1], to_words(np.uint64(2**40 + 3)))
        np.testing.assert_array_equal(slots[1, 0], to_words(np.uint64(9)))
        for row, slot in ((0, 2), (1, 1), (1, 2)):
            np.testing.assert_array_equal(slots[row, slot], PADDING_INDEX)

    def test_host_numpy_at_the_conventional_dtypes(self) -> None:
        out = _pack(self.RECORDS, self.IDS, self.STARTS)
        assert all(isinstance(leaf, np.ndarray) for leaf in out.values())
        assert out["tokens"].dtype == np.int32
        assert out["tokens_segment_ids"].dtype == np.int32
        assert out["tokens_positions"].dtype == np.int32
        assert out["tokens_segment_indices"].dtype == np.uint32
        assert sorted(out) == [
            "tokens", "tokens_positions", "tokens_segment_ids", "tokens_segment_indices",
        ]  # fmt: skip

    def test_a_full_row_and_a_full_slot_count(self) -> None:
        out = _pack([[1, 2], [3, 4], [5, 6, 7, 8]], [0, 1, 2], [0, 3], row_length=8)
        np.testing.assert_array_equal(out["tokens"], [[1, 2, 3, 4, 5, 6, 7, 8]])
        np.testing.assert_array_equal(out["tokens_segment_ids"], [[1, 1, 2, 2, 3, 3, 3, 3]])

    def test_a_zero_length_record_has_a_slot_and_no_cells(self) -> None:
        out = _pack([[1, 2], [], [3]], [4, 5, 6], [0, 3])
        np.testing.assert_array_equal(out["tokens"], [[1, 2, 3, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(out["tokens_segment_ids"], [[1, 1, 3, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(out["tokens_positions"], [[0, 1, 0, 0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(out["tokens_segment_indices"][0, 1], to_words(np.uint64(5)))


class TestSeveralFieldsAndShapes:
    def test_each_field_fills_its_own_row_length(self) -> None:
        tokens, token_offsets = _ragged([np.array([1, 2, 3], np.int32), np.array([4], np.int32)])
        targets, target_offsets = _ragged([np.array([9], np.int32), np.array([8, 7], np.int32)])
        out = pack_rows(
            {"tokens": tokens, "targets": targets},
            {"tokens": token_offsets, "targets": target_offsets},
            to_words(np.array([0, 1], np.uint64)),
            np.array([0, 2], np.int64),
            row_lengths={"tokens": 6, "targets": 4},
            max_segments=2,
        )
        np.testing.assert_array_equal(out["tokens"], [[1, 2, 3, 4, 0, 0]])
        np.testing.assert_array_equal(out["targets"], [[9, 8, 7, 0]])
        np.testing.assert_array_equal(out["targets_segment_ids"], [[1, 2, 2, 0]])
        np.testing.assert_array_equal(out["tokens_segment_indices"], out["targets_segment_indices"])

    def test_a_record_with_trailing_axes_keeps_them(self) -> None:
        first = np.arange(6, dtype=np.float32).reshape(2, 3)
        second = np.full((1, 3), 9.0, np.float32)
        values, offsets = _ragged([first, second])
        out = pack_rows(
            {"atoms": values},
            {"atoms": offsets},
            to_words(np.array([3, 4], np.uint64)),
            np.array([0, 2], np.int64),
            row_lengths={"atoms": 4},
            max_segments=2,
        )
        assert out["atoms"].shape == (1, 4, 3)
        np.testing.assert_array_equal(out["atoms"][0, :2], first)
        np.testing.assert_array_equal(out["atoms"][0, 2], second[0])
        np.testing.assert_array_equal(out["atoms"][0, 3], np.zeros(3))
        np.testing.assert_array_equal(out["atoms_segment_ids"], [[1, 1, 2, 0]])


class TestRefusals:
    def test_a_row_longer_than_its_row_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"row 0.*'tokens'.*9.*row length 8"):
            _pack([[1] * 5, [2] * 4], [0, 1], [0, 2])

    def test_a_row_with_more_records_than_max_segments_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"row 0.*4 records.*max_segments 3"):
            _pack([[1], [2], [3], [4]], [0, 1, 2, 3], [0, 4])

    def test_offsets_that_disagree_with_the_identities_are_refused(self) -> None:
        values, offsets = _ragged([[1], [2]])
        with pytest.raises(ValueError, match="offsets"):
            pack_rows(
                {"tokens": values},
                {"tokens": offsets},
                to_words(np.array([0, 1, 2], np.uint64)),
                np.array([0, 3], np.int64),
                row_lengths={"tokens": 8},
                max_segments=3,
            )
