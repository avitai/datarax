"""Grain-style batched random-access indices, checked before a read sees them."""

from __future__ import annotations

import pytest

from datarax.sources._index_validation import validate_index_batch


def test_valid_indices_are_returned_as_integers_in_order() -> None:
    assert validate_index_batch([3, 0, 2], 4) == [3, 0, 2]


@pytest.mark.parametrize("indices", [[0, 3], [-1]], ids=["past-the-end", "negative"])
def test_indices_outside_the_records_are_refused(indices: list[int]) -> None:
    with pytest.raises(IndexError, match="out of range"):
        validate_index_batch(indices, 3)
