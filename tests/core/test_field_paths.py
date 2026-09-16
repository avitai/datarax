"""Dotted field paths address nested record fields without mutating the record."""

from __future__ import annotations

import pytest

from datarax.core.field_paths import get_field, set_field


def _record() -> dict:
    return {"audio": {"signal": 1, "rate": 16000}, "label": 3}


def test_get_and_set_follow_dotted_paths_without_mutating_the_record() -> None:
    record = _record()

    updated = set_field(record, "audio.signal", 2)

    assert get_field(record, "audio.signal") == 1
    assert get_field(updated, "audio.signal") == 2
    assert updated == {"audio": {"signal": 2, "rate": 16000}, "label": 3}
    assert set_field(record, "label", 4) == {"audio": {"signal": 1, "rate": 16000}, "label": 4}


def test_set_creates_a_missing_nested_field() -> None:
    assert set_field({"label": 3}, "audio.signal", 1) == {"label": 3, "audio": {"signal": 1}}


def test_get_names_the_missing_path() -> None:
    with pytest.raises(KeyError, match=r"audio\.gain"):
        get_field(_record(), "audio.gain")


def test_set_refuses_to_descend_into_a_non_dict() -> None:
    with pytest.raises(ValueError, match="label"):
        set_field(_record(), "label.value", 1)
