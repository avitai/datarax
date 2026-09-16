"""Every operator kind produces what it produced when the fixture was recorded.

The operator framework's redesign changes how a record's PRNG key reaches ``apply`` and how a
batch is vectorized. Neither is meant to change what an operator produces, and this module is what
makes that claim testable: ``scripts/generate_per_record_outputs.py`` recorded the output of every
operator kind, and each case here recomputes it and compares exactly.

A case fails loudly and by name, so a commit that changes the machinery learns which operator it
changed rather than that something, somewhere, moved.

The generator also records which entries are expected to change once a stochastic child receives
the record key through a ``probability=1.0`` wrapper. Those entries are pinned here exactly like
the others: the commit that makes the change regenerates the fixture, and that regeneration is
visible in its diff.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.scripts.script_loader import load_script


generator = load_script("generate_per_record_outputs")

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "operators" / "per_record_outputs.npz"
)


@pytest.fixture(scope="module")
def recorded() -> dict[str, np.ndarray]:
    """Return the recorded outputs, read once for the module."""
    with np.load(FIXTURE_PATH, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


@pytest.mark.parametrize(
    ("name", "build", "data"),
    generator.CASES,
    ids=[case[0] for case in generator.CASES],
)
def test_operator_output_is_unchanged(
    name: str,
    build: Any,
    data: dict[str, Any],
    recorded: dict[str, np.ndarray],
) -> None:
    """The operator produces exactly the arrays recorded for it."""
    operator = build()

    out_data, _ = operator._apply_on_raw(data, {}, None, generator.INDICES, generator.EPOCH)
    produced = generator.entries_for(name, out_data)

    assert produced, f"{name} produced no entries to compare"
    missing = sorted(set(produced) - set(recorded))
    assert missing == [], f"{name} produced entries the fixture does not hold: {missing}"
    for key, value in produced.items():
        np.testing.assert_array_equal(value, recorded[key], err_msg=f"{key} differs")


def test_pipeline_outputs_are_unchanged(recorded: dict[str, np.ndarray]) -> None:
    """Two epochs of a shuffled pipeline produce exactly the batches recorded for them."""
    produced = generator.pipeline_entries()

    assert produced, "the pipeline produced no entries to compare"
    for key, value in produced.items():
        np.testing.assert_array_equal(value, recorded[key], err_msg=f"{key} differs")


def test_fixture_covers_every_case(recorded: dict[str, np.ndarray]) -> None:
    """The fixture holds one recording per case the generator defines.

    Without this, deleting a case from the generator would leave its recording in the fixture
    unread, and the suite would report a coverage it no longer has.
    """
    assert list(recorded["case names"]) == [name for name, _, _ in generator.CASES]


def test_intended_change_entries_name_recorded_cases(recorded: dict[str, np.ndarray]) -> None:
    """Each entry marked as intended to change is a case the fixture actually records."""
    names = set(recorded["case names"].tolist())
    marked = set(recorded["intended change"].tolist())

    assert marked, "no entries are marked, so the marking would silently verify nothing"
    assert marked <= names, f"marked entries that are not cases: {sorted(marked - names)}"
