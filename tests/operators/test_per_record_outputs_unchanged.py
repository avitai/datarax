"""Every operator kind produces what it produced when the fixture was recorded.

The operator framework's redesign changes how a record's PRNG key reaches ``apply`` and how a
batch is vectorized. Neither is meant to change what an operator produces, and this module is what
makes that claim testable: ``scripts/generate_per_record_outputs.py`` recorded the output of every
operator kind, and each case here recomputes it and compares.

A case fails loudly and by name, so a commit that changes the machinery learns which operator it
changed rather than that something, somewhere, moved.

The generator also records which entries are expected to change once a stochastic child receives
the record key through a ``probability=1.0`` wrapper. Those entries are pinned here like the
others: the commit that makes the change regenerates the fixture, and that regeneration is visible
in its diff.

What is compared exactly, and what is not. Record indices, labels, masks and the case and marker
names are integers or strings, and every one of them must match exactly: a change in which record
gets which augmentation shows up there immediately. Float entries are compared within float32
rounding, because a reduction reassociates differently on different architectures and the fixture
is a recording from one of them. Measured between x86-64 and arm64, those differences are half a
float32 ULP at unit scale, while a real change in behaviour differs by orders of magnitude.

The bound is relative, with an absolute floor scaled to each entry's own peak magnitude. A purely
element-wise relative bound collapses toward zero for an element near zero, and it carries no
meaning at all on a logarithmic quantity such as the loudness operator's decibels. It stays a
detector rather than an eraser: a thousand-ULP change to a recorded entry still fails, as do a NaN,
an infinity, a changed shape and a flipped sign.

An operator can only be recorded where its output is well conditioned, and the generator chooses
its inputs accordingly: ``scripts/generate_per_record_outputs.py`` records why the audio case is
broadband rather than a single tone.
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

# Eight float32 ULP at the entry's own scale. Measured against a one-ULP change to each case's
# input: the worst entry moves 2.24 ULP of its peak, so the bound keeps a factor of three, while a
# thousand-ULP change to any entry still fails.
FLOAT_RTOL = 8 * float(np.finfo(np.float32).eps)


def assert_entry_matches(key: str, produced: np.ndarray, recorded: np.ndarray) -> None:
    """Compare one recorded entry, exactly where exactness carries across architectures."""
    assert produced.dtype == recorded.dtype, f"{key}: dtype {produced.dtype} != {recorded.dtype}"
    assert produced.shape == recorded.shape, f"{key}: shape {produced.shape} != {recorded.shape}"
    if produced.dtype.kind == "f":
        # Scale the absolute floor to the entry's own peak. 17 of the 69 float entries hold exact
        # zeros -- the dropout masks, salt-and-pepper and poisson noise -- and an element-wise
        # relative bound gives an exact zero no tolerance whatever: measured on one machine, a
        # single input ULP already pushes 15 of 61 entries past eight ULP element-wise, the worst
        # to 3224, while against the peak that same case moves 2.24. A decibel field is
        # logarithmic, where an element-wise relative bound carries no meaning either.
        peak = float(np.abs(recorded).max(initial=0.0))
        np.testing.assert_allclose(
            produced,
            recorded,
            rtol=FLOAT_RTOL,
            atol=FLOAT_RTOL * peak,
            err_msg=f"{key} differs",
        )
    else:
        np.testing.assert_array_equal(produced, recorded, err_msg=f"{key} differs")


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
    """The operator produces the arrays recorded for it."""
    operator = build()

    out_data, _ = operator._apply_on_raw(data, {}, None, generator.INDICES, generator.EPOCH)
    produced = generator.entries_for(name, out_data)

    assert produced, f"{name} produced no entries to compare"
    missing = sorted(set(produced) - set(recorded))
    assert missing == [], f"{name} produced entries the fixture does not hold: {missing}"
    for key, value in produced.items():
        assert_entry_matches(key, value, recorded[key])


def test_pipeline_outputs_are_unchanged(recorded: dict[str, np.ndarray]) -> None:
    """Two epochs of a shuffled pipeline produce the batches recorded for them."""
    produced = generator.pipeline_entries()

    assert produced, "the pipeline produced no entries to compare"
    for key, value in produced.items():
        assert_entry_matches(key, value, recorded[key])


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
