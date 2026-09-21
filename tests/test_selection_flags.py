"""The command-line flags that choose which categories of test a run collects.

Each category has a flag that selects it alone and one that drops it. The CI jobs rely on the
pairing: a job whose machine cannot hold a measurement steady drops that category, and the job
built for it selects it. A category with no dropping flag is collected by every run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.conftest import _apply_explicit_deselect_flags, _deselect_unselected_test_types


@dataclass
class _Item:
    """A collected test, carrying the markers its selection depends on."""

    name: str
    keywords: set[str] = field(default_factory=set)


class _Hook:
    """The deselection hook, recording what it was told."""

    def __init__(self) -> None:
        self.deselected: list[_Item] = []

    def pytest_deselected(self, items: list[_Item]) -> None:
        """Record the deselected items."""
        self.deselected.extend(items)


class _Config:
    """The parts of a pytest config the selection helpers use."""

    def __init__(self) -> None:
        self.hook = _Hook()


def _collected() -> list[_Item]:
    """One test of each category, plus one carrying no marker."""
    return [
        _Item("unit"),
        _Item("integration", {"integration"}),
        _Item("end_to_end", {"end_to_end"}),
        _Item("benchmark", {"benchmark"}),
    ]


def _names(items: list[_Item]) -> set[str]:
    return {item.name for item in items}


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("integration", {"integration"}),
        ("end_to_end", {"end_to_end"}),
        ("benchmark", {"benchmark"}),
    ],
)
def test_requesting_a_category_collects_only_that_category(
    requested: str, expected: set[str]
) -> None:
    """The job built for a category runs that category and nothing else."""
    items = _collected()

    _deselect_unselected_test_types(
        _Config(),
        items,
        run_integration=requested == "integration",
        run_end_to_end=requested == "end_to_end",
        run_benchmark=requested == "benchmark",
    )

    assert _names(items) == expected


def test_requesting_nothing_collects_everything() -> None:
    """Without a category flag the positive selectors take no items out."""
    items = _collected()

    _deselect_unselected_test_types(
        _Config(), items, run_integration=False, run_end_to_end=False, run_benchmark=False
    )

    assert _names(items) == {"unit", "integration", "end_to_end", "benchmark"}


@pytest.mark.parametrize(
    ("dropped", "expected"),
    [
        ("integration", {"unit", "end_to_end", "benchmark"}),
        ("end_to_end", {"unit", "integration", "benchmark"}),
        ("benchmark", {"unit", "integration", "end_to_end"}),
    ],
)
def test_dropping_a_category_leaves_the_others(dropped: str, expected: set[str]) -> None:
    """Every category can be dropped, so a job can exclude what its machine cannot measure."""
    items = _collected()

    _apply_explicit_deselect_flags(
        _Config(),
        items,
        skip_integration=dropped == "integration",
        skip_end_to_end=dropped == "end_to_end",
        skip_benchmark=dropped == "benchmark",
    )

    assert _names(items) == expected


def test_the_unit_gate_drops_every_category_that_needs_its_own_machine() -> None:
    """The flags the unit jobs pass leave the plain tests and nothing else.

    Integration and end-to-end tests have their own jobs, and benchmark tests assert
    wall-clock figures that a shared runner cannot hold steady.
    """
    items = _collected()

    _apply_explicit_deselect_flags(
        _Config(), items, skip_integration=True, skip_end_to_end=True, skip_benchmark=True
    )

    assert _names(items) == {"unit"}


def test_a_deselected_item_is_reported_once() -> None:
    """An item dropped by two flags at once is reported to pytest a single time."""
    config = _Config()
    both: Any = _Item("integration_benchmark", {"integration", "benchmark"})
    items = [both, _Item("unit")]

    _apply_explicit_deselect_flags(
        config, items, skip_integration=True, skip_end_to_end=False, skip_benchmark=True
    )

    assert _names(items) == {"unit"}
    assert [item.name for item in config.hook.deselected] == ["integration_benchmark"]
