"""The derived-status check fails when the README or the coverage matrix drifts."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.scripts.script_loader import load_script


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derive_status() -> ModuleType:
    return load_script("derive_status")


@pytest.fixture(scope="module")
def metrics(derive_status: ModuleType) -> dict[str, Any]:
    return {m.label: m for m in derive_status.collect_metrics(REPO_ROOT, "datarax")}


def test_every_documented_number_matches_the_tree(metrics: dict[str, Any]) -> None:
    drifted = [label for label, metric in metrics.items() if metric.is_drifted]
    assert drifted == []


def test_every_label_is_asserted_somewhere(metrics: dict[str, Any]) -> None:
    unasserted = [label for label, metric in metrics.items() if metric.asserted is None]
    assert unasserted == []


def test_measurements_are_consistent_with_each_other(metrics: dict[str, Any]) -> None:
    scenarios = int(metrics["scenarios"].measured)
    comparable = int(metrics["comparable_scenarios"].measured)
    datarax_only = int(metrics["datarax_only_scenarios"].measured)
    assert scenarios > 0
    assert comparable + datarax_only <= scenarios
    assert int(metrics["peer_frameworks"].measured) > 0
    per_scenario = dict(pair.split(":") for pair in metrics["scenario_frameworks"].measured.split())
    assert len(per_scenario) == scenarios


def test_a_stale_matrix_count_is_drift(derive_status: ModuleType, tmp_path: Path) -> None:
    matrix = derive_status.MATRIX_PATH.read_text()
    match = re.search(r"## Datarax-exclusive scenarios \((\d+)\)", matrix)
    assert match is not None
    stale = matrix.replace(
        match.group(0), f"## Datarax-exclusive scenarios ({int(match.group(1)) + 1})", 1
    )
    edited = tmp_path / "COVERAGE_MATRIX.md"
    edited.write_text(stale)

    metrics = derive_status.collect_metrics(
        REPO_ROOT, "datarax", documents={derive_status.MATRIX_PATH: edited}
    )

    assert [m.label for m in metrics if m.is_drifted] == ["datarax_only_scenarios"]


def test_a_stale_adapter_row_is_drift(derive_status: ModuleType, tmp_path: Path) -> None:
    matrix = derive_status.MATRIX_PATH.read_text()
    assert "| Google Grain | 25 |" in matrix
    edited = tmp_path / "COVERAGE_MATRIX.md"
    edited.write_text(matrix.replace("| Google Grain | 25 |", "| Google Grain | 26 |", 1))

    metrics = derive_status.collect_metrics(
        REPO_ROOT, "datarax", documents={derive_status.MATRIX_PATH: edited}
    )

    assert [m.label for m in metrics if m.is_drifted] == ["adapter_coverage"]
