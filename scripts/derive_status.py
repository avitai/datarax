#!/usr/bin/env python3
"""Derive datarax's status from its tree and check it against the docs.

Two run modes::

    uv run python scripts/derive_status.py            # print the derived table
    uv run python scripts/derive_status.py --check     # exit 1 on any drift (CI)

The README and ``benchmarks/COVERAGE_MATRIX.md`` state how many benchmark scenarios
exist, how many peer frameworks the adapters cover, and how many scenarios each
adapter supports. Those numbers rot as adapters and scenarios change; this script
measures them from the adapter registry and the scenario modules and fails loudly
when a documented value disagrees with the measured one.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger("derive_status")

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_NAME = "datarax"
README_PATH = REPO_ROOT / "README.md"
MATRIX_PATH = REPO_ROOT / "benchmarks" / "COVERAGE_MATRIX.md"
# The two datarax measurement dimensions are not peers.
_DATARAX_ADAPTERS = ("Datarax", "Datarax-scan")
# The matrix names the two datarax adapters by their iteration path.
_MATRIX_ADAPTER_NAMES = {"Datarax": "Datarax (iter)", "Datarax-scan": "Datarax (scan)"}

# label -> (document, regex with ONE capture group pulling the asserted value).
ASSERTIONS: dict[str, tuple[Path, str]] = {
    "scenarios": (README_PATH, r"across (\d+) standardized scenarios"),
    "peer_frameworks": (README_PATH, r"against (\d+) peer frameworks"),
    "comparable_scenarios": (MATRIX_PATH, r"\*\*(\d+) of \d+ scenarios run on"),
    "datarax_only_scenarios": (MATRIX_PATH, r"## Datarax-exclusive scenarios \((\d+)\)"),
    "adapter_coverage": (MATRIX_PATH, r"## Per-adapter scenario coverage\n\n((?:.*\n)+?)\n"),
    "scenario_frameworks": (
        MATRIX_PATH,
        r"## Per-scenario framework counts\n\n```\n((?:.*\n)+?)```",
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class Metric:
    """A derived metric paired with whatever the docs assert for it."""

    label: str
    measured: str
    asserted: str | None

    @property
    def is_drifted(self) -> bool:
        """Whether an asserted value exists and disagrees with measurement."""
        return self.asserted is not None and self.asserted != self.measured


# Runs in a child interpreter: importing the adapter registry initialises JAX and every
# installed peer framework, which must not happen inside the checking process.
_COVERAGE_PROBE = """
import json
from collections import Counter
import benchmarks.adapters as adapters
from benchmarks.scenarios import discover_scenarios

scenarios = sorted({module.SCENARIO_ID for module in discover_scenarios()})
supported = {
    cls().name: sorted(s for s in cls._supported() if s in scenarios)
    for cls in adapters._ADAPTER_REGISTRY.values()
}
print(json.dumps({"scenarios": scenarios, "supported": supported}))
"""


@functools.cache
def _coverage_facts() -> dict[str, object]:
    """Measure scenarios and per-adapter support in a subprocess, once per run."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _COVERAGE_PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _scenarios() -> list[str]:
    return list(_coverage_facts()["scenarios"])  # type: ignore[arg-type]


def _supported() -> dict[str, list[str]]:
    return dict(_coverage_facts()["supported"])  # type: ignore[arg-type]


def measure_scenarios(_root: Path, _package: str) -> str:
    """Count the benchmark scenarios the scenario packages define."""
    return str(len(_scenarios()))


def measure_peer_frameworks(_root: Path, _package: str) -> str:
    """Count the registered adapters other than datarax's own two."""
    return str(sum(1 for name in _supported() if name not in _DATARAX_ADAPTERS))


def _frameworks_per_scenario() -> dict[str, int]:
    supported = _supported()
    return {sid: sum(1 for names in supported.values() if sid in names) for sid in _scenarios()}


def measure_comparable_scenarios(_root: Path, _package: str) -> str:
    """Count the scenarios at least three frameworks support."""
    return str(sum(1 for count in _frameworks_per_scenario().values() if count >= 3))


def measure_datarax_only_scenarios(_root: Path, _package: str) -> str:
    """Count the scenarios only datarax's adapters support."""
    supported = _supported()
    peers = {name for name in supported if name not in _DATARAX_ADAPTERS}
    return str(sum(1 for sid in _scenarios() if not any(sid in supported[name] for name in peers)))


def measure_adapter_coverage(_root: Path, _package: str) -> str:
    """Render each adapter's scenario count as the matrix table's first two columns."""
    rows = [
        f"| {_MATRIX_ADAPTER_NAMES.get(name, name)} | {len(scenarios)} |"
        for name, scenarios in _supported().items()
    ]
    return "\n".join(sorted(rows, key=_coverage_sort_key))


def _coverage_sort_key(row: str) -> tuple[int, str]:
    name, count = row.strip("| ").split(" | ")
    return (-int(count), name)


def measure_scenario_frameworks(_root: Path, _package: str) -> str:
    """Render the per-scenario framework counts as ``ID:count`` pairs."""
    return " ".join(f"{sid}:{count}" for sid, count in sorted(_frameworks_per_scenario().items()))


MEASUREMENTS: dict[str, Callable[[Path, str], str]] = {
    "scenarios": measure_scenarios,
    "peer_frameworks": measure_peer_frameworks,
    "comparable_scenarios": measure_comparable_scenarios,
    "datarax_only_scenarios": measure_datarax_only_scenarios,
    "adapter_coverage": measure_adapter_coverage,
    "scenario_frameworks": measure_scenario_frameworks,
}


def _normalise(label: str, text: str) -> str:
    """Bring a captured document fragment onto the measurement's rendering."""
    if label == "adapter_coverage":
        rows = [
            "| " + " | ".join(cell.strip() for cell in line.strip("|").split("|")[:2]) + " |"
            for line in text.splitlines()
            if line.startswith("| ") and not line.startswith("| Adapter") and "---" not in line
        ]
        return "\n".join(sorted(rows, key=_coverage_sort_key))
    if label == "scenario_frameworks":
        return " ".join(sorted(text.split()))
    return text.strip()


def _asserted_value(label: str, documents: dict[Path, str]) -> str | None:
    """Extract the asserted value for ``label`` from its document, if declared."""
    entry = ASSERTIONS.get(label)
    if entry is None:
        return None
    path, pattern = entry
    match = re.search(pattern, documents.get(path, ""))
    return _normalise(label, match.group(1)) if match else None


def collect_metrics(
    root: Path, package: str, documents: dict[Path, Path] | None = None
) -> list[Metric]:
    """Run every measurement and pair it with its asserted value (if any).

    Args:
        root: Repository root.
        package: Import name of the package under ``src/``.
        documents: Optional replacement paths per asserted document, for tests.

    Returns:
        One metric per measurement.
    """
    replacements = documents or {}
    texts = {
        path: replacements.get(path, path).read_text()
        for path in {entry[0] for entry in ASSERTIONS.values()}
        if replacements.get(path, path).is_file()
    }
    return [
        Metric(label=label, measured=measure(root, package), asserted=_asserted_value(label, texts))
        for label, measure in MEASUREMENTS.items()
    ]


def render_table(metrics: list[Metric]) -> str:
    """Format the metrics as a fixed-width table for human reading."""
    header = f"{'metric':<24} {'measured':<20} {'asserted':<20} drift"
    rows = [
        f"{m.label:<24} {_short(m.measured):<20} {_short(m.asserted or '—'):<20} "
        f"{'DRIFT' if m.is_drifted else 'ok'}"
        for m in metrics
    ]
    return "\n".join([header, "-" * len(header), *rows])


def _short(value: str) -> str:
    first = value.splitlines()[0] if value else value
    return first if len(first) <= 20 else first[:17] + "..."


def main() -> int:
    """Print the derived table; with ``--check``, exit non-zero on drift."""
    parser = argparse.ArgumentParser(description="Derive and verify repo status.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any asserted value disagrees with measurement",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    metrics = collect_metrics(REPO_ROOT, PACKAGE_NAME)
    print(render_table(metrics))

    drifted = [m for m in metrics if m.is_drifted]
    if args.check and drifted:
        for metric in drifted:
            logger.error(
                "drift: %s\n  asserted: %s\n  measured: %s",
                metric.label,
                metric.asserted,
                metric.measured,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
