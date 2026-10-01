#!/usr/bin/env python3
"""Audit every extra the lockfile resolves, not only the ones the test suite installs.

uv resolves one lockfile across all extras, but extras declared as conflicting cannot be exported
together, so the extras are covered in groups that each hold no conflicting pair. Each group is
exported from the lock and audited by pip-audit with a fresh cache (a cached advisory database
can hide an advisory published since). The run fails on an advisory not in ``IGNORED``, and on an
entry of ``IGNORED`` that no advisory matches any more, so the list cannot outlive its reasons.

Usage:
    uv run python scripts/audit_lock.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

_TORCH = (
    "torch, benchmark extra: no lockfile change closes it. PYSEC-2026-139 has no fixed release; "
    "PYSEC-2025-194 is fixed in 2.13.0, above the deliberate <2.11 ceiling (2.11 wheels link "
    "libcudart.so.13 and would move the stack to CUDA 13). Revisit when that ceiling lifts."
)
_STREAMING = (
    "benchmark extra only, through mosaicml-streaming (the MosaicML peer adapter): its latest "
    "release, 0.13.0, requires transformers<5 and paramiko<4. Paramiko's advisory covers every "
    "release through 4.0.0. Revisit when mosaicml-streaming lifts those bounds."
)
_AUTOMATION = (
    "automation extra only (SkyPilot benchmark orchestration, a conflicting extra no other install "
    "resolves with): every SkyPilot release, 0.13.0 included, requires click<8.2; vastai-sdk 0.2.x "
    "pins aiohttp==3.10.1 and wheel 0.45.1 comes with it, while every vastai 1.x pins "
    "cryptography==49.0.0, which the cryptography>=50 floor refuses. Revisit with SkyPilot or vastai."
)

# Advisories that no lockfile change can close today, each with the reason.
IGNORED: dict[str, str] = {
    "PYSEC-2026-139": _TORCH,
    "PYSEC-2025-194": _TORCH,
    **dict.fromkeys(
        (
            "CVE-2026-80047",
            "PYSEC-2025-217",
            "PYSEC-2026-2288",
            "PYSEC-2026-2289",
            "PYSEC-2026-2290",
            "PYSEC-2026-3929",
            "PYSEC-2026-2858",
        ),
        _STREAMING,
    ),
    **dict.fromkeys(
        (
            *(f"PYSEC-2026-{n}" for n in (1097, 1099, 1100, 1101, *range(1103, 1110))),
            *(f"PYSEC-2026-{n}" for n in range(2094, 2114)),
            "PYSEC-2026-237",
            "PYSEC-2026-3545",
            "PYSEC-2026-3546",
            "PYSEC-2026-3547",
            "PYSEC-2026-2132",
            "CVE-2026-24049",
        ),
        _AUTOMATION,
    ),
}


def declared_conflicts(project: Mapping) -> list[set[str]]:
    """The sets of extras ``[tool.uv] conflicts`` declares unable to resolve together."""
    return [
        {item["extra"] for item in conflict if "extra" in item}
        for conflict in project.get("tool", {}).get("uv", {}).get("conflicts", [])
    ]


def extras_groups(extras: Sequence[str], conflicts: Sequence[set[str]]) -> list[list[str]]:
    """Groups of extras, none holding a conflicting pair, that together hold every extra.

    Each extra not yet covered seeds a group, which then takes every extra compatible with it,
    so an extra that conflicts with nothing is audited in every group's resolution.
    """
    groups: list[list[str]] = []
    for seed in extras:
        if any(seed in group for group in groups):
            continue
        group = [seed]
        for extra in extras:
            if extra not in group and not any(c <= {*group, extra} for c in conflicts):
                group.append(extra)
        groups.append(sorted(group, key=list(extras).index))
    return groups


def _advisories(reports: Iterable[Mapping]) -> set[tuple[str, str, str, frozenset[str]]]:
    return {
        (dependency["name"], dependency["version"], vuln["id"], frozenset(vuln["aliases"]))
        for report in reports
        for dependency in report["dependencies"]
        for vuln in dependency.get("vulns", [])
    }


def findings(reports: Iterable[Mapping], ignored: Mapping[str, str]) -> list[tuple[str, str, str]]:
    """Advisories in pip-audit JSON reports that ``ignored`` names by neither id nor alias."""
    return sorted(
        (name, version, identifier)
        for name, version, identifier, aliases in _advisories(reports)
        if identifier not in ignored and not aliases & ignored.keys()
    )


def stale_ignores(reports: Iterable[Mapping], ignored: Mapping[str, str]) -> list[str]:
    """Entries of ``ignored`` that no advisory in the reports matches by id or alias."""
    named = {
        name for *_, identifier, aliases in _advisories(reports) for name in (identifier, *aliases)
    }
    return sorted(identifier for identifier in ignored if identifier not in named)


def _audit_group(group: Sequence[str], work_dir: Path) -> Mapping:
    """Export one group of extras from the lock and return pip-audit's JSON report on it."""
    name = "-".join(group)
    requirements = work_dir / f"{name}.txt"
    report = work_dir / f"{name}.json"
    extras = [argument for extra in group for argument in ("--extra", extra)]
    subprocess.run(
        ["uv", "export", "--frozen", "--all-groups", "--no-emit-project", *extras,
         "--output-file", str(requirements)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )  # fmt: skip
    audited = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--requirement", str(requirements), "--disable-pip",
         "--no-deps", "--cache-dir", str(work_dir / "cache"), "--format", "json",
         "--output", str(report)],
        cwd=REPO_ROOT, check=False,
    )  # fmt: skip
    if audited.returncode not in (0, 1):  # 1: advisories found, which the caller judges
        raise subprocess.CalledProcessError(audited.returncode, audited.args)
    return json.loads(report.read_text())


def main() -> None:
    """Audit every group of extras and exit non-zero on a finding or a stale ignore."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    groups = extras_groups(
        list(project["project"]["optional-dependencies"]), declared_conflicts(project)
    )
    with tempfile.TemporaryDirectory() as work_dir:
        reports = [_audit_group(group, Path(work_dir)) for group in groups]
    for group in groups:
        print(f"audited extras: {', '.join(group)}")
    unignored = findings(reports, IGNORED)
    stale = stale_ignores(reports, IGNORED)
    for name, version, identifier in unignored:
        print(f"advisory: {name} {version} {identifier}")
    for identifier in stale:
        print(f"stale ignore (no advisory matches it): {identifier}")
    sys.exit(1 if unignored or stale else 0)


if __name__ == "__main__":
    main()
