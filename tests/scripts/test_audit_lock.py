"""Tests for scripts/audit_lock.py, which audits every extra the lockfile resolves."""

from __future__ import annotations

import tomllib
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script, REPO_ROOT


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return load_script("audit_lock")


def _vuln(identifier: str, *aliases: str) -> dict:
    return {"id": identifier, "fix_versions": [], "aliases": list(aliases), "description": ""}


def _report(*dependencies: tuple[str, list[dict]]) -> dict:
    return {
        "dependencies": [
            {"name": name, "version": "1.0", "vulns": vulns} for name, vulns in dependencies
        ],
        "fixes": [],
    }


class TestExtrasGroups:
    """Conflicting extras cannot be exported together, so the audit covers them in groups."""

    def test_extras_without_conflicts_form_one_group(self, audit: ModuleType) -> None:
        assert audit.extras_groups(["a", "b", "c"], []) == [["a", "b", "c"]]

    def test_conflicting_extras_never_share_a_group(self, audit: ModuleType) -> None:
        conflicts = [{"a", "b"}, {"a", "c"}]

        groups = audit.extras_groups(["a", "b", "c", "d"], conflicts)

        assert all(not pair <= set(group) for group in groups for pair in conflicts)
        assert set().union(*groups) == {"a", "b", "c", "d"}

    def test_an_extra_compatible_with_every_group_joins_each(self, audit: ModuleType) -> None:
        groups = audit.extras_groups(["a", "b", "d"], [{"a", "b"}])

        assert all("d" in group for group in groups)

    def test_the_repository_s_groups_cover_every_extra(self, audit: ModuleType) -> None:
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        extras = list(project["project"]["optional-dependencies"])
        conflicts = audit.declared_conflicts(project)

        groups = audit.extras_groups(extras, conflicts)

        assert len(groups) > 1
        assert set().union(*groups) == set(extras)
        assert all(not pair <= set(group) for group in groups for pair in conflicts)


class TestFindings:
    """An advisory fails the audit unless it is ignored, and an ignore must still match."""

    def test_an_unignored_advisory_is_a_finding(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1")]))

        assert audit.findings([report], {}) == [("pkg", "1.0", "PYSEC-1")]

    def test_an_advisory_ignored_by_its_id_or_an_alias_is_not(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1"), _vuln("PYSEC-2", "CVE-2")]))

        assert audit.findings([report], {"PYSEC-1": "why", "CVE-2": "why"}) == []

    def test_an_advisory_reported_twice_is_one_finding(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1"), _vuln("PYSEC-1")]))

        assert audit.findings([report, report], {}) == [("pkg", "1.0", "PYSEC-1")]

    def test_an_ignore_no_advisory_matches_is_stale(self, audit: ModuleType) -> None:
        report = _report(("pkg", [_vuln("PYSEC-1", "CVE-1")]))

        stale = audit.stale_ignores([report], {"CVE-1": "why", "PYSEC-9": "why"})

        assert stale == ["PYSEC-9"]

    def test_every_ignored_advisory_carries_its_reason(self, audit: ModuleType) -> None:
        assert audit.IGNORED
        assert all(reason.strip() for reason in audit.IGNORED.values())
