"""Tests for performance gap detection engine.

RED phase: defines expected behavior for GapDetector.
"""

from __future__ import annotations

import time
from pathlib import Path

from benchmarks.tests.test_analysis.conftest import make_comparative_results


class TestGapDetector:
    """Tests for the performance gap detection and prioritization engine."""

    def _make_comparative(self, frameworks=None, scenarios=None):
        from benchmarks.runners.full_runner import ComparativeResults

        return ComparativeResults(
            results=make_comparative_results(frameworks, scenarios),
            environment={},
            platform="cpu",
            timestamp=time.time(),
        )

    def test_detects_gaps_above_threshold(self):
        """Gaps > 1.5x must be detected."""
        from benchmarks.analysis.gap_detection import GapDetector

        # Datarax at 500, SPDL at 1500 → 3.0x gap
        frameworks = {
            "Datarax": {"CV-1": 500.0},
            "SPDL": {"CV-1": 1500.0},
        }
        comparative = self._make_comparative(frameworks, ["CV-1"])

        detector = GapDetector(comparative, warning_threshold=1.5)
        gaps = detector.detect()

        assert len(gaps) >= 1
        assert gaps[0].scenario_id == "CV-1"
        assert gaps[0].gap_ratio >= 1.5

    def test_no_gaps_when_comparable(self):
        """Scenarios within 1.2x should not produce gaps."""
        from benchmarks.analysis.gap_detection import GapDetector

        frameworks = {
            "Datarax": {"CV-1": 1000.0},
            "Grain": {"CV-1": 1100.0},  # 1.1x — within threshold
        }
        comparative = self._make_comparative(frameworks, ["CV-1"])

        detector = GapDetector(comparative, warning_threshold=1.5)
        gaps = detector.detect()

        assert len(gaps) == 0

    def test_maps_gaps_to_priority(self):
        """Detected gaps must be mapped to P0-P5 priorities."""
        from benchmarks.analysis.gap_detection import GapDetector

        frameworks = {
            "Datarax": {"CV-1": 500.0, "NLP-1": 400.0},
            "SPDL": {"CV-1": 1500.0},
            "Grain": {"NLP-1": 900.0},
        }
        comparative = self._make_comparative(frameworks, ["CV-1", "NLP-1"])

        detector = GapDetector(comparative)
        gaps = detector.detect()

        for gap in gaps:
            assert gap.priority.startswith("P")
            assert gap.severity in ("warning", "action_required")

    def test_generates_backlog_markdown(self, tmp_path: Path):
        """generate_backlog must create a markdown file."""
        from benchmarks.analysis.gap_detection import GapDetector

        frameworks = {
            "Datarax": {"CV-1": 500.0},
            "SPDL": {"CV-1": 1500.0},
        }
        comparative = self._make_comparative(frameworks, ["CV-1"])

        detector = GapDetector(comparative)
        output_path = tmp_path / "optimization_backlog.md"
        detector.generate_backlog(output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "CV-1" in content
        assert "#" in content  # Markdown headings

    def test_handles_exclusive_scenarios(self):
        """Scenarios where only Datarax runs (no other frameworks) produce no gaps."""
        from benchmarks.analysis.gap_detection import GapDetector

        frameworks = {
            "Datarax": {"PC-2": 500.0},
            # No other frameworks for PC-2
        }
        comparative = self._make_comparative(frameworks, ["PC-2"])

        detector = GapDetector(comparative)
        gaps = detector.detect()

        # No other frameworks means no gaps
        assert len(gaps) == 0
