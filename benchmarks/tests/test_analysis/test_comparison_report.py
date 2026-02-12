"""Tests for comparative analysis report generator.

RED phase: defines expected behavior for ComparisonReportGenerator.
"""

from __future__ import annotations

import time

from benchmarks.tests.test_analysis.conftest import make_comparative_results


class TestComparisonReportGenerator:
    """Tests for the comparative analysis markdown report."""

    def _make_comparative(self, frameworks=None, scenarios=None):
        from benchmarks.runners.full_runner import ComparativeResults

        return ComparativeResults(
            results=make_comparative_results(frameworks, scenarios),
            environment={},
            platform="cpu",
            timestamp=time.time(),
        )

    def test_generates_markdown(self):
        """generate() must return valid markdown string."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        assert isinstance(md, str)
        assert len(md) > 100
        assert "#" in md  # Contains headings

    def test_includes_strengths_section(self):
        """Report must include a Strengths section."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        # Datarax leads in CV-1 (1200 vs 1100 best alternative)
        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        assert "Strength" in md or "strength" in md

    def test_includes_parity_section(self):
        """Report must include a Parity section."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        assert "Parity" in md or "parity" in md or "Comparable" in md

    def test_includes_gaps_section(self):
        """Report must include a Gaps section for underperforming scenarios."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        # Create scenario where Datarax trails significantly
        frameworks = {
            "Datarax": {"CV-1": 500.0, "NLP-1": 800.0},
            "SPDL": {"CV-1": 1500.0, "NLP-1": 700.0},
        }
        comparative = self._make_comparative(frameworks, ["CV-1", "NLP-1"])
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        assert "Gap" in md or "gap" in md or "Optimization" in md

    def test_includes_positioning_summary(self):
        """Report must include a positioning summary section."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        assert "Summary" in md or "summary" in md or "Position" in md

    def test_uses_real_numbers_not_placeholders(self):
        """Report must contain actual throughput numbers, not placeholders."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate()

        # Should NOT contain placeholder text
        assert "TODO" not in md
        assert "TBD" not in md
        assert "N/A" not in md
        # Should contain actual numbers
        assert any(char.isdigit() for char in md)

    def test_includes_chart_references(self):
        """Report must reference chart files when chart_dir is provided."""
        from benchmarks.analysis.comparison_report import ComparisonReportGenerator

        comparative = self._make_comparative()
        gen = ComparisonReportGenerator(comparative)
        md = gen.generate(chart_dir="/path/to/charts")

        assert "charts" in md.lower() or ".png" in md or ".svg" in md
