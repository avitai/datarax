"""Abstract base class for benchmark result exporters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchkit.models import Run


class Exporter(ABC):
    """Base class for exporting benchmark results to external platforms."""

    @abstractmethod
    def export_run(self, run: Run) -> str:
        """Export a Run. Returns a URL or identifier."""

    @abstractmethod
    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis results (regressions, rankings, etc.)."""
