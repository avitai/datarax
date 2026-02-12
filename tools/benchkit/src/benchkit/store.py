"""JSON-per-run file backend with baseline management.

Directory layout:
    benchmark-data/
    ├── runs/
    │   ├── 2026-02-10T09-05-00_abc123.json
    │   └── ...
    ├── baselines/
    │   └── main.json
    └── config.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from benchkit.models import MetricDef, Run, TrendPoint, TrendSeries


class Store:
    """JSON-per-run file backend with baseline management."""

    def __init__(self, path: Path | str) -> None:
        """Initialize Store.

        Args:
            path: Root directory for storing runs, baselines, and config.
        """
        self._path = Path(path)
        self._runs_dir = self._path / "runs"
        self._baselines_dir = self._path / "baselines"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._baselines_dir.mkdir(parents=True, exist_ok=True)

        self.metric_defs: dict[str, MetricDef] = {}
        self.wandb_project: str | None = None
        self.wandb_entity: str | None = None
        self._load_config()

    def _load_config(self) -> None:
        config_path = self._path / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            for name, md_data in data.get("metric_defs", {}).items():
                self.metric_defs[name] = MetricDef.from_dict(md_data)
            self.wandb_project = data.get("wandb_project")
            self.wandb_entity = data.get("wandb_entity")

    def _run_path(self, run_id: str) -> Path:
        return self._runs_dir / f"{run_id}.json"

    def save(self, run: Run) -> Path:
        """Save a run as JSON. Returns path to saved file."""
        path = self._run_path(run.id)
        path.write_text(json.dumps(run.to_dict(), indent=2))
        return path

    def load(self, run_id: str) -> Run:
        """Load a run by ID."""
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return Run.from_dict(json.loads(path.read_text()))

    def list_runs(self, branch: str | None = None) -> list[Run]:
        """List all runs, optionally filtered by branch. Sorted by timestamp desc."""
        runs: list[Run] = []
        for p in self._runs_dir.glob("*.json"):
            run = Run.from_dict(json.loads(p.read_text()))
            if branch is None or run.branch == branch:
                runs.append(run)
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs

    def latest(self) -> Run:
        """Load the most recent run."""
        runs = self.list_runs()
        if not runs:
            raise FileNotFoundError("No runs in store")
        return runs[0]

    def query(self, **tags: str) -> list[Run]:
        """Find runs where any point matches all given tag filters."""
        results: list[Run] = []
        for p in self._runs_dir.glob("*.json"):
            run = Run.from_dict(json.loads(p.read_text()))
            for point in run.points:
                if all(point.tags.get(k) == v for k, v in tags.items()):
                    results.append(run)
                    break
        return results

    def set_baseline(self, run_id: str) -> None:
        """Copy a run to baselines/main.json."""
        src = self._run_path(run_id)
        if not src.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        shutil.copy2(src, self._baselines_dir / "main.json")

    def get_baseline(self) -> Run | None:
        """Load baseline, or None if not set."""
        path = self._baselines_dir / "main.json"
        if not path.exists():
            return None
        return Run.from_dict(json.loads(path.read_text()))

    def ingest(self, path: Path, format: str = "auto") -> Run:
        """Import results from external JSON file and save to store."""
        data = json.loads(Path(path).read_text())
        run = Run.from_dict(data)
        self.save(run)
        return run

    def extract_trend(
        self,
        metric: str,
        point_name: str,
        tags: dict[str, str],
        *,
        n_runs: int | None = None,
    ) -> TrendSeries:
        """Extract time-series trend for a metric across stored runs.

        Scans all runs for points matching (point_name, tags), extracts
        the named metric, and returns a TrendSeries ordered oldest-first.

        Args:
            metric: Metric name to track (e.g., "throughput").
            point_name: Point name to match (e.g., "CV-1/small").
            tags: Tags that must all match on the point.
            n_runs: If set, return only the N most recent data points.

        Returns:
            TrendSeries with one TrendPoint per matching run.
        """
        runs = self.list_runs()  # Sorted desc by timestamp

        trend_points: list[TrendPoint] = []
        for run in runs:
            for point in run.points:
                if point.name != point_name:
                    continue
                if not all(point.tags.get(k) == v for k, v in tags.items()):
                    continue
                if metric not in point.metrics:
                    continue
                m = point.metrics[metric]
                trend_points.append(
                    TrendPoint(
                        run_id=run.id,
                        timestamp=run.timestamp,
                        value=m.value,
                        commit=run.commit,
                        lower=m.lower,
                        upper=m.upper,
                    )
                )
                break  # One match per run

        # Sort oldest → newest (list_runs gives newest first)
        trend_points.sort(key=lambda tp: tp.timestamp)

        # Apply n_runs limit (most recent N)
        if n_runs is not None and len(trend_points) > n_runs:
            trend_points = trend_points[-n_runs:]

        return TrendSeries(
            metric=metric,
            point_name=point_name,
            tags=tags,
            points=trend_points,
        )
