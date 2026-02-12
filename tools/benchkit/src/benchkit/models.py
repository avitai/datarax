"""benchkit data model: MetricDef, Metric, Point, Run, and analysis result types.

All types are plain dataclasses with to_dict()/from_dict() for JSON serde.
No external dependencies — pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4


class MetricPriority(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


def is_higher_better(md: MetricDef | None) -> bool:
    """Whether higher values are better for this metric.

    Returns True for "higher" or unknown (None) metrics, False for "lower".
    "info" metrics return True by convention (no ranking semantics).
    """
    return md is None or md.direction != "lower"


@dataclass
class MetricDef:
    """How to interpret a metric — semantics that W&B doesn't track natively."""

    name: str
    unit: str
    direction: Literal["higher", "lower", "info"]
    group: str = ""
    priority: MetricPriority = MetricPriority.SECONDARY
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "group": self.group,
            "priority": self.priority.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricDef:
        """Deserialize from a dictionary."""
        return cls(
            name=data["name"],
            unit=data["unit"],
            direction=data["direction"],
            group=data.get("group", ""),
            priority=MetricPriority(data.get("priority", "secondary")),
            description=data.get("description", ""),
        )


@dataclass
class Metric:
    """Single metric value with optional distribution data."""

    value: float
    lower: float | None = None
    upper: float | None = None
    samples: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary, omitting None fields."""
        d: dict[str, Any] = {"value": self.value}
        if self.lower is not None:
            d["lower"] = self.lower
        if self.upper is not None:
            d["upper"] = self.upper
        if self.samples is not None:
            d["samples"] = self.samples
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Deserialize from a dictionary."""
        return cls(
            value=data["value"],
            lower=data.get("lower"),
            upper=data.get("upper"),
            samples=data.get("samples"),
        )


@dataclass
class Point:
    """One benchmark + one configuration."""

    name: str
    scenario: str
    tags: dict[str, str]
    metrics: dict[str, Metric]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "scenario": self.scenario,
            "tags": self.tags,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Point:
        """Deserialize from a dictionary, reconstructing nested Metric objects."""
        return cls(
            name=data["name"],
            scenario=data["scenario"],
            tags=data["tags"],
            metrics={k: Metric.from_dict(v) for k, v in data["metrics"].items()},
        )


@dataclass
class Run:
    """One execution of a benchmark suite."""

    points: list[Point]
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    commit: str | None = None
    branch: str | None = None
    environment: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metric_defs: dict[str, MetricDef] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "commit": self.commit,
            "branch": self.branch,
            "environment": self.environment,
            "metadata": self.metadata,
            "metric_defs": {k: v.to_dict() for k, v in self.metric_defs.items()},
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Deserialize from a dictionary, reconstructing nested objects."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            commit=data.get("commit"),
            branch=data.get("branch"),
            environment=data.get("environment", {}),
            metadata=data.get("metadata", {}),
            metric_defs={k: MetricDef.from_dict(v) for k, v in data.get("metric_defs", {}).items()},
            points=[Point.from_dict(p) for p in data["points"]],
        )


# --- Analysis result types ---


@dataclass
class Regression:
    """A detected performance regression."""

    metric: str
    point_name: str
    baseline_value: float
    current_value: float
    delta_pct: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric": self.metric,
            "point_name": self.point_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "delta_pct": self.delta_pct,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Regression:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class RankEntry:
    """One row in a ranking table."""

    label: str
    value: float
    rank: int
    is_best: bool
    delta_from_best: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "label": self.label,
            "value": self.value,
            "rank": self.rank,
            "is_best": self.is_best,
            "delta_from_best": self.delta_from_best,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RankEntry:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class SignificanceResult:
    """Result of a statistical significance test."""

    p_value: float
    statistic: float
    effect_size: float
    significant: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "p_value": self.p_value,
            "statistic": self.statistic,
            "effect_size": self.effect_size,
            "significant": self.significant,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SignificanceResult:
        """Deserialize from a dictionary."""
        return cls(**data)


@dataclass
class ScalingLaw:
    """Power-law fit: value = a * size^b."""

    coefficient: float
    exponent: float
    r_squared: float
    complexity: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "coefficient": self.coefficient,
            "exponent": self.exponent,
            "r_squared": self.r_squared,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalingLaw:
        """Deserialize from a dictionary."""
        return cls(**data)


# --- Trend tracking ---


@dataclass
class TrendPoint:
    """One data point in a time-series trend."""

    run_id: str
    timestamp: datetime
    value: float
    commit: str | None = None
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary, omitting None fields."""
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
        }
        if self.commit is not None:
            d["commit"] = self.commit
        if self.lower is not None:
            d["lower"] = self.lower
        if self.upper is not None:
            d["upper"] = self.upper
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendPoint:
        """Deserialize from a dictionary."""
        return cls(
            run_id=data["run_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            value=data["value"],
            commit=data.get("commit"),
            lower=data.get("lower"),
            upper=data.get("upper"),
        )


@dataclass
class TrendSeries:
    """Time-series trend for a single metric across multiple runs."""

    metric: str
    point_name: str
    tags: dict[str, str]
    points: list[TrendPoint]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric": self.metric,
            "point_name": self.point_name,
            "tags": self.tags,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrendSeries:
        """Deserialize from a dictionary, reconstructing nested TrendPoints."""
        return cls(
            metric=data["metric"],
            point_name=data["point_name"],
            tags=data["tags"],
            points=[TrendPoint.from_dict(p) for p in data["points"]],
        )
