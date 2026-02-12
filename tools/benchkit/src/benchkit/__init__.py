"""benchkit — Benchmark analysis library with W&B dashboard integration."""

from benchkit.analysis import (
    aggregate_score,
    bootstrap_ci,
    detect_regressions,
    pareto_front,
    rank_table,
    scaling_fit,
    significance_test,
)
from benchkit.models import (
    Metric,
    MetricDef,
    MetricPriority,
    Point,
    RankEntry,
    Regression,
    Run,
    ScalingLaw,
    SignificanceResult,
    TrendPoint,
    TrendSeries,
    is_higher_better,
)
from benchkit.store import Store

__all__ = [
    # Data model
    "Metric",
    "MetricDef",
    "MetricPriority",
    "Point",
    "RankEntry",
    "Regression",
    "Run",
    "ScalingLaw",
    "SignificanceResult",
    "TrendPoint",
    "TrendSeries",
    # Store
    "Store",
    # Utilities
    "is_higher_better",
    # Analysis
    "aggregate_score",
    "bootstrap_ci",
    "detect_regressions",
    "pareto_front",
    "rank_table",
    "scaling_fit",
    "significance_test",
]
