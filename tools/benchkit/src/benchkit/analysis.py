"""Benchmark analysis: regression detection, bootstrap CI, ranking, and more.

Note: The engine layer (src/datarax/benchmarking/) has parallel implementations
of bootstrap CI (statistics.py, uses numpy) and regression detection
(regression.py, uses BenchmarkResult). This duplication is intentional —
benchkit's core functions are zero-dependency. Statistical tests use scipy
(optional, install with benchkit[stats]).
"""

from __future__ import annotations

import math
import random

from benchkit.models import (
    MetricDef,
    Point,
    RankEntry,
    Regression,
    Run,
    ScalingLaw,
    SignificanceResult,
    is_higher_better,
)


# --- Core analysis ---


def detect_regressions(
    run: Run,
    baseline: Run,
    threshold: float = 0.05,
) -> list[Regression]:
    """Flag metrics that degraded beyond threshold.

    Uses MetricDef.direction: 'higher' metrics regress when they decrease,
    'lower' metrics regress when they increase. 'info' metrics are skipped.
    """
    # Merge metric_defs from both runs (current takes precedence)
    metric_defs = {**baseline.metric_defs, **run.metric_defs}

    # Build lookup of baseline points by (name, tags) composite key.
    # Multiple frameworks can share a name (e.g., "CV-1/small"); tags disambiguate.
    def _point_key(p: Point) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (p.name, tuple(sorted(p.tags.items())))

    baseline_lookup: dict[tuple[str, tuple[tuple[str, str], ...]], Point] = {
        _point_key(p): p for p in baseline.points
    }

    regressions: list[Regression] = []
    for point in run.points:
        bp = baseline_lookup.get(_point_key(point))
        if bp is None:
            continue

        for metric_name, current_metric in point.metrics.items():
            md = metric_defs.get(metric_name)
            if md is None or md.direction == "info":
                continue

            baseline_metric = bp.metrics.get(metric_name)
            if baseline_metric is None:
                continue

            bv = baseline_metric.value
            cv = current_metric.value

            if bv == 0:
                continue

            delta_pct = ((cv - bv) / abs(bv)) * 100.0

            is_regression = False
            if md.direction == "higher" and delta_pct < -threshold * 100:
                is_regression = True
            elif md.direction == "lower" and delta_pct > threshold * 100:
                is_regression = True

            if is_regression:
                regressions.append(
                    Regression(
                        metric=metric_name,
                        point_name=point.name,
                        baseline_value=bv,
                        current_value=cv,
                        delta_pct=delta_pct,
                        direction=md.direction,
                    )
                )

    return regressions


def bootstrap_ci(
    samples: list[float],
    confidence: float = 0.95,
    n_boot: int = 1000,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval. Pure Python (no scipy/numpy)."""
    if not samples:
        raise ValueError("Cannot compute CI on empty samples")

    n = len(samples)
    if n == 1:
        return (samples[0], samples[0])

    # Bootstrap resampling
    boot_means: list[float] = []
    for _ in range(n_boot):
        resample = random.choices(samples, k=n)
        boot_means.append(sum(resample) / n)

    boot_means.sort()

    alpha = 1 - confidence
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1

    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))

    return (boot_means[lo_idx], boot_means[hi_idx])


def rank_table(
    run: Run,
    metric: str,
    group_by_tag: str = "framework",
) -> list[RankEntry]:
    """Rank entries by metric value. Uses MetricDef.direction for best detection."""
    md = run.metric_defs.get(metric)
    higher_is_better = is_higher_better(md)

    # Group by tag value, take the first point per group
    groups: dict[str, float] = {}
    for point in run.points:
        label = point.tags.get(group_by_tag, point.name)
        if label not in groups and metric in point.metrics:
            groups[label] = point.metrics[metric].value

    # Sort by value
    sorted_items = sorted(
        groups.items(),
        key=lambda x: x[1],
        reverse=higher_is_better,
    )

    if not sorted_items:
        return []

    best_value = sorted_items[0][1]
    entries: list[RankEntry] = []
    for rank, (label, value) in enumerate(sorted_items, start=1):
        if best_value != 0:
            delta = abs((value - best_value) / best_value) * 100.0
        else:
            delta = 0.0

        entries.append(
            RankEntry(
                label=label,
                value=value,
                rank=rank,
                is_best=(rank == 1),
                delta_from_best=delta,
            )
        )

    return entries


# --- Statistical analysis (optional scipy) ---


def significance_test(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
) -> SignificanceResult:
    """Wilcoxon signed-rank test for paired samples.

    Tests whether two related samples have the same distribution.
    Uses scipy.stats.wilcoxon when available, falls back to a pure-Python
    approximation for small samples.

    Args:
        a: First sample (e.g., baseline measurements).
        b: Second sample (e.g., current measurements). Must be same length as a.
        alpha: Significance threshold (default 0.05).

    Returns:
        SignificanceResult with p_value, statistic, effect_size (Cohen's d),
        significant flag, and method name.

    Raises:
        ValueError: If samples are empty or have different lengths.
    """
    if not a or not b:
        raise ValueError("Cannot test significance on empty samples")
    if len(a) != len(b):
        raise ValueError(f"Paired test requires equal lengths: len(a)={len(a)}, len(b)={len(b)}")

    n = len(a)

    # Cohen's d effect size
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a) / max(n - 1, 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / max(n - 1, 1)
    pooled_std = math.sqrt((var_a + var_b) / 2) if (var_a + var_b) > 0 else 0.0
    effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    try:
        from scipy.stats import wilcoxon

        # zero_method='wilcox' drops zero differences; alternative='two-sided'
        stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return SignificanceResult(
            p_value=float(p_value),
            statistic=float(stat),
            effect_size=effect_size,
            significant=bool(p_value < alpha),
            method="wilcoxon",
        )
    except ImportError:
        # Pure-Python fallback: sign test (less powerful but no dependencies)
        diffs = [ai - bi for ai, bi in zip(a, b)]
        non_zero = [d for d in diffs if d != 0.0]
        if not non_zero:
            return SignificanceResult(
                p_value=1.0,
                statistic=0.0,
                effect_size=effect_size,
                significant=False,
                method="wilcoxon",
            )
        positives = sum(1 for d in non_zero if d > 0)
        nn = len(non_zero)
        # Two-tailed sign test approximation
        # Under H0, positives ~ Binomial(nn, 0.5)
        k = min(positives, nn - positives)
        # P(X <= k) using binomial CDF approximation
        p_value = 0.0
        for i in range(k + 1):
            p_value += _binom_coeff(nn, i) * (0.5**nn)
        p_value *= 2.0  # Two-tailed
        p_value = min(p_value, 1.0)
        return SignificanceResult(
            p_value=p_value,
            statistic=float(positives),
            effect_size=effect_size,
            significant=p_value < alpha,
            method="wilcoxon",
        )


def _binom_coeff(n: int, k: int) -> int:
    """Binomial coefficient C(n, k). Pure Python."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def pareto_front(
    points: list[Point],
    x_metric: str,
    y_metric: str,
    *,
    metric_defs: dict[str, MetricDef] | None = None,
) -> list[Point]:
    """Identify Pareto-optimal points for two metrics.

    A point is Pareto-optimal if no other point is strictly better on both
    metrics. Uses MetricDef.direction to determine "better": higher is better
    for "higher" metrics, lower is better for "lower" metrics.

    Args:
        points: List of benchmark points to analyze.
        x_metric: First metric name.
        y_metric: Second metric name.
        metric_defs: Optional metric definitions for direction. If not provided,
            defaults to higher-is-better for both metrics.

    Returns:
        List of Pareto-optimal points (subset of input, same order).
    """
    if not points:
        return []

    defs = metric_defs or {}
    x_higher = is_higher_better(defs.get(x_metric))
    y_higher = is_higher_better(defs.get(y_metric))

    # Extract values; skip points missing either metric
    indexed: list[tuple[int, float, float]] = []
    for i, p in enumerate(points):
        if x_metric in p.metrics and y_metric in p.metrics:
            xv = p.metrics[x_metric].value
            yv = p.metrics[y_metric].value
            indexed.append((i, xv, yv))

    # Check dominance: point j dominates point i if j is strictly better on both
    front_indices: list[int] = []
    for i, xi, yi in indexed:
        dominated = False
        for j, xj, yj in indexed:
            if i == j:
                continue
            x_better = (xj > xi) if x_higher else (xj < xi)
            x_equal = xj == xi
            y_better = (yj > yi) if y_higher else (yj < yi)
            y_equal = yj == yi
            # j dominates i if j is at least as good on both AND strictly better on at least one
            if (x_better or x_equal) and (y_better or y_equal) and (x_better or y_better):
                dominated = True
                break
        if not dominated:
            front_indices.append(i)

    return [points[i] for i in front_indices]


def aggregate_score(run: Run, weights: dict[str, float]) -> dict[str, float]:
    """Weighted aggregate score across metrics.

    Normalizes each metric to [0, 1] range (best = 1.0, worst = 0.0),
    then computes a weighted sum. Uses MetricDef.direction for normalization:
    "higher" metrics normalize as (value - min) / (max - min),
    "lower" metrics normalize as (max - value) / (max - min).

    Args:
        run: Benchmark run with points and metric_defs.
        weights: {metric_name: weight} — weights are normalized to sum to 1.0.

    Returns:
        {framework_label: aggregate_score} where score is in [0, 1].
    """
    if not run.points:
        return {}

    # Collect per-framework metrics
    frameworks: dict[str, dict[str, float]] = {}
    for point in run.points:
        fw = point.tags.get("framework", point.name)
        if fw not in frameworks:
            frameworks[fw] = {}
        for metric_name in weights:
            if metric_name in point.metrics:
                frameworks[fw][metric_name] = point.metrics[metric_name].value

    if not frameworks:
        return {}

    # Find min/max per metric for normalization
    metric_ranges: dict[str, tuple[float, float]] = {}
    for metric_name in weights:
        values = [
            fw_metrics[metric_name]
            for fw_metrics in frameworks.values()
            if metric_name in fw_metrics
        ]
        if values:
            metric_ranges[metric_name] = (min(values), max(values))

    # Normalize weights to sum to 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        norm_weights = weights

    # Compute scores
    scores: dict[str, float] = {}
    for fw, fw_metrics in frameworks.items():
        score = 0.0
        for metric_name, weight in norm_weights.items():
            if metric_name not in fw_metrics or metric_name not in metric_ranges:
                continue
            val = fw_metrics[metric_name]
            mn, mx = metric_ranges[metric_name]
            if mx == mn:
                normalized = 1.0  # All values equal → all are "best"
            elif is_higher_better(run.metric_defs.get(metric_name)):
                normalized = (val - mn) / (mx - mn)
            else:
                normalized = (mx - val) / (mx - mn)
            score += weight * normalized
        scores[fw] = score

    return scores


def scaling_fit(sizes: list[float], values: list[float]) -> ScalingLaw:
    """Fit power-law: value = a * size^b using log-linear regression.

    Takes log of both sides: log(value) = log(a) + b * log(size),
    then fits a linear regression. Pure Python (no scipy/numpy needed).

    Args:
        sizes: Input sizes (e.g., batch sizes, dataset sizes).
        values: Measured values (e.g., throughput, latency).

    Returns:
        ScalingLaw with coefficient (a), exponent (b), r_squared, and
        complexity classification string.

    Raises:
        ValueError: If inputs are empty or have different lengths.
    """
    if not sizes or not values:
        raise ValueError("Cannot fit scaling law on empty data")
    if len(sizes) != len(values):
        raise ValueError(f"Mismatched lengths: len(sizes)={len(sizes)}, len(values)={len(values)}")

    # Filter out non-positive values (can't take log)
    pairs = [(s, v) for s, v in zip(sizes, values) if s > 0 and v > 0]
    if not pairs:
        return ScalingLaw(coefficient=0.0, exponent=0.0, r_squared=0.0, complexity="O(1)")

    log_sizes = [math.log(s) for s, _ in pairs]
    log_values = [math.log(v) for _, v in pairs]
    m = len(pairs)

    # Handle constant values (all values equal → exponent = 0)
    if all(v == log_values[0] for v in log_values):
        return ScalingLaw(
            coefficient=math.exp(log_values[0]),
            exponent=0.0,
            r_squared=1.0,
            complexity="O(1)",
        )

    # Linear regression: log_value = intercept + slope * log_size
    mean_x = sum(log_sizes) / m
    mean_y = sum(log_values) / m

    ss_xx = sum((x - mean_x) ** 2 for x in log_sizes)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_sizes, log_values))
    ss_yy = sum((y - mean_y) ** 2 for y in log_values)

    if ss_xx == 0:
        # All sizes are the same — can't determine scaling
        return ScalingLaw(
            coefficient=math.exp(mean_y),
            exponent=0.0,
            r_squared=0.0,
            complexity="O(1)",
        )

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # R-squared
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(log_sizes, log_values))
    r_squared = 1.0 - (ss_res / ss_yy) if ss_yy > 0 else 0.0

    # Classify complexity
    exponent = slope
    complexity = _classify_complexity(exponent)

    return ScalingLaw(
        coefficient=math.exp(intercept),
        exponent=round(exponent, 4),
        r_squared=round(r_squared, 6),
        complexity=complexity,
    )


_COMPLEXITY_TOLERANCE = 0.15
_COMPLEXITY_CLASSES: list[tuple[float, str]] = [
    (0.0, "O(1)"),
    (0.5, "O(sqrt(n))"),
    (1.0, "O(n)"),
    (1.5, "O(n^1.5)"),
    (2.0, "O(n^2)"),
    (3.0, "O(n^3)"),
]


def _classify_complexity(exponent: float) -> str:
    """Classify a power-law exponent into Big-O notation."""
    for target, label in _COMPLEXITY_CLASSES:
        if abs(exponent - target) < _COMPLEXITY_TOLERANCE:
            return label
    return f"O(n^{exponent:.1f})"
