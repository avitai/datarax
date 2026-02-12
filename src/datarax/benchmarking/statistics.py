"""Statistical analysis for benchmark measurements.

Replaces inline statistics.mean/stdev in regression.py (fixes P4)
and _analyze_timing() from AdvancedProfiler.

Note: benchkit (tools/benchkit/src/benchkit/analysis.py) has a parallel
bootstrap_ci() using pure Python (no numpy/scipy). That duplication is
intentional — benchkit is zero-dependency by design.

Design ref: Section 6.2.3, Section 9.2 of the benchmark report.
"""

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

# --- Statistical thresholds (Section 6.2.3, 9.2 of benchmark report) ---

# Coefficient of variation threshold for measurement stability.
# CV < this value → "stable" measurement (low noise).
STABILITY_CV_THRESHOLD: float = 0.10

# Bootstrap confidence interval significance level.
# alpha=0.05 gives a 95% CI: [2.5th percentile, 97.5th percentile].
BOOTSTRAP_CI_ALPHA: float = 0.05

# Modified Z-score threshold for outlier detection (Iglewicz & Hoaglin).
OUTLIER_Z_THRESHOLD: float = 3.5

# MAD consistency constant: 1 / Φ⁻¹(3/4) ≈ 0.6745.
# Scales MAD to be a consistent estimator of σ for normal distributions.
_MAD_CONSISTENCY_CONSTANT: float = 0.6745


@dataclass
class StatisticalResult:
    """Summary statistics with confidence intervals.

    Attributes:
        mean: Arithmetic mean.
        median: Median value.
        std: Sample standard deviation (ddof=1).
        min: Minimum value.
        max: Maximum value.
        cv: Coefficient of variation (std / mean).
        ci_lower: 95% bootstrap CI lower bound.
        ci_upper: 95% bootstrap CI upper bound.
        n: Number of samples.
        is_stable: True when CV < 10% (Section 6.2.3 threshold).
    """

    mean: float
    median: float
    std: float
    min: float
    max: float
    cv: float
    ci_lower: float
    ci_upper: float
    n: int
    is_stable: bool


class StatisticalAnalyzer:
    """Statistical analysis for benchmark measurements.

    Provides:
    - Summary statistics with bootstrap confidence intervals
    - Welch's t-test for throughput comparison (unequal variances)
    - Mann-Whitney U for latency distribution comparison (non-parametric)
    - Modified Z-score outlier detection

    Args:
        bootstrap_resamples: Number of bootstrap resamples for CI computation.
        seed: Random seed for reproducible bootstrap sampling.
    """

    def __init__(self, bootstrap_resamples: int = 1000, seed: int = 42):
        """Initialize StatisticsCalculator.

        Args:
            bootstrap_resamples: Number of bootstrap resamples for CI computation.
            seed: Random seed for reproducible bootstrap sampling.
        """
        self._bootstrap_resamples = bootstrap_resamples
        self._rng = np.random.default_rng(seed)

    def summarize(self, samples: Sequence[float]) -> StatisticalResult:
        """Compute summary statistics with bootstrap CI.

        Args:
            samples: Sequence of measurement values (at least 1).

        Returns:
            StatisticalResult with all computed statistics.
        """
        arr = np.array(samples, dtype=np.float64)
        n = len(arr)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        cv = std / mean if mean != 0 else 0.0

        # Bootstrap CI
        if n <= 1:
            ci_lower, ci_upper = mean, mean
        else:
            bootstrap_means = np.array(
                [
                    float(np.mean(self._rng.choice(arr, size=n, replace=True)))
                    for _ in range(self._bootstrap_resamples)
                ]
            )
            lo = (BOOTSTRAP_CI_ALPHA / 2) * 100
            hi = (1 - BOOTSTRAP_CI_ALPHA / 2) * 100
            ci_lower = float(np.percentile(bootstrap_means, lo))
            ci_upper = float(np.percentile(bootstrap_means, hi))

        return StatisticalResult(
            mean=mean,
            median=median,
            std=std,
            min=min_val,
            max=max_val,
            cv=cv,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n=n,
            is_stable=cv < STABILITY_CV_THRESHOLD,
        )

    def welch_t_test(self, a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
        """Welch's t-test for unequal variances.

        Used for throughput comparison per Section 9.2.

        Args:
            a: Baseline measurements.
            b: Current measurements.

        Returns:
            Tuple of (t_statistic, p_value).
        """
        result = scipy_stats.ttest_ind(a, b, equal_var=False)
        return (float(result.statistic), float(result.pvalue))

    def mann_whitney_u(self, a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
        """Non-parametric test for latency distributions.

        Used for latency p99 comparison per Section 9.2.

        Args:
            a: Baseline measurements.
            b: Current measurements.

        Returns:
            Tuple of (u_statistic, p_value).
        """
        result = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return (float(result.statistic), float(result.pvalue))

    def detect_outliers(
        self, samples: Sequence[float], threshold: float = OUTLIER_Z_THRESHOLD
    ) -> list[int]:
        """Modified Z-score outlier detection.

        Uses median absolute deviation (MAD) instead of standard deviation
        for robustness against the outliers themselves.

        Args:
            samples: Sequence of values to check.
            threshold: Modified Z-score threshold (default 3.5).

        Returns:
            List of indices where outliers are detected.
        """
        arr = np.array(samples, dtype=np.float64)
        if len(arr) < 3:
            return []
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if mad == 0:
            return []
        modified_z = _MAD_CONSISTENCY_CONSTANT * (arr - median) / mad
        return [int(i) for i in np.where(np.abs(modified_z) > threshold)[0]]
