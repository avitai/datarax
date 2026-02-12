"""Tests for StatisticalAnalyzer and StatisticalResult.

TDD tests written first per Section 6.2.3 / Section 9.2 of the benchmark report.
Verifies: summary statistics, bootstrap CI, Welch's t-test, Mann-Whitney U,
outlier detection, and edge cases (single sample, all-same values).
"""

import numpy as np
import pytest

from datarax.benchmarking.statistics import (
    STABILITY_CV_THRESHOLD,
    StatisticalAnalyzer,
    StatisticalResult,
)


class TestStatisticalResult:
    """Tests for StatisticalResult dataclass."""

    def test_creation_with_all_fields(self):
        result = StatisticalResult(
            mean=10.0,
            median=9.5,
            std=2.0,
            min=6.0,
            max=14.0,
            cv=0.2,
            ci_lower=9.0,
            ci_upper=11.0,
            n=20,
            is_stable=False,
        )
        assert result.mean == 10.0
        assert result.median == 9.5
        assert result.std == 2.0
        assert result.n == 20
        assert result.is_stable is False


class TestStatisticalAnalyzerSummarize:
    """Tests for StatisticalAnalyzer.summarize()."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer(seed=42)

    def test_computes_mean_correctly(self):
        samples = [10.0, 20.0, 30.0]
        result = self.analyzer.summarize(samples)
        assert result.mean == pytest.approx(20.0)

    def test_computes_median_correctly(self):
        samples = [1.0, 2.0, 10.0]
        result = self.analyzer.summarize(samples)
        assert result.median == pytest.approx(2.0)

    def test_computes_std_correctly(self):
        samples = [10.0, 10.0, 10.0]
        result = self.analyzer.summarize(samples)
        assert result.std == pytest.approx(0.0)

    def test_computes_min_max(self):
        samples = [5.0, 1.0, 9.0, 3.0]
        result = self.analyzer.summarize(samples)
        assert result.min == pytest.approx(1.0)
        assert result.max == pytest.approx(9.0)

    def test_computes_cv_correctly(self):
        """CV = std / mean."""
        samples = [10.0, 20.0, 30.0]
        result = self.analyzer.summarize(samples)
        expected_cv = result.std / result.mean
        assert result.cv == pytest.approx(expected_cv)

    def test_bootstrap_ci_contains_true_mean(self):
        """95% CI from 1000 resamples should contain the sample mean."""
        rng = np.random.default_rng(123)
        samples = rng.normal(100.0, 5.0, size=50).tolist()
        result = self.analyzer.summarize(samples)
        assert result.ci_lower <= result.mean <= result.ci_upper

    def test_bootstrap_ci_width_reasonable(self):
        """CI should be narrower than the data range for large samples."""
        rng = np.random.default_rng(456)
        samples = rng.normal(50.0, 2.0, size=100).tolist()
        result = self.analyzer.summarize(samples)
        ci_width = result.ci_upper - result.ci_lower
        data_range = result.max - result.min
        assert ci_width < data_range

    def test_is_stable_true_when_cv_below_threshold(self):
        """is_stable is True when CV < 0.10 (10% threshold per report)."""
        # Low variance data → CV < 10%
        samples = [100.0, 101.0, 99.0, 100.5, 99.5]
        result = self.analyzer.summarize(samples)
        assert result.cv < STABILITY_CV_THRESHOLD
        assert result.is_stable is True

    def test_is_stable_false_when_cv_above_threshold(self):
        """is_stable is False when CV >= 0.10."""
        # High variance data → CV > 10%
        samples = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = self.analyzer.summarize(samples)
        assert result.cv > STABILITY_CV_THRESHOLD
        assert result.is_stable is False

    def test_single_sample_edge_case(self):
        """Single sample: std=0, CI=(value, value), is_stable=True."""
        result = self.analyzer.summarize([42.0])
        assert result.mean == pytest.approx(42.0)
        assert result.std == pytest.approx(0.0)
        assert result.ci_lower == pytest.approx(42.0)
        assert result.ci_upper == pytest.approx(42.0)
        assert result.is_stable is True
        assert result.n == 1

    def test_all_same_values(self):
        """All-same values: cv=0, is_stable=True."""
        result = self.analyzer.summarize([7.0, 7.0, 7.0, 7.0])
        assert result.cv == pytest.approx(0.0)
        assert result.is_stable is True

    def test_n_field_correct(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.analyzer.summarize(samples)
        assert result.n == 5


class TestStatisticalAnalyzerTests:
    """Tests for Welch's t-test and Mann-Whitney U."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer(seed=42)

    def test_welch_t_test_significant_for_different_distributions(self):
        """Clearly different distributions should produce p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(100.0, 2.0, size=30).tolist()
        b = rng.normal(120.0, 2.0, size=30).tolist()
        t_stat, p_value = self.analyzer.welch_t_test(a, b)
        assert p_value < 0.05

    def test_welch_t_test_not_significant_for_same_distribution(self):
        """Same distribution should produce p > 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(100.0, 2.0, size=30).tolist()
        b = rng.normal(100.0, 2.0, size=30).tolist()
        t_stat, p_value = self.analyzer.welch_t_test(a, b)
        assert p_value > 0.05

    def test_welch_t_test_returns_tuple(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = self.analyzer.welch_t_test(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mann_whitney_significant_for_different_distributions(self):
        """Clearly different distributions should produce p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(10.0, 1.0, size=30).tolist()
        b = rng.normal(20.0, 1.0, size=30).tolist()
        u_stat, p_value = self.analyzer.mann_whitney_u(a, b)
        assert p_value < 0.05

    def test_mann_whitney_not_significant_for_same_distribution(self):
        """Same distribution should produce p > 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(10.0, 1.0, size=30).tolist()
        b = rng.normal(10.0, 1.0, size=30).tolist()
        u_stat, p_value = self.analyzer.mann_whitney_u(a, b)
        assert p_value > 0.05

    def test_mann_whitney_returns_tuple(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = self.analyzer.mann_whitney_u(a, b)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestStatisticalAnalyzerOutliers:
    """Tests for outlier detection."""

    def setup_method(self):
        self.analyzer = StatisticalAnalyzer(seed=42)

    def test_detects_known_outliers(self):
        """Known outlier (1000.0) should be detected."""
        samples = [10.0, 11.0, 10.5, 9.5, 10.2, 1000.0]
        outliers = self.analyzer.detect_outliers(samples)
        assert 5 in outliers  # index of 1000.0

    def test_no_outliers_in_uniform_data(self):
        """Uniform data should have no outliers."""
        samples = [10.0, 10.1, 9.9, 10.05, 9.95]
        outliers = self.analyzer.detect_outliers(samples)
        assert len(outliers) == 0

    def test_returns_indices(self):
        """Outlier detection returns list of integer indices."""
        samples = [1.0, 1.0, 1.0, 100.0]
        outliers = self.analyzer.detect_outliers(samples)
        assert all(isinstance(i, int) for i in outliers)

    def test_too_few_samples(self):
        """Fewer than 3 samples should return no outliers."""
        assert self.analyzer.detect_outliers([1.0, 2.0]) == []
        assert self.analyzer.detect_outliers([1.0]) == []

    def test_custom_threshold(self):
        """Lower threshold should detect more outliers."""
        samples = [10.0, 10.0, 10.0, 15.0, 10.0]
        strict = self.analyzer.detect_outliers(samples, threshold=2.0)
        lenient = self.analyzer.detect_outliers(samples, threshold=5.0)
        assert len(strict) >= len(lenient)
