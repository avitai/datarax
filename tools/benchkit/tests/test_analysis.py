"""Tests for benchkit analysis: regressions, bootstrap CI, ranking, and statistics."""

import pytest

from benchkit.analysis import (
    aggregate_score,
    bootstrap_ci,
    detect_regressions,
    pareto_front,
    rank_table,
    scaling_fit,
    significance_test,
)
from benchkit.models import Metric, MetricDef, Point, Run
from conftest import make_run


class TestDetectRegressions:
    def test_higher_is_better_regression_on_decrease(self):
        """Throughput dropped 12% → should flag regression."""
        baseline = make_run(
            {"Datarax": {"throughput": 5000.0}},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        current = make_run(
            {"Datarax": {"throughput": 4400.0}},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        assert len(regs) == 1
        assert regs[0].metric == "throughput"
        assert regs[0].delta_pct < 0  # Negative = degradation

    def test_higher_is_better_no_regression_on_improvement(self):
        """Throughput increased → no regression."""
        baseline = make_run(
            {"Datarax": {"throughput": 5000.0}},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        current = make_run(
            {"Datarax": {"throughput": 5500.0}},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        assert len(regs) == 0

    def test_lower_is_better_regression_on_increase(self):
        """Latency increased 20% → should flag regression."""
        baseline = make_run(
            {"Datarax": {"latency": 10.0}},
            metric_defs={
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        current = make_run(
            {"Datarax": {"latency": 12.0}},
            metric_defs={
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        assert len(regs) == 1
        assert regs[0].direction == "lower"

    def test_lower_is_better_no_regression_on_decrease(self):
        """Latency decreased → no regression."""
        baseline = make_run(
            {"Datarax": {"latency": 10.0}},
            metric_defs={
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        current = make_run(
            {"Datarax": {"latency": 8.0}},
            metric_defs={
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        assert len(regs) == 0

    def test_info_metrics_skipped(self):
        """direction='info' metrics are never flagged."""
        baseline = make_run(
            {"Datarax": {"timestamp": 1000.0}},
            metric_defs={
                "timestamp": MetricDef("timestamp", "s", "info"),
            },
        )
        current = make_run(
            {"Datarax": {"timestamp": 2000.0}},
            metric_defs={
                "timestamp": MetricDef("timestamp", "s", "info"),
            },
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        assert len(regs) == 0

    def test_threshold_sensitivity(self):
        """3% degradation: flagged at 2% threshold, not at 5%."""
        baseline = make_run(
            {"Datarax": {"throughput": 5000.0}},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        current = make_run(
            {"Datarax": {"throughput": 4850.0}},  # -3%
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        assert len(detect_regressions(current, baseline, threshold=0.02)) == 1
        assert len(detect_regressions(current, baseline, threshold=0.05)) == 0

    def test_multiple_metrics_multiple_points(self):
        """Multiple regressions detected across metrics and points."""
        defs = {
            "throughput": MetricDef("throughput", "elem/s", "higher"),
            "latency": MetricDef("latency", "ms", "lower"),
        }
        baseline = Run(
            points=[
                Point(
                    "CV-1/Datarax",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0), "latency": Metric(10.0)},
                ),
                Point(
                    "CV-1/Grain",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4800.0), "latency": Metric(12.0)},
                ),
            ],
            metric_defs=defs,
        )
        current = Run(
            points=[
                Point(
                    "CV-1/Datarax",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(4000.0), "latency": Metric(15.0)},
                ),
                Point(
                    "CV-1/Grain",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4700.0), "latency": Metric(12.5)},
                ),
            ],
            metric_defs=defs,
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        # Datarax throughput dropped 20%, latency increased 50%
        # Grain latency increased ~4% (below threshold)
        assert len(regs) >= 2

    def test_same_name_different_tags(self):
        """Points with same name but different tags are matched correctly.

        Real-world adapters use name='CV-1/small' for all frameworks,
        distinguishing them via tags={'framework': 'Datarax'} etc.
        """
        defs = {
            "throughput": MetricDef("throughput", "elem/s", "higher"),
        }
        baseline = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(20000.0)},
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(19000.0)},
                ),
            ],
            metric_defs=defs,
        )
        current = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(18500.0)},  # -7.5% regression
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(18800.0)},  # -1.05% (within threshold)
                ),
            ],
            metric_defs=defs,
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        # Should detect Datarax regression (7.5% > 5%) but not Grain (1.05% < 5%)
        assert len(regs) == 1
        assert regs[0].point_name == "CV-1/small"
        # Verify it matched the Datarax point, not Grain
        assert regs[0].baseline_value == 20000.0
        assert regs[0].current_value == 18500.0

    def test_missing_point_in_current_skipped(self):
        """Points in baseline but not in current are skipped, not crash."""
        baseline = make_run(
            {"Datarax": {"throughput": 5000.0}, "Grain": {"throughput": 4800.0}},
            metric_defs={"throughput": MetricDef("throughput", "elem/s", "higher")},
        )
        current = make_run(
            {"Datarax": {"throughput": 4000.0}},
            metric_defs={"throughput": MetricDef("throughput", "elem/s", "higher")},
        )
        regs = detect_regressions(current, baseline, threshold=0.05)
        # Should only flag Datarax, not crash on missing Grain
        assert len(regs) == 1
        assert regs[0].point_name == "CV-1/Datarax"


class TestBootstrapCI:
    def test_basic_ci(self):
        """Bootstrap CI returns (lower, upper) tuple."""
        samples = [10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5]
        lower, upper = bootstrap_ci(samples, confidence=0.95, n_boot=500)
        assert lower < upper
        # Mean is ~12.25, CI should contain it
        assert lower <= 12.25
        assert upper >= 12.25

    def test_tight_distribution(self):
        """Identical values → very tight CI."""
        samples = [100.0] * 20
        lower, upper = bootstrap_ci(samples, confidence=0.95, n_boot=500)
        assert abs(lower - 100.0) < 0.01
        assert abs(upper - 100.0) < 0.01

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], confidence=0.95)

    def test_single_sample(self):
        """Single sample → degenerate CI (lower == upper == value)."""
        lower, upper = bootstrap_ci([42.0], confidence=0.95)
        assert lower == 42.0
        assert upper == 42.0

    def test_confidence_levels(self):
        """Higher confidence → wider CI."""
        samples = list(range(100))
        low_90, high_90 = bootstrap_ci(samples, confidence=0.90, n_boot=1000)
        low_99, high_99 = bootstrap_ci(samples, confidence=0.99, n_boot=1000)
        width_90 = high_90 - low_90
        width_99 = high_99 - low_99
        assert width_99 >= width_90


class TestRankTable:
    def test_basic_ranking_higher_is_better(self):
        """Highest throughput gets rank 1."""
        run = Run(
            points=[
                Point(
                    "CV-1/Datarax",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
                Point(
                    "CV-1/Grain",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4800.0)},
                ),
                Point(
                    "CV-1/PyTorch",
                    scenario="CV-1",
                    tags={"framework": "PyTorch"},
                    metrics={"throughput": Metric(4500.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        ranks = rank_table(run, "throughput", group_by_tag="framework")
        assert ranks[0].label == "Datarax"
        assert ranks[0].rank == 1
        assert ranks[0].is_best is True
        assert ranks[0].delta_from_best == 0.0
        assert ranks[1].rank == 2
        assert ranks[2].rank == 3

    def test_ranking_lower_is_better(self):
        """Lowest latency gets rank 1."""
        run = Run(
            points=[
                Point(
                    "CV-1/Datarax",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"latency": Metric(10.0)},
                ),
                Point(
                    "CV-1/Grain",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"latency": Metric(14.0)},
                ),
            ],
            metric_defs={
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        ranks = rank_table(run, "latency", group_by_tag="framework")
        assert ranks[0].label == "Datarax"
        assert ranks[0].is_best is True
        assert ranks[1].label == "Grain"
        assert ranks[1].delta_from_best > 0

    def test_delta_from_best_calculation(self):
        """delta_from_best is percentage difference from the best value."""
        run = Run(
            points=[
                Point(
                    "CV-1/A",
                    scenario="CV-1",
                    tags={"framework": "A"},
                    metrics={"throughput": Metric(100.0)},
                ),
                Point(
                    "CV-1/B",
                    scenario="CV-1",
                    tags={"framework": "B"},
                    metrics={"throughput": Metric(80.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        ranks = rank_table(run, "throughput", group_by_tag="framework")
        assert ranks[1].delta_from_best == pytest.approx(20.0)  # 20% behind


class TestSignificanceTest:
    """Tests for Wilcoxon signed-rank significance test."""

    def test_identical_samples_not_significant(self):
        """Identical distributions should NOT be significant."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = significance_test(a, b)
        assert result.significant is False
        assert result.method == "wilcoxon"

    def test_clearly_different_samples_significant(self):
        """Very different distributions should be significant."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5]
        b = [50.0, 51.0, 52.0, 53.0, 54.0, 50.5, 51.5, 52.5]
        result = significance_test(a, b)
        assert result.significant is True
        assert result.p_value < 0.05

    def test_returns_significance_result_type(self):
        """Return type must be SignificanceResult."""
        from benchkit.models import SignificanceResult

        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [6.0, 7.0, 8.0, 9.0, 10.0]
        result = significance_test(a, b)
        assert isinstance(result, SignificanceResult)

    def test_effect_size_is_nonnegative(self):
        """Effect size (Cohen's d) should be non-negative."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [20.0, 21.0, 22.0, 23.0, 24.0]
        result = significance_test(a, b)
        assert result.effect_size >= 0.0

    def test_custom_alpha(self):
        """Custom alpha threshold changes significance determination."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [11.0, 12.0, 13.0, 14.0, 15.0]
        # With strict alpha, may not be significant
        result_strict = significance_test(a, b, alpha=0.001)
        # p_value should be the same regardless of alpha
        result_loose = significance_test(a, b, alpha=0.99)
        assert result_strict.p_value == result_loose.p_value
        # But significance determination should differ (or at least be consistent)
        if result_strict.significant:
            assert result_loose.significant  # If strict passes, loose must too

    def test_empty_raises(self):
        """Empty samples should raise ValueError."""
        with pytest.raises(ValueError):
            significance_test([], [])

    def test_unequal_lengths_raises(self):
        """Unequal sample lengths should raise ValueError (Wilcoxon is paired)."""
        with pytest.raises(ValueError):
            significance_test([1.0, 2.0], [3.0, 4.0, 5.0])

    def test_serde_round_trip(self):
        """Result should serialize and deserialize correctly."""
        a = [10.0, 11.0, 12.0, 13.0, 14.0]
        b = [20.0, 21.0, 22.0, 23.0, 24.0]
        result = significance_test(a, b)
        data = result.to_dict()
        from benchkit.models import SignificanceResult

        restored = SignificanceResult.from_dict(data)
        assert restored == result


class TestParetoFront:
    """Tests for Pareto front identification."""

    def _make_point(self, name, throughput, latency):
        return Point(
            name=name,
            scenario="S",
            tags={"framework": name},
            metrics={
                "throughput": Metric(throughput),
                "latency": Metric(latency),
            },
        )

    def test_single_point_is_pareto_optimal(self):
        """A single point is always Pareto-optimal."""
        p = self._make_point("A", 100.0, 10.0)
        result = pareto_front([p], "throughput", "latency")
        assert len(result) == 1
        assert result[0].name == "A"

    def test_dominated_point_excluded(self):
        """A point dominated on both metrics is excluded."""
        # A: throughput=100, latency=10 (best on both → Pareto)
        # B: throughput=50, latency=20 (worse on both → dominated)
        a = self._make_point("A", 100.0, 10.0)
        b = self._make_point("B", 50.0, 20.0)
        defs = {
            "throughput": MetricDef("throughput", "elem/s", "higher"),
            "latency": MetricDef("latency", "ms", "lower"),
        }
        result = pareto_front([a, b], "throughput", "latency", metric_defs=defs)
        assert len(result) == 1
        assert result[0].name == "A"

    def test_tradeoff_points_both_on_front(self):
        """Points with tradeoffs (better on one, worse on other) are both Pareto."""
        # A: high throughput, high latency
        # B: low throughput, low latency
        a = self._make_point("A", 100.0, 20.0)
        b = self._make_point("B", 50.0, 5.0)
        defs = {
            "throughput": MetricDef("throughput", "elem/s", "higher"),
            "latency": MetricDef("latency", "ms", "lower"),
        }
        result = pareto_front([a, b], "throughput", "latency", metric_defs=defs)
        names = {p.name for p in result}
        assert names == {"A", "B"}

    def test_three_points_mixed(self):
        """Three points: two on Pareto front, one dominated."""
        a = self._make_point("A", 100.0, 20.0)  # High tp, high lat
        b = self._make_point("B", 50.0, 5.0)  # Low tp, low lat
        c = self._make_point("C", 40.0, 25.0)  # Worse than A on both
        defs = {
            "throughput": MetricDef("throughput", "elem/s", "higher"),
            "latency": MetricDef("latency", "ms", "lower"),
        }
        result = pareto_front([a, b, c], "throughput", "latency", metric_defs=defs)
        names = {p.name for p in result}
        assert names == {"A", "B"}

    def test_empty_returns_empty(self):
        """Empty input returns empty list."""
        result = pareto_front([], "throughput", "latency")
        assert result == []

    def test_all_equal_all_pareto(self):
        """If all points have equal metrics, all are Pareto-optimal."""
        a = self._make_point("A", 50.0, 10.0)
        b = self._make_point("B", 50.0, 10.0)
        result = pareto_front([a, b], "throughput", "latency")
        assert len(result) == 2

    def test_default_direction_higher_is_better(self):
        """Without metric_defs, both metrics default to higher-is-better."""
        # A: (100, 10) — better on throughput, worse on latency
        # B: (50, 20)  — worse on throughput, better on latency
        a = self._make_point("A", 100.0, 10.0)
        b = self._make_point("B", 50.0, 20.0)
        # Without defs: both higher=better → A dominates on throughput, B on latency
        result = pareto_front([a, b], "throughput", "latency")
        names = {p.name for p in result}
        assert names == {"A", "B"}


class TestAggregateScore:
    """Tests for weighted aggregate scoring."""

    def test_single_metric_equals_raw_value(self):
        """With one metric weighted 1.0, score equals the metric value."""
        run = Run(
            points=[
                Point(
                    "CV-1/A",
                    scenario="CV-1",
                    tags={"framework": "A"},
                    metrics={"throughput": Metric(5000.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        scores = aggregate_score(run, {"throughput": 1.0})
        assert "A" in scores
        assert scores["A"] == pytest.approx(1.0)  # Normalized: best = 1.0

    def test_two_frameworks_ranked_correctly(self):
        """Higher aggregate score means better overall performance."""
        run = Run(
            points=[
                Point(
                    "S/A",
                    "S",
                    tags={"framework": "A"},
                    metrics={"throughput": Metric(100.0), "latency": Metric(10.0)},
                ),
                Point(
                    "S/B",
                    "S",
                    tags={"framework": "B"},
                    metrics={"throughput": Metric(80.0), "latency": Metric(20.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        scores = aggregate_score(run, {"throughput": 0.5, "latency": 0.5})
        # A is better on both → higher score
        assert scores["A"] > scores["B"]

    def test_weights_influence_ranking(self):
        """Changing weights can change which framework wins."""
        run = Run(
            points=[
                Point(
                    "S/A",
                    "S",
                    tags={"framework": "A"},
                    metrics={"throughput": Metric(100.0), "latency": Metric(50.0)},
                ),
                Point(
                    "S/B",
                    "S",
                    tags={"framework": "B"},
                    metrics={"throughput": Metric(50.0), "latency": Metric(5.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
                "latency": MetricDef("latency", "ms", "lower"),
            },
        )
        # Weight throughput heavily → A wins
        scores_tp = aggregate_score(run, {"throughput": 0.9, "latency": 0.1})
        assert scores_tp["A"] > scores_tp["B"]

        # Weight latency heavily → B wins
        scores_lat = aggregate_score(run, {"throughput": 0.1, "latency": 0.9})
        assert scores_lat["B"] > scores_lat["A"]

    def test_empty_run_returns_empty(self):
        """Run with no points returns empty dict."""
        run = Run(points=[])
        scores = aggregate_score(run, {"throughput": 1.0})
        assert scores == {}

    def test_scores_between_0_and_1(self):
        """All scores should be normalized between 0 and 1."""
        run = Run(
            points=[
                Point("S/A", "S", tags={"framework": "A"}, metrics={"throughput": Metric(100.0)}),
                Point("S/B", "S", tags={"framework": "B"}, metrics={"throughput": Metric(50.0)}),
                Point("S/C", "S", tags={"framework": "C"}, metrics={"throughput": Metric(75.0)}),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        scores = aggregate_score(run, {"throughput": 1.0})
        for score in scores.values():
            assert 0.0 <= score <= 1.0


class TestScalingFit:
    """Tests for power-law scaling fit."""

    def test_linear_scaling(self):
        """Verify value = 2*size gives exponent ~ 1.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        values = [s * 2.0 for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(1.0, abs=0.1)
        assert result.r_squared > 0.95
        assert result.complexity == "O(n)"

    def test_constant_scaling(self):
        """Verify value = constant gives exponent ~ 0.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0]
        values = [42.0] * 5
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(0.0, abs=0.2)
        assert result.complexity == "O(1)"

    def test_quadratic_scaling(self):
        """Verify value = size^2 gives exponent ~ 2.0."""
        sizes = [1.0, 2.0, 4.0, 8.0, 16.0]
        values = [s**2 for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(2.0, abs=0.1)
        assert result.r_squared > 0.95
        assert result.complexity == "O(n^2)"

    def test_sublinear_scaling(self):
        """Verify value = sqrt(size) gives exponent ~ 0.5."""
        import math

        sizes = [1.0, 4.0, 9.0, 16.0, 25.0, 36.0]
        values = [math.sqrt(s) for s in sizes]
        result = scaling_fit(sizes, values)
        assert result.exponent == pytest.approx(0.5, abs=0.1)

    def test_returns_scaling_law_type(self):
        """Return type must be ScalingLaw."""
        from benchkit.models import ScalingLaw

        sizes = [1.0, 2.0, 4.0]
        values = [1.0, 2.0, 4.0]
        result = scaling_fit(sizes, values)
        assert isinstance(result, ScalingLaw)

    def test_empty_raises(self):
        """Empty inputs should raise ValueError."""
        with pytest.raises(ValueError):
            scaling_fit([], [])

    def test_mismatched_lengths_raises(self):
        """Mismatched input lengths should raise ValueError."""
        with pytest.raises(ValueError):
            scaling_fit([1.0, 2.0], [1.0])

    def test_serde_round_trip(self):
        """Result should serialize and deserialize correctly."""
        sizes = [1.0, 2.0, 4.0, 8.0]
        values = [s * 3.0 for s in sizes]
        result = scaling_fit(sizes, values)
        data = result.to_dict()
        from benchkit.models import ScalingLaw

        restored = ScalingLaw.from_dict(data)
        assert restored == result
