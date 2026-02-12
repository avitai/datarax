"""Tests for benchkit data model: MetricDef, Metric, Point, Run, and result types."""

from datetime import datetime

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
)


class TestMetricDef:
    def test_basic_construction(self):
        md = MetricDef(name="throughput", unit="elem/s", direction="higher")
        assert md.name == "throughput"
        assert md.unit == "elem/s"
        assert md.direction == "higher"
        assert md.group == ""
        assert md.priority == MetricPriority.SECONDARY
        assert md.description == ""

    def test_full_construction(self):
        md = MetricDef(
            name="latency_p50",
            unit="ms",
            direction="lower",
            group="Latency",
            priority=MetricPriority.PRIMARY,
            description="Median latency",
        )
        assert md.group == "Latency"
        assert md.priority == MetricPriority.PRIMARY
        assert md.description == "Median latency"

    def test_direction_values(self):
        for d in ("higher", "lower", "info"):
            md = MetricDef(name="x", unit="", direction=d)
            assert md.direction == d

    def test_serde_round_trip(self):
        md = MetricDef(
            name="throughput",
            unit="elem/s",
            direction="higher",
            group="Throughput",
            priority=MetricPriority.PRIMARY,
            description="Elements per second",
        )
        data = md.to_dict()
        restored = MetricDef.from_dict(data)
        assert restored == md

    def test_priority_enum_values(self):
        assert MetricPriority.PRIMARY == "primary"
        assert MetricPriority.SECONDARY == "secondary"


class TestMetric:
    def test_value_only(self):
        m = Metric(value=42.0)
        assert m.value == 42.0
        assert m.lower is None
        assert m.upper is None
        assert m.samples is None

    def test_with_ci_bounds(self):
        m = Metric(value=100.0, lower=95.0, upper=105.0)
        assert m.lower == 95.0
        assert m.upper == 105.0

    def test_with_samples(self):
        m = Metric(value=100.0, samples=[95.0, 100.0, 105.0])
        assert len(m.samples) == 3

    def test_serde_round_trip(self):
        m = Metric(value=42.0, lower=40.0, upper=44.0, samples=[40.0, 42.0, 44.0])
        data = m.to_dict()
        restored = Metric.from_dict(data)
        assert restored == m

    def test_serde_without_optionals(self):
        m = Metric(value=42.0)
        data = m.to_dict()
        restored = Metric.from_dict(data)
        assert restored == m
        assert restored.lower is None


class TestPoint:
    def test_basic_construction(self):
        p = Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax"},
            metrics={"throughput": Metric(5000.0)},
        )
        assert p.name == "CV-1/small"
        assert p.scenario == "CV-1"
        assert p.tags["framework"] == "Datarax"
        assert p.metrics["throughput"].value == 5000.0

    def test_serde_round_trip(self):
        p = Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax", "variant": "small"},
            metrics={
                "throughput": Metric(5000.0),
                "latency_p50": Metric(12.0, lower=10.0, upper=14.0),
            },
        )
        data = p.to_dict()
        restored = Point.from_dict(data)
        assert restored == p


class TestRun:
    def test_minimal_construction(self):
        """Run requires only points — all other fields have defaults."""
        run = Run(points=[])
        assert run.points == []
        assert isinstance(run.id, str)
        assert len(run.id) == 12
        assert isinstance(run.timestamp, datetime)
        assert run.commit is None
        assert run.branch is None
        assert run.environment == {}
        assert run.metadata == {}
        assert run.metric_defs == {}

    def test_full_construction(self):
        now = datetime(2026, 2, 10, 12, 0, 0)
        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
            ],
            id="abc123def456",
            timestamp=now,
            commit="abc123",
            branch="main",
            environment={"cpu": "AMD Ryzen"},
            metadata={"runner": "full"},
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )
        assert run.id == "abc123def456"
        assert run.timestamp == now
        assert run.commit == "abc123"
        assert len(run.points) == 1

    def test_serde_round_trip(self):
        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4800.0)},
                ),
            ],
            commit="abc123",
            branch="main",
            environment={"cpu": "AMD Ryzen"},
            metric_defs={
                "throughput": MetricDef(
                    "throughput",
                    "elem/s",
                    "higher",
                    group="Throughput",
                    priority=MetricPriority.PRIMARY,
                ),
            },
        )
        data = run.to_dict()
        restored = Run.from_dict(data)
        assert restored.commit == run.commit
        assert restored.branch == run.branch
        assert len(restored.points) == 2
        assert restored.points[0].metrics["throughput"].value == 5000.0
        assert restored.metric_defs["throughput"].direction == "higher"

    def test_auto_generated_id_unique(self):
        r1 = Run(points=[])
        r2 = Run(points=[])
        assert r1.id != r2.id


class TestRegression:
    def test_construction(self):
        r = Regression(
            metric="throughput",
            point_name="CV-1/small",
            baseline_value=5000.0,
            current_value=4500.0,
            delta_pct=-10.0,
            direction="higher",
        )
        assert r.delta_pct == -10.0

    def test_serde_round_trip(self):
        r = Regression(
            metric="throughput",
            point_name="CV-1/small",
            baseline_value=5000.0,
            current_value=4500.0,
            delta_pct=-10.0,
            direction="higher",
        )
        data = r.to_dict()
        restored = Regression.from_dict(data)
        assert restored == r


class TestRankEntry:
    def test_construction(self):
        entry = RankEntry(
            label="Datarax",
            value=5000.0,
            rank=1,
            is_best=True,
            delta_from_best=0.0,
        )
        assert entry.is_best is True
        assert entry.rank == 1

    def test_serde_round_trip(self):
        entry = RankEntry(
            label="Datarax",
            value=5000.0,
            rank=1,
            is_best=True,
            delta_from_best=0.0,
        )
        data = entry.to_dict()
        restored = RankEntry.from_dict(data)
        assert restored == entry


class TestSignificanceResult:
    def test_construction(self):
        sr = SignificanceResult(
            p_value=0.03,
            statistic=2.5,
            effect_size=0.8,
            significant=True,
            method="wilcoxon",
        )
        assert sr.significant is True

    def test_serde_round_trip(self):
        sr = SignificanceResult(
            p_value=0.03,
            statistic=2.5,
            effect_size=0.8,
            significant=True,
            method="wilcoxon",
        )
        data = sr.to_dict()
        restored = SignificanceResult.from_dict(data)
        assert restored == sr


class TestScalingLaw:
    def test_construction(self):
        sl = ScalingLaw(
            coefficient=1.5,
            exponent=0.9,
            r_squared=0.98,
            complexity="O(n)",
        )
        assert sl.r_squared == 0.98

    def test_serde_round_trip(self):
        sl = ScalingLaw(
            coefficient=1.5,
            exponent=0.9,
            r_squared=0.98,
            complexity="O(n)",
        )
        data = sl.to_dict()
        restored = ScalingLaw.from_dict(data)
        assert restored == sl


class TestTrendPoint:
    def test_construction(self):
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10, 12, 0, 0),
            value=5000.0,
            commit="def456",
        )
        assert tp.value == 5000.0
        assert tp.commit == "def456"

    def test_optional_fields(self):
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10),
            value=5000.0,
        )
        assert tp.commit is None
        assert tp.lower is None
        assert tp.upper is None

    def test_serde_round_trip(self):
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10, 12, 0, 0),
            value=5000.0,
            commit="def456",
            lower=4800.0,
            upper=5200.0,
        )
        data = tp.to_dict()
        restored = TrendPoint.from_dict(data)
        assert restored == tp

    def test_serde_without_optionals(self):
        tp = TrendPoint(
            run_id="abc123",
            timestamp=datetime(2026, 2, 10),
            value=5000.0,
        )
        data = tp.to_dict()
        restored = TrendPoint.from_dict(data)
        assert restored == tp
        assert "commit" not in data
        assert "lower" not in data


class TestTrendSeries:
    def test_construction(self):
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            points=[
                TrendPoint("r1", datetime(2026, 2, 1), 4800.0),
                TrendPoint("r2", datetime(2026, 2, 2), 5000.0),
            ],
        )
        assert ts.metric == "throughput"
        assert len(ts.points) == 2

    def test_serde_round_trip(self):
        ts = TrendSeries(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            points=[
                TrendPoint("r1", datetime(2026, 2, 1), 4800.0, commit="aaa"),
                TrendPoint("r2", datetime(2026, 2, 2), 5000.0, commit="bbb"),
            ],
        )
        data = ts.to_dict()
        restored = TrendSeries.from_dict(data)
        assert restored == ts

    def test_empty_points(self):
        ts = TrendSeries(
            metric="latency",
            point_name="NLP-1",
            tags={},
            points=[],
        )
        assert ts.points == []
        data = ts.to_dict()
        restored = TrendSeries.from_dict(data)
        assert restored == ts
