"""Tests for benchkit Store: JSON backend with baseline management."""

import json
from datetime import datetime

import pytest

from benchkit.models import Metric, MetricPriority, Point, Run
from benchkit.store import Store


@pytest.fixture
def tmp_store(tmp_path):
    """Create a Store rooted at a temporary directory."""
    return Store(tmp_path / "benchmark-data")


class TestSaveLoad:
    def test_save_creates_file(self, tmp_store, sample_run):
        path = tmp_store.save(sample_run)
        assert path.exists()
        assert path.suffix == ".json"

    def test_load_round_trip(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        loaded = tmp_store.load(sample_run.id)
        assert loaded.id == sample_run.id
        assert loaded.commit == sample_run.commit
        assert len(loaded.points) == 2
        assert loaded.points[0].metrics["throughput"].value == 5000.0

    def test_load_preserves_metric_defs(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        loaded = tmp_store.load(sample_run.id)
        assert "throughput" in loaded.metric_defs
        assert loaded.metric_defs["throughput"].direction == "higher"
        assert loaded.metric_defs["throughput"].priority == MetricPriority.PRIMARY

    def test_load_nonexistent_raises(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.load("nonexistent_id")


class TestListRuns:
    def test_list_empty_store(self, tmp_store):
        runs = tmp_store.list_runs()
        assert runs == []

    def test_list_returns_all_runs(self, tmp_store):
        r1 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        r2 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 2, 12, 0, 0),
        )
        tmp_store.save(r1)
        tmp_store.save(r2)
        runs = tmp_store.list_runs()
        assert len(runs) == 2

    def test_list_sorted_desc_by_timestamp(self, tmp_store):
        r1 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        r2 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 2, 12, 0, 0),
        )
        tmp_store.save(r1)
        tmp_store.save(r2)
        runs = tmp_store.list_runs()
        assert runs[0].timestamp > runs[1].timestamp

    def test_list_filter_by_branch(self, tmp_store):
        r1 = Run(points=[], branch="main")
        r2 = Run(points=[], branch="feature")
        tmp_store.save(r1)
        tmp_store.save(r2)
        runs = tmp_store.list_runs(branch="main")
        assert len(runs) == 1
        assert runs[0].branch == "main"


class TestLatest:
    def test_latest_returns_most_recent(self, tmp_store):
        r1 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )
        r2 = Run(
            points=[],
            branch="main",
            timestamp=datetime(2026, 1, 2, 12, 0, 0),
        )
        tmp_store.save(r1)
        tmp_store.save(r2)
        latest = tmp_store.latest()
        assert latest.id == r2.id

    def test_latest_empty_store_raises(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.latest()


class TestQuery:
    def test_query_by_tag(self, tmp_store):
        r1 = Run(
            points=[
                Point(
                    "X",
                    scenario="S",
                    tags={"framework": "Datarax"},
                    metrics={"x": Metric(1.0)},
                ),
            ],
        )
        r2 = Run(
            points=[
                Point(
                    "Y",
                    scenario="S",
                    tags={"framework": "Grain"},
                    metrics={"x": Metric(1.0)},
                ),
            ],
        )
        tmp_store.save(r1)
        tmp_store.save(r2)

        results = tmp_store.query(framework="Datarax")
        assert len(results) == 1
        assert results[0].id == r1.id


class TestBaseline:
    def test_set_and_get_baseline(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        tmp_store.set_baseline(sample_run.id)
        baseline = tmp_store.get_baseline()
        assert baseline is not None
        assert baseline.id == sample_run.id

    def test_get_baseline_when_none(self, tmp_store):
        assert tmp_store.get_baseline() is None

    def test_set_baseline_nonexistent_raises(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.set_baseline("nonexistent_id")


class TestIngest:
    def test_ingest_benchkit_json(self, tmp_store, sample_run):
        """Ingest a benchkit-format JSON file."""
        json_path = tmp_store._path / "external.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(sample_run.to_dict(), indent=2))

        ingested = tmp_store.ingest(json_path)
        assert ingested.id == sample_run.id
        assert len(ingested.points) == 2

        # Verify it's also saved in the store
        loaded = tmp_store.load(ingested.id)
        assert loaded.id == ingested.id


class TestConfigLoading:
    def test_config_loads_metric_defs(self, tmp_path):
        """Store loads MetricDefs from config.json if present."""
        store_path = tmp_path / "benchmark-data"
        store_path.mkdir(parents=True)
        config = {
            "title": "Test Benchmarks",
            "metric_defs": {
                "throughput": {
                    "name": "throughput",
                    "unit": "elem/s",
                    "direction": "higher",
                    "group": "Throughput",
                    "priority": "primary",
                },
            },
        }
        (store_path / "config.json").write_text(json.dumps(config))

        store = Store(store_path)
        assert "throughput" in store.metric_defs
        assert store.metric_defs["throughput"].direction == "higher"
        assert store.metric_defs["throughput"].priority == MetricPriority.PRIMARY

    def test_no_config_empty_metric_defs(self, tmp_path):
        """Store works fine without config.json."""
        store = Store(tmp_path / "benchmark-data")
        assert store.metric_defs == {}

    def test_config_loads_wandb_settings(self, tmp_path):
        """Store loads wandb_project and wandb_entity from config.json."""
        store_path = tmp_path / "benchmark-data"
        store_path.mkdir(parents=True)
        config = {
            "wandb_project": "my-benchmarks",
            "wandb_entity": "my-team",
        }
        (store_path / "config.json").write_text(json.dumps(config))

        store = Store(store_path)
        assert store.wandb_project == "my-benchmarks"
        assert store.wandb_entity == "my-team"

    def test_no_config_wandb_defaults(self, tmp_path):
        """Without config.json, wandb settings are None."""
        store = Store(tmp_path / "benchmark-data")
        assert store.wandb_project is None
        assert store.wandb_entity is None


class TestExtractTrend:
    """Tests for Store.extract_trend() — trend tracking."""

    def _make_store_with_runs(self, tmp_path, n_runs=5):
        """Create a store with n_runs, each with increasing throughput."""
        store = Store(tmp_path / "benchmark-data")
        runs = []
        for i in range(n_runs):
            run = Run(
                points=[
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Datarax"},
                        metrics={"throughput": Metric(5000.0 + i * 100)},
                    ),
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Grain"},
                        metrics={"throughput": Metric(4800.0 + i * 50)},
                    ),
                ],
                timestamp=datetime(2026, 2, 1 + i, 12, 0, 0),
                commit=f"commit_{i}",
                branch="main",
            )
            store.save(run)
            runs.append(run)
        return store, runs

    def test_extract_trend_returns_trend_series(self, tmp_path):
        """extract_trend returns a TrendSeries."""
        from benchkit.models import TrendSeries

        store, _ = self._make_store_with_runs(tmp_path, n_runs=3)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        assert isinstance(trend, TrendSeries)
        assert trend.metric == "throughput"

    def test_trend_has_correct_number_of_points(self, tmp_path):
        """Trend should have one point per matching run."""
        store, _ = self._make_store_with_runs(tmp_path, n_runs=5)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        assert len(trend.points) == 5

    def test_trend_ordered_by_timestamp(self, tmp_path):
        """Trend points should be ordered oldest → newest."""
        store, _ = self._make_store_with_runs(tmp_path, n_runs=4)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        for i in range(len(trend.points) - 1):
            assert trend.points[i].timestamp < trend.points[i + 1].timestamp

    def test_trend_values_match_runs(self, tmp_path):
        """Trend values should match the metric values from the runs."""
        store, runs = self._make_store_with_runs(tmp_path, n_runs=3)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        expected = [5000.0, 5100.0, 5200.0]
        actual = [tp.value for tp in trend.points]
        assert actual == expected

    def test_trend_filters_by_tags(self, tmp_path):
        """Trend should only include points matching the tags filter."""
        store, _ = self._make_store_with_runs(tmp_path, n_runs=3)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Grain"},
        )
        expected = [4800.0, 4850.0, 4900.0]
        actual = [tp.value for tp in trend.points]
        assert actual == expected

    def test_trend_n_runs_limits_output(self, tmp_path):
        """n_runs parameter limits to N most recent runs."""
        store, _ = self._make_store_with_runs(tmp_path, n_runs=5)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            n_runs=3,
        )
        assert len(trend.points) == 3
        # Should be the 3 most recent
        expected = [5200.0, 5300.0, 5400.0]
        actual = [tp.value for tp in trend.points]
        assert actual == expected

    def test_trend_includes_commit(self, tmp_path):
        """Verify TrendPoints include the commit hash from each run."""
        store, runs = self._make_store_with_runs(tmp_path, n_runs=2)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        assert trend.points[0].commit == "commit_0"
        assert trend.points[1].commit == "commit_1"

    def test_trend_empty_store(self, tmp_path):
        """Empty store returns TrendSeries with no points."""
        store = Store(tmp_path / "benchmark-data")
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )
        assert trend.points == []

    def test_trend_no_matching_points(self, tmp_path):
        """No matching points returns empty TrendSeries."""
        store, _ = self._make_store_with_runs(tmp_path, n_runs=3)
        trend = store.extract_trend(
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "NonExistent"},
        )
        assert trend.points == []
