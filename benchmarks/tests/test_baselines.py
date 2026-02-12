"""Tests for BaselineStore.

TDD: Write tests first, then implement.
Design ref: Sections 9.1, 9.2 of the benchmark report.
"""

from pathlib import Path

from benchmarks.core.baselines import BaselineStore
from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample


def _make_result(
    wall_clock_sec: float = 1.0,
    num_batches: int = 50,
    num_elements: int = 5000,
    per_batch_time: float | None = None,
) -> BenchmarkResult:
    """Create a BenchmarkResult for testing."""
    if per_batch_time is None:
        per_batch_time = wall_clock_sec / num_batches
    return BenchmarkResult(
        framework="Datarax",
        scenario_id="CV-1",
        variant="small",
        timing=TimingSample(
            wall_clock_sec=wall_clock_sec,
            per_batch_times=[per_batch_time] * num_batches,
            first_batch_time=per_batch_time * 2,
            num_batches=num_batches,
            num_elements=num_elements,
        ),
        resources=None,
        environment={"jax_version": "0.4.35"},
        config={"batch_size": 100},
    )


class TestBaselineStoreInit:
    """Test BaselineStore initialization."""

    def test_creates_directory(self, tmp_path: Path):
        """Test BaselineStore creates baselines directory if missing."""
        baselines_dir = tmp_path / "new_baselines"
        BaselineStore(baselines_dir)
        assert baselines_dir.exists()

    def test_accepts_existing_directory(self, tmp_path: Path):
        """Test BaselineStore works with existing directory."""
        store = BaselineStore(tmp_path)
        assert store.baselines_dir == tmp_path


class TestBaselineStoreSaveLoad:
    """Test save/load operations."""

    def test_save_creates_json_file(self, tmp_path: Path):
        """Test saving a baseline creates a JSON file."""
        store = BaselineStore(tmp_path)
        result = _make_result()
        store.save("CV-1_small", result)

        filepath = tmp_path / "CV-1_small.json"
        assert filepath.exists()

    def test_load_returns_dict(self, tmp_path: Path):
        """Test loading a baseline returns its data."""
        store = BaselineStore(tmp_path)
        result = _make_result(wall_clock_sec=2.0, num_elements=10000)
        store.save("test_baseline", result)

        loaded = store.load("test_baseline")
        assert loaded is not None
        assert loaded["framework"] == "Datarax"
        assert loaded["scenario_id"] == "CV-1"

    def test_load_nonexistent_returns_none(self, tmp_path: Path):
        """Test loading a nonexistent baseline returns None."""
        store = BaselineStore(tmp_path)
        loaded = store.load("nonexistent")
        assert loaded is None

    def test_save_load_round_trip(self, tmp_path: Path):
        """Test that save → load preserves key data."""
        store = BaselineStore(tmp_path)
        result = _make_result(wall_clock_sec=1.5, num_elements=7500)
        store.save("round_trip", result)

        loaded = store.load("round_trip")
        assert loaded is not None
        assert loaded["timing"]["wall_clock_sec"] == 1.5
        assert loaded["timing"]["num_elements"] == 7500

    def test_overwrite_existing(self, tmp_path: Path):
        """Test saving to same name overwrites."""
        store = BaselineStore(tmp_path)
        store.save("baseline", _make_result(wall_clock_sec=1.0))
        store.save("baseline", _make_result(wall_clock_sec=2.0))

        loaded = store.load("baseline")
        assert loaded["timing"]["wall_clock_sec"] == 2.0


class TestBaselineStoreCompare:
    """Test baseline comparison with statistical analysis."""

    def test_compare_pass_same_performance(self, tmp_path: Path):
        """Test comparison passes when performance is equivalent."""
        store = BaselineStore(tmp_path)
        baseline = _make_result(wall_clock_sec=1.0, num_elements=5000)
        store.save("baseline", baseline)

        # Current result is similar
        current = _make_result(wall_clock_sec=1.0, num_elements=5000)
        verdict = store.compare("baseline", current)

        assert verdict["status"] == "pass"

    def test_compare_warning_on_mild_regression(self, tmp_path: Path):
        """Test comparison warns on mild throughput regression."""
        store = BaselineStore(tmp_path)
        # Baseline: 5000 elements in 1.0s = 5000 el/s
        baseline = _make_result(wall_clock_sec=1.0, num_elements=5000)
        store.save("baseline", baseline)

        # Current: 5000 elements in 1.3s = ~3846 el/s (23% slower)
        current = _make_result(wall_clock_sec=1.3, num_elements=5000)
        verdict = store.compare("baseline", current)

        assert verdict["status"] in ("warning", "failure")

    def test_compare_failure_on_severe_regression(self, tmp_path: Path):
        """Test comparison fails on severe throughput regression."""
        store = BaselineStore(tmp_path)
        baseline = _make_result(wall_clock_sec=1.0, num_elements=5000)
        store.save("baseline", baseline)

        # Current: 2x slower
        current = _make_result(wall_clock_sec=2.0, num_elements=5000)
        verdict = store.compare("baseline", current)

        assert verdict["status"] == "failure"

    def test_compare_pass_on_improvement(self, tmp_path: Path):
        """Test comparison passes when current is faster."""
        store = BaselineStore(tmp_path)
        baseline = _make_result(wall_clock_sec=2.0, num_elements=5000)
        store.save("baseline", baseline)

        # Current: 2x faster
        current = _make_result(wall_clock_sec=1.0, num_elements=5000)
        verdict = store.compare("baseline", current)

        assert verdict["status"] == "pass"

    def test_compare_nonexistent_baseline_returns_none(self, tmp_path: Path):
        """Test comparing against nonexistent baseline returns None."""
        store = BaselineStore(tmp_path)
        current = _make_result()
        verdict = store.compare("nonexistent", current)

        assert verdict is None

    def test_compare_returns_metrics(self, tmp_path: Path):
        """Test comparison returns metric details."""
        store = BaselineStore(tmp_path)
        baseline = _make_result(wall_clock_sec=1.0, num_elements=5000)
        store.save("baseline", baseline)

        current = _make_result(wall_clock_sec=1.0, num_elements=5000)
        verdict = store.compare("baseline", current)

        assert "throughput_ratio" in verdict
        assert "baseline_throughput" in verdict
        assert "current_throughput" in verdict


class TestBaselineStoreDualGate:
    """Test the dual-gate comparison (statistical AND practical significance)."""

    def _make_result_with_jitter(
        self,
        mean_batch_time: float,
        jitter: float,
        num_batches: int = 20,
        num_elements: int = 5000,
    ) -> BenchmarkResult:
        """Create a result with realistic per-batch time variation."""
        import random

        rng = random.Random(42)
        per_batch_times = [
            mean_batch_time + rng.uniform(-jitter, jitter) for _ in range(num_batches)
        ]
        wall_clock_sec = sum(per_batch_times)
        return BenchmarkResult(
            framework="Datarax",
            scenario_id="CV-1",
            variant="small",
            timing=TimingSample(
                wall_clock_sec=wall_clock_sec,
                per_batch_times=per_batch_times,
                first_batch_time=per_batch_times[0] * 2,
                num_batches=num_batches,
                num_elements=num_elements,
            ),
            resources=None,
            environment={"jax_version": "0.4.35"},
            config={"batch_size": 100},
        )

    def test_small_regression_passes_despite_statistical_significance(
        self,
        tmp_path: Path,
    ):
        """A 5% slowdown should PASS even if t-test says it's significant.

        With enough samples and low jitter, t-test detects even tiny
        differences. The dual gate prevents flagging these as regressions.
        """
        store = BaselineStore(tmp_path)
        # Baseline: mean batch time 0.001s → 5M elem/s
        baseline = self._make_result_with_jitter(0.001, jitter=0.0001)
        store.save("test", baseline)

        # Current: 5% slower (0.00105s) → still above 0.80 ratio
        current = self._make_result_with_jitter(0.00105, jitter=0.0001)
        verdict = store.compare("test", current)

        assert verdict is not None
        assert verdict["status"] == "pass"

    def test_large_regression_fails(self, tmp_path: Path):
        """A 50% slowdown should FAIL — both gates trigger."""
        store = BaselineStore(tmp_path)
        baseline = self._make_result_with_jitter(0.001, jitter=0.0001)
        store.save("test", baseline)

        # Current: 50% slower → throughput ratio ~0.67
        current = self._make_result_with_jitter(0.0015, jitter=0.0001)
        verdict = store.compare("test", current)

        assert verdict is not None
        assert verdict["status"] == "failure"

    def test_custom_failure_ratio(self, tmp_path: Path):
        """Custom failure_ratio threshold works."""
        store = BaselineStore(tmp_path)
        baseline = self._make_result_with_jitter(0.001, jitter=0.0001)
        store.save("test", baseline)

        # 30% slower → ratio ~0.77
        current = self._make_result_with_jitter(0.0013, jitter=0.0001)

        # Default failure_ratio=0.80 → should fail (0.77 < 0.80)
        verdict_default = store.compare("test", current)
        assert verdict_default["status"] == "failure"

        # Stricter failure_ratio=0.70 → should pass (0.77 > 0.70)
        verdict_strict = store.compare("test", current, failure_ratio=0.70)
        assert verdict_strict["status"] in ("pass", "warning")


class TestBaselineStoreArchive:
    """Test baseline archiving."""

    def test_archive_moves_file(self, tmp_path: Path):
        """Test archiving moves baseline to archive subdirectory."""
        store = BaselineStore(tmp_path)
        store.save("old_baseline", _make_result())

        store.archive("old_baseline")

        # Original should be gone
        assert not (tmp_path / "old_baseline.json").exists()
        # Should be in archive/
        archive_dir = tmp_path / "archive"
        assert archive_dir.exists()
        archived_files = list(archive_dir.glob("old_baseline_*.json"))
        assert len(archived_files) == 1

    def test_archive_nonexistent_does_nothing(self, tmp_path: Path):
        """Test archiving nonexistent baseline is a no-op."""
        store = BaselineStore(tmp_path)
        store.archive("nonexistent")  # Should not raise

    def test_list_baselines(self, tmp_path: Path):
        """Test listing available baselines."""
        store = BaselineStore(tmp_path)
        store.save("baseline_a", _make_result())
        store.save("baseline_b", _make_result())

        baselines = store.list_baselines()
        assert "baseline_a" in baselines
        assert "baseline_b" in baselines
        assert len(baselines) == 2
