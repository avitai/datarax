"""Tests for regression.py - Performance regression detection."""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from datarax.benchmarking.profiler import ProfileResult
from datarax.benchmarking.regression import (
    PerformanceRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)


class TestRegressionSeverity(unittest.TestCase):
    """Tests for the RegressionSeverity enum."""

    def test_severity_values(self):
        """Test that all severity levels have correct string values."""
        self.assertEqual(RegressionSeverity.LOW.value, "low")
        self.assertEqual(RegressionSeverity.MEDIUM.value, "medium")
        self.assertEqual(RegressionSeverity.HIGH.value, "high")
        self.assertEqual(RegressionSeverity.CRITICAL.value, "critical")


class TestPerformanceRegression(unittest.TestCase):
    """Tests for the PerformanceRegression dataclass."""

    def test_regression_creation(self):
        """Test creating a PerformanceRegression instance."""
        regression = PerformanceRegression(
            metric_name="timing.latency",
            baseline_value=1.0,
            current_value=1.5,
            regression_percent=50.0,
            severity=RegressionSeverity.HIGH,
            description="Latency increased by 50%",
        )

        self.assertEqual(regression.metric_name, "timing.latency")
        self.assertEqual(regression.baseline_value, 1.0)
        self.assertEqual(regression.current_value, 1.5)
        self.assertEqual(regression.regression_percent, 50.0)
        self.assertEqual(regression.severity, RegressionSeverity.HIGH)
        self.assertEqual(regression.description, "Latency increased by 50%")
        self.assertIsInstance(regression.timestamp, datetime)
        self.assertEqual(regression.metadata, {})

    def test_regression_with_metadata(self):
        """Test PerformanceRegression with custom metadata."""
        metadata = {"baseline_std": 0.1, "samples": 10}
        regression = PerformanceRegression(
            metric_name="memory.peak_mb",
            baseline_value=100.0,
            current_value=150.0,
            regression_percent=50.0,
            severity=RegressionSeverity.CRITICAL,
            description="Memory usage increased",
            metadata=metadata,
        )

        self.assertEqual(regression.metadata, metadata)

    def test_regression_to_dict(self):
        """Test converting PerformanceRegression to dictionary."""
        timestamp = datetime(2025, 1, 1, 12, 0, 0)
        regression = PerformanceRegression(
            metric_name="timing.execution_ms",
            baseline_value=10.0,
            current_value=15.0,
            regression_percent=50.0,
            severity=RegressionSeverity.MEDIUM,
            description="Execution time increased",
            timestamp=timestamp,
            metadata={"test": "data"},
        )

        result = regression.to_dict()

        self.assertEqual(result["metric_name"], "timing.execution_ms")
        self.assertEqual(result["baseline_value"], 10.0)
        self.assertEqual(result["current_value"], 15.0)
        self.assertEqual(result["regression_percent"], 50.0)
        self.assertEqual(result["severity"], "medium")
        self.assertEqual(result["description"], "Execution time increased")
        self.assertEqual(result["timestamp"], "2025-01-01T12:00:00")
        self.assertEqual(result["metadata"], {"test": "data"})


class TestRegressionReport(unittest.TestCase):
    """Tests for the RegressionReport dataclass."""

    def test_empty_report(self):
        """Test creating an empty RegressionReport."""
        report = RegressionReport()

        self.assertEqual(report.regressions, [])
        self.assertEqual(report.improvements, [])
        self.assertEqual(report.stable_metrics, [])
        self.assertEqual(report.summary, {})
        self.assertIsInstance(report.timestamp, datetime)

    def test_has_regressions_empty(self):
        """Test has_regressions() with no regressions."""
        report = RegressionReport()
        self.assertFalse(report.has_regressions())

    def test_has_regressions_with_data(self):
        """Test has_regressions() with regressions present."""
        regression = PerformanceRegression(
            metric_name="test",
            baseline_value=1.0,
            current_value=2.0,
            regression_percent=100.0,
            severity=RegressionSeverity.LOW,
            description="Test regression",
        )
        report = RegressionReport(regressions=[regression])

        self.assertTrue(report.has_regressions())

    def test_has_critical_regressions_none(self):
        """Test has_critical_regressions() with no critical regressions."""
        regression = PerformanceRegression(
            metric_name="test",
            baseline_value=1.0,
            current_value=1.1,
            regression_percent=10.0,
            severity=RegressionSeverity.LOW,
            description="Minor regression",
        )
        report = RegressionReport(regressions=[regression])

        self.assertFalse(report.has_critical_regressions())

    def test_has_critical_regressions_present(self):
        """Test has_critical_regressions() with critical regression."""
        critical = PerformanceRegression(
            metric_name="critical_metric",
            baseline_value=1.0,
            current_value=2.0,
            regression_percent=100.0,
            severity=RegressionSeverity.CRITICAL,
            description="Critical issue",
        )
        report = RegressionReport(regressions=[critical])

        self.assertTrue(report.has_critical_regressions())

    def test_get_worst_regression_empty(self):
        """Test get_worst_regression() with no regressions."""
        report = RegressionReport()
        self.assertIsNone(report.get_worst_regression())

    def test_get_worst_regression(self):
        """Test get_worst_regression() returns highest percentage."""
        low = PerformanceRegression(
            metric_name="low",
            baseline_value=1.0,
            current_value=1.1,
            regression_percent=10.0,
            severity=RegressionSeverity.LOW,
            description="Low",
        )
        high = PerformanceRegression(
            metric_name="high",
            baseline_value=1.0,
            current_value=2.0,
            regression_percent=100.0,
            severity=RegressionSeverity.CRITICAL,
            description="High",
        )
        medium = PerformanceRegression(
            metric_name="medium",
            baseline_value=1.0,
            current_value=1.5,
            regression_percent=50.0,
            severity=RegressionSeverity.MEDIUM,
            description="Medium",
        )

        report = RegressionReport(regressions=[low, medium, high])
        worst = report.get_worst_regression()

        self.assertIsNotNone(worst)
        assert worst is not None  # for type checker
        self.assertEqual(worst.metric_name, "high")
        self.assertEqual(worst.regression_percent, 100.0)

    def test_report_to_dict(self):
        """Test converting RegressionReport to dictionary."""
        regression = PerformanceRegression(
            metric_name="test",
            baseline_value=1.0,
            current_value=2.0,
            regression_percent=100.0,
            severity=RegressionSeverity.HIGH,
            description="Test",
        )
        improvement = {"metric_name": "improved", "improvement_percent": 20.0}

        report = RegressionReport(
            regressions=[regression],
            improvements=[improvement],
            stable_metrics=["stable.metric"],
            summary={"total": 1},
        )

        result = report.to_dict()

        self.assertEqual(len(result["regressions"]), 1)
        self.assertEqual(result["improvements"], [improvement])
        self.assertEqual(result["stable_metrics"], ["stable.metric"])
        self.assertEqual(result["summary"], {"total": 1})
        self.assertIn("timestamp", result)

    def test_report_save(self):
        """Test saving RegressionReport to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nested" / "report.json"

            regression = PerformanceRegression(
                metric_name="test",
                baseline_value=1.0,
                current_value=2.0,
                regression_percent=100.0,
                severity=RegressionSeverity.LOW,
                description="Test",
            )
            report = RegressionReport(regressions=[regression])

            # Save report
            report.save(filepath)

            # Verify file exists and is valid JSON
            self.assertTrue(filepath.exists())

            with open(filepath, "r") as f:
                data = json.load(f)

            self.assertEqual(len(data["regressions"]), 1)
            self.assertEqual(data["regressions"][0]["metric_name"], "test")


class TestRegressionDetector(unittest.TestCase):
    """Tests for the RegressionDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline_dir = Path(self.temp_dir) / "baselines"

    def test_detector_initialization(self):
        """Test RegressionDetector initialization."""
        detector = RegressionDetector(
            baseline_dir=self.baseline_dir,
            regression_threshold=0.15,
            critical_threshold=0.30,
            min_samples=5,
        )

        self.assertEqual(detector.baseline_dir, self.baseline_dir)
        self.assertTrue(detector.baseline_dir.exists())
        self.assertEqual(detector.regression_threshold, 0.15)
        self.assertEqual(detector.critical_threshold, 0.30)
        self.assertEqual(detector.min_samples, 5)

    def test_add_baseline(self):
        """Test adding baseline measurements."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir)

        profile = ProfileResult(
            timing_metrics={"execution_ms": 10.0},
            memory_metrics={"peak_mb": 100.0},
            gpu_metrics={},
        )

        detector.add_baseline(profile, "test_benchmark")

        # Verify baseline was added
        summary = detector.get_baseline_summary()
        self.assertIn("test_benchmark", summary)
        self.assertEqual(summary["test_benchmark"]["num_samples"], 1)

    def test_detect_regressions_no_baseline(self):
        """Test regression detection with no baseline data."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir)

        profile = ProfileResult(
            timing_metrics={"execution_ms": 10.0},
            memory_metrics={},
            gpu_metrics={},
        )

        with self.assertWarns(UserWarning):
            report = detector.detect_regressions(profile, "unknown_benchmark")

        self.assertEqual(report.summary["status"], "no_baseline")

    def test_detect_regressions_insufficient_samples(self):
        """Test regression detection with insufficient samples."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir, min_samples=5)

        # Add only 2 baselines (less than min_samples=5)
        for i in range(2):
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0 + i},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, "test_benchmark")

        # Try to detect regressions
        current = ProfileResult(
            timing_metrics={"execution_ms": 20.0},
            memory_metrics={},
            gpu_metrics={},
        )

        with self.assertWarns(UserWarning):
            report = detector.detect_regressions(current, "test_benchmark")

        self.assertEqual(report.summary["status"], "insufficient_data")

    def test_detect_regression_timing_increase(self):
        """Test detecting timing regression (higher is worse)."""
        detector = RegressionDetector(
            baseline_dir=self.baseline_dir, regression_threshold=0.10, min_samples=3
        )

        # Add 3 baseline measurements around 10ms
        for i in range(3):
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0 + i * 0.1},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, "test_benchmark")

        # Current measurement is 20% higher (should be detected as regression)
        current = ProfileResult(
            timing_metrics={"execution_ms": 12.0},
            memory_metrics={},
            gpu_metrics={},
        )

        report = detector.detect_regressions(current, "test_benchmark")

        self.assertTrue(report.has_regressions())
        self.assertEqual(len(report.regressions), 1)
        self.assertEqual(report.regressions[0].metric_name, "timing.execution_ms")
        self.assertGreater(report.regressions[0].regression_percent, 10.0)

    def test_detect_improvement(self):
        """Test detecting performance improvement."""
        detector = RegressionDetector(
            baseline_dir=self.baseline_dir, regression_threshold=0.10, min_samples=3
        )

        # Add baselines around 10ms
        for i in range(3):
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, "test_benchmark")

        # Current measurement is 20% lower (improvement for timing)
        current = ProfileResult(
            timing_metrics={"execution_ms": 8.0},
            memory_metrics={},
            gpu_metrics={},
        )

        report = detector.detect_regressions(current, "test_benchmark")

        self.assertFalse(report.has_regressions())
        self.assertGreater(len(report.improvements), 0)

    def test_detect_stable_metric(self):
        """Test detecting stable metrics (within threshold)."""
        detector = RegressionDetector(
            baseline_dir=self.baseline_dir, regression_threshold=0.10, min_samples=3
        )

        # Add baselines
        for i in range(3):
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, "test_benchmark")

        # Current measurement within 5% (below 10% threshold)
        current = ProfileResult(
            timing_metrics={"execution_ms": 10.3},
            memory_metrics={},
            gpu_metrics={},
        )

        report = detector.detect_regressions(current, "test_benchmark")

        self.assertFalse(report.has_regressions())
        self.assertIn("timing.execution_ms", report.stable_metrics)

    def test_severity_determination(self):
        """Test that severity levels are assigned correctly."""
        detector = RegressionDetector(
            baseline_dir=self.baseline_dir,
            regression_threshold=0.10,
            critical_threshold=0.25,
            min_samples=3,
        )

        # Test LOW severity (10-15%)
        self.assertEqual(detector._determine_severity(12.0), RegressionSeverity.LOW)

        # Test MEDIUM severity (15-20%)
        self.assertEqual(detector._determine_severity(17.0), RegressionSeverity.MEDIUM)

        # Test HIGH severity (20-25%)
        self.assertEqual(detector._determine_severity(22.0), RegressionSeverity.HIGH)

        # Test CRITICAL severity (>25%)
        self.assertEqual(detector._determine_severity(30.0), RegressionSeverity.CRITICAL)

    def test_clear_specific_baseline(self):
        """Test clearing specific benchmark baseline."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir)

        # Add baselines for two benchmarks
        for benchmark in ["bench1", "bench2"]:
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, benchmark)

        # Clear only bench1
        detector.clear_baselines("bench1")

        summary = detector.get_baseline_summary()
        self.assertNotIn("bench1", summary)
        self.assertIn("bench2", summary)

    def test_clear_all_baselines(self):
        """Test clearing all baselines."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir)

        # Add baselines
        profile = ProfileResult(
            timing_metrics={"execution_ms": 10.0},
            memory_metrics={},
            gpu_metrics={},
        )
        detector.add_baseline(profile, "test_benchmark")

        # Clear all
        detector.clear_baselines()

        summary = detector.get_baseline_summary()
        self.assertEqual(len(summary), 0)

    def test_baseline_persistence(self):
        """Test that baselines are saved and loaded from disk."""
        # Create detector and add baseline
        detector1 = RegressionDetector(baseline_dir=self.baseline_dir)
        profile = ProfileResult(
            timing_metrics={"execution_ms": 10.0},
            memory_metrics={},
            gpu_metrics={},
        )
        detector1.add_baseline(profile, "test_benchmark")

        # Create new detector with same baseline_dir
        detector2 = RegressionDetector(baseline_dir=self.baseline_dir)

        # Verify baseline was loaded
        summary = detector2.get_baseline_summary()
        self.assertIn("test_benchmark", summary)
        self.assertEqual(summary["test_benchmark"]["num_samples"], 1)

    def test_max_samples_limit(self):
        """Test that old baselines are removed when limit is reached."""
        detector = RegressionDetector(baseline_dir=self.baseline_dir)

        # Add 60 baselines (exceeds max_samples=50)
        for i in range(60):
            profile = ProfileResult(
                timing_metrics={"execution_ms": 10.0 + i},
                memory_metrics={},
                gpu_metrics={},
            )
            detector.add_baseline(profile, "test_benchmark")

        summary = detector.get_baseline_summary()
        # Should be capped at 50
        self.assertEqual(summary["test_benchmark"]["num_samples"], 50)


if __name__ == "__main__":
    unittest.main()
