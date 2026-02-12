"""
Test suite for MonitoredDAGExecutor.

File: tests/monitoring/test_dag_monitor.py
"""

import time
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any

from datarax.monitoring.dag_monitor import MonitoredDAGExecutor, monitored_pipeline
from datarax.monitoring.callbacks import MetricsObserver
from datarax.monitoring.metrics import MetricRecord
from datarax.dag.nodes import DataSourceNode, BatchNode, OperatorNode
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig, StructuralConfig


class MockDataSource(DataSourceModule):
    """Mock data source for testing."""

    # REQUIRED: Annotate data attribute with nnx.data() to prevent NNX container errors
    data: list = nnx.data()

    def __init__(
        self,
        data: list,
        *,
        config: StructuralConfig | None = None,
        rngs: nnx.Rngs | None = None,
    ):
        super().__init__(config or StructuralConfig(), rngs=rngs)
        self.data = data
        self.index = nnx.Variable(0)

    def __iter__(self):
        self.index.set_value(0)
        return self

    def __next__(self):
        if self.index.get_value() >= len(self.data):
            raise StopIteration
        item = self.data[self.index.get_value()]
        self.index.set_value(self.index.get_value() + 1)
        return item


class MockOperator(OperatorModule):
    """Mock operator for testing."""

    def __init__(
        self,
        factor: float = 2.0,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = "test_operator",
    ):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, rngs=rngs, name=name or "test_operator")
        self.factor = factor

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], Any]:
        result_data = jax.tree.map(lambda x: x * self.factor, data)
        return result_data, state, metadata


class MetricsCapture(MetricsObserver):
    """Observer that captures metrics for testing."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.metrics: list[MetricRecord] = []
        self.update_count = 0

    def update(self, metrics: list[MetricRecord]) -> None:
        self.update_count += 1
        self.metrics.extend(metrics)

    def get_metric_names(self) -> set[str]:
        return {m.name for m in self.metrics}  # type: ignore

    def get_metrics_by_component(self, component: str) -> list[MetricRecord]:
        return [m for m in self.metrics if m.component == component]


class TestMonitoredDAGExecutor:
    """Test suite for MonitoredDAGExecutor."""

    def test_basic_monitoring(self):
        """Test basic monitoring functionality."""
        # Create test data
        data = [{"value": float(i)} for i in range(10)]
        source = MockDataSource(data)

        # Create monitored executor
        executor = MonitoredDAGExecutor(metrics_enabled=True)
        executor.add(DataSourceNode(source)).batch(3).operate(MockOperator(name="test_transform"))

        # Add observer
        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process data
        batches = list(executor)

        # Verify batches processed
        assert len(batches) == 4  # 10 items / 3 batch size + 1 incomplete

        # Verify metrics collected
        assert observer.update_count > 0
        assert len(observer.metrics) > 0

        # Check for expected metrics
        metric_names = observer.get_metric_names()
        assert "pipeline_config" in metric_names
        assert "batch_size" in metric_names
        assert "total_batches" in metric_names
        assert "total_elements" in metric_names

    def test_per_node_metrics(self):
        """Test that metrics are collected per node."""
        data = [{"value": float(i)} for i in range(20)]

        executor = MonitoredDAGExecutor(metrics_enabled=True)
        executor.add(DataSourceNode(MockDataSource(data)))
        executor.batch(5)
        executor.operate(MockOperator(factor=2.0, name="test_transform_1"))
        executor.operate(MockOperator(factor=3.0, name="test_transform_2"))

        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process all data
        list(executor)

        # Check node-specific metrics
        node_metrics = observer.get_metrics_by_component("nodes")
        node_metric_names = {m.name for m in node_metrics}

        # Should have metrics for each node type
        assert any("DataSource" in name for name in node_metric_names)
        assert any("Batch" in name for name in node_metric_names)
        assert any("transform" in name.lower() for name in node_metric_names)

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        data = [{"value": jnp.ones((100, 100))} for i in range(5)]

        executor = MonitoredDAGExecutor(metrics_enabled=True, track_memory=True)
        executor.add(DataSourceNode(MockDataSource(data))).batch(2)

        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process data
        list(executor)

        # Check for memory metrics (if psutil available)
        try:
            import psutil  # noqa: F401

            system_metrics = observer.get_metrics_by_component("system")
            if system_metrics:
                metric_names = {m.name for m in system_metrics}
                assert "memory_rss_mb" in metric_names or "memory_vms_mb" in metric_names
        except ImportError:
            # psutil not available, skip memory check
            pass

    def test_performance_report(self):
        """Test performance report generation."""
        data = [{"value": float(i)} for i in range(50)]

        executor = MonitoredDAGExecutor(name="test_pipeline", metrics_enabled=True)
        executor.add(DataSourceNode(MockDataSource(data)))
        executor.batch(10)
        executor.operate(MockOperator(name="test_transform"))

        # Process data
        list(executor)

        # Get performance report
        report = executor.get_performance_report()

        # Verify report structure
        assert report["pipeline_name"] == "test_pipeline"
        assert report["total_batches"] == 5
        assert report["total_elements"] == 50
        assert "total_time" in report
        assert "configuration" in report
        assert "statistics" in report
        assert "nodes" in report

        # Check configuration
        config = report["configuration"]
        assert config["metrics_enabled"] is True
        assert "enforce_batch" in config
        assert "enable_caching" in config

    def test_cache_statistics(self):
        """Test cache statistics collection."""

        data = [{"value": float(i)} for i in range(10)]

        executor = MonitoredDAGExecutor(enable_caching=True, metrics_enabled=True)

        # Build pipeline with cache
        executor.add(DataSourceNode(MockDataSource(data)))
        executor.batch(2)
        executor.cache(10)  # Add cache node
        executor.operate(MockOperator(name="test_transform"))

        # Process data twice to generate cache hits
        list(executor)

        # Reset source and process again
        executor._source_node = DataSourceNode(MockDataSource(data))
        list(executor)

        # Get report with cache stats
        report = executor.get_performance_report()
        stats = report["statistics"]

        # Should have cache statistics
        cache_keys = [k for k in stats.keys() if "cache" in k]
        assert len(cache_keys) > 0

    def test_notification_threshold(self):
        """Test that notification threshold works."""
        data = [{"value": float(i)} for i in range(100)]

        executor = MonitoredDAGExecutor(
            metrics_enabled=True,
            notify_threshold=10,  # Notify every 10 metrics
        )
        executor.add(DataSourceNode(MockDataSource(data))).batch(1)

        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process data
        list(executor)

        # Should have multiple updates due to threshold
        assert observer.update_count > 1

    def test_metrics_disabled(self):
        """Test that metrics can be disabled."""
        data = [{"value": float(i)} for i in range(10)]

        executor = MonitoredDAGExecutor(metrics_enabled=False)
        executor.add(DataSourceNode(MockDataSource(data))).batch(3)

        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process data
        list(executor)

        # Should not receive metrics
        assert observer.update_count == 0
        assert len(observer.metrics) == 0

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        data = [{"value": float(i)} for i in range(10)]

        executor = MonitoredDAGExecutor(metrics_enabled=True)
        executor.add(DataSourceNode(MockDataSource(data))).batch(2)

        # Process once
        list(executor)

        # Check counters
        assert executor.total_batches_processed.get_value() > 0
        assert executor.total_elements_processed.get_value() > 0

        # Reset metrics
        executor.reset_metrics()

        # Verify reset
        assert executor.total_batches_processed.get_value() == 0
        assert executor.total_elements_processed.get_value() == 0
        assert len(executor._mon.node_timers) == 0
        assert len(executor._mon.node_counts) == 0

    def test_monitored_pipeline_convenience(self):
        """Test monitored_pipeline convenience function."""
        source = MockDataSource([1, 2, 3, 4, 5])
        batch = BatchNode(batch_size=2)
        transform = OperatorNode(MockOperator(name="test_transform"))

        # Create using convenience function
        executor = monitored_pipeline(source, batch, transform, metrics_enabled=True)

        assert isinstance(executor, MonitoredDAGExecutor)
        assert executor.metrics.enabled is True

        # Process data
        batches = list(executor)
        assert len(batches) == 3  # 5 items / 2 batch size + 1 incomplete

    def test_complex_dag_monitoring(self):
        """Test monitoring of complex DAG structures."""

        data = [{"value": jnp.array([float(i)])} for i in range(20)]

        executor = MonitoredDAGExecutor(metrics_enabled=True)

        # Build complex DAG
        executor.add(DataSourceNode(MockDataSource(data)))
        executor.batch(5)

        # Parallel processing
        executor.parallel(
            [
                OperatorNode(MockOperator(factor=2.0, name="test_transform_1")),
                OperatorNode(MockOperator(factor=3.0, name="test_transform_2")),
            ]
        )
        executor.merge("mean")

        observer = MetricsCapture()
        executor.callbacks.register(observer)

        # Process data
        list(executor)

        # Verify metrics for parallel execution
        assert observer.update_count > 0

        # Check that we got metrics for parallel nodes
        node_metrics = observer.get_metrics_by_component("nodes")
        assert len(node_metrics) > 0


class TestMonitoredDAGExecutorIntegration:
    """Integration tests for MonitoredDAGExecutor."""

    def test_compatibility_with_reporters(self):
        """Test compatibility with existing reporters."""
        from datarax.monitoring.reporters import ConsoleReporter

        data = [{"value": float(i)} for i in range(10)]

        executor = MonitoredDAGExecutor(metrics_enabled=True)
        executor.add(DataSourceNode(MockDataSource(data))).batch(3)

        # Add console reporter (should work without errors)
        reporter = ConsoleReporter()
        executor.callbacks.register(reporter)

        # Process data - should print to console
        list(executor)

    def test_performance_comparison(self):
        """Test performance impact of monitoring."""
        data = [{"value": jnp.ones((100, 100))} for i in range(100)]

        # Without monitoring
        executor_no_metrics = MonitoredDAGExecutor(metrics_enabled=False)
        executor_no_metrics.add(DataSourceNode(MockDataSource(data))).batch(10)

        start = time.time()
        list(executor_no_metrics)
        time_no_metrics = time.time() - start

        # With monitoring
        executor_with_metrics = MonitoredDAGExecutor(metrics_enabled=True)
        executor_with_metrics.add(DataSourceNode(MockDataSource(data))).batch(10)

        start = time.time()
        list(executor_with_metrics)
        time_with_metrics = time.time() - start

        # Monitoring overhead should be reasonable (< 50%)
        overhead = (time_with_metrics - time_no_metrics) / time_no_metrics
        assert overhead < 0.5, f"Monitoring overhead too high: {overhead:.2%}"


if __name__ == "__main__":
    # Run tests
    test_basic = TestMonitoredDAGExecutor()
    test_basic.test_basic_monitoring()
    test_basic.test_per_node_metrics()
    test_basic.test_memory_tracking()
    test_basic.test_performance_report()
    test_basic.test_cache_statistics()
    test_basic.test_notification_threshold()
    test_basic.test_metrics_disabled()
    test_basic.test_reset_metrics()
    test_basic.test_monitored_pipeline_convenience()
    test_basic.test_complex_dag_monitoring()

    test_integration = TestMonitoredDAGExecutorIntegration()
    test_integration.test_compatibility_with_reporters()
    test_integration.test_performance_comparison()

    print("✓ All MonitoredDAGExecutor tests passed!")
