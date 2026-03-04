"""Monitoring system for DAG executor.

This module provides MonitoredDAGExecutor that integrates metrics collection
with the DAGExecutor pipeline system.
"""

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterator
import time
import jax
import flax.nnx as nnx

from datarax.dag.dag_executor import DAGExecutor
from datarax.dag.nodes import Node
from datarax.monitoring.callbacks import CallbackRegistry
from datarax.monitoring.metrics import MetricsCollector, MetricRecord
from datarax.typing import Batch


@dataclass
class _MonitoringState:
    """Mutable pipeline monitoring state."""

    notify_counter: int = 0
    notify_threshold: int = 100
    node_timers: dict[str, list[float]] = field(default_factory=dict)
    node_counts: dict[str, int] = field(default_factory=dict)
    pipeline_start_time: float | None = None
    batch_times: list[float] = field(default_factory=list)


class MonitoredDAGExecutor(DAGExecutor):
    """DAG executor with integrated metrics collection.

    This class extends DAGExecutor with metrics collection,
    performance monitoring, and reporting capabilities.

    Key Features:

        - Automatic metrics collection for all nodes
        - Performance tracking per node
        - Memory usage monitoring
        - Throughput measurement
        - Custom metrics support
        - Real-time reporting via callbacks

    Examples:
        Basic monitoring:

        ```python
        executor = MonitoredDAGExecutor()
        executor.add(source).batch(32).transform(normalize())

        # Add observers
        from datarax.monitoring.reporters import ConsoleReporter
        executor.callbacks.register(ConsoleReporter())

        # Process with monitoring
        for batch in executor:
            # Metrics automatically collected
            process(batch)
        ```
    """

    def __init__(
        self,
        graph: Node | None = None,
        *,
        rngs: nnx.Rngs | None = None,
        enable_caching: bool = True,
        jit_compile: bool = False,
        enforce_batch: bool = True,
        metrics_enabled: bool = True,
        notify_threshold: int = 100,
        track_memory: bool = True,
        name: str | None = None,
    ):
        """Initialize monitored DAG executor.

        Args:
            graph: Initial graph node or None
            rngs: RNG state for the pipeline
            enable_caching: Whether to cache intermediate results
            jit_compile: Whether to JIT compile the pipeline
            enforce_batch: Whether to enforce batch-first processing
            metrics_enabled: Whether to enable metrics collection
            notify_threshold: Notify observers after this many metrics
            track_memory: Whether to track memory usage
            name: Optional name for the executor
        """
        super().__init__(
            graph=graph,
            rngs=rngs,
            enable_caching=enable_caching,
            jit_compile=jit_compile,
            enforce_batch=enforce_batch,
            name=name or "MonitoredDAGExecutor",
        )

        # Monitoring components
        self.metrics = MetricsCollector(enabled=metrics_enabled)
        self.callbacks = CallbackRegistry()
        self.track_memory = track_memory

        # Monitoring state
        self._mon = _MonitoringState(notify_threshold=notify_threshold)

        # Performance metrics
        self.total_batches_processed = nnx.Variable(0)
        self.total_elements_processed = nnx.Variable(0)
        self.total_processing_time = nnx.Variable(0.0)

    def __iter__(self) -> Iterator[Batch]:
        """Create monitored iterator for the pipeline.

        Returns:
            Iterator that yields batches with metrics collection
        """
        # Start pipeline metrics
        if self.metrics.enabled:
            self._mon.pipeline_start_time = time.time()
            self.metrics.start_timer("pipeline_iteration", "pipeline")

            # Record pipeline configuration
            self.metrics.record_metric(
                "pipeline_config",
                1.0,
                component="pipeline",
                metadata={
                    "enforce_batch": self.enforce_batch,
                    "enable_caching": self.enable_caching,
                    "jit_compile": self.jit_compile,
                    "graph_type": type(self.graph).__name__,
                },
            )

            # Track DataSourceNode metrics since it's not in the execution graph
            if self._source_node is not None:
                node_name = "DataSource"
                if node_name not in self._mon.node_timers:
                    self._mon.node_timers[node_name] = []
                    self._mon.node_counts[node_name] = 0

        # Initialize batch counter
        self._batch_count = 0

        # Call parent's __iter__ which sets up self._iterator and returns self
        return super().__iter__()

    def __next__(self) -> Batch:
        """Get next batch with metrics collection.

        Returns:
            Next processed batch with monitoring

        Raises:
            StopIteration: When pipeline is exhausted
        """
        try:
            batch_start = time.time() if self.metrics.enabled else None
            source_start = self._start_source_timer()

            # Get next batch from parent
            batch = super().__next__()

            self._record_source_metrics(source_start)
            self._record_batch_iteration_metrics(batch, batch_start)

            return batch

        except StopIteration:
            # Finalize metrics when iteration completes
            if self.metrics.enabled:
                self._finalize_metrics()
            raise

    def _start_source_timer(self) -> float | None:
        """Start timer for DataSource pseudo-node metrics."""
        if self.metrics.enabled and self._source_node is not None:
            return time.time()
        return None

    def _record_source_metrics(self, source_start: float | None) -> None:
        """Record DataSource timing if source timing was enabled."""
        if source_start is None:
            return
        source_time = time.time() - source_start
        node_name = "DataSource"
        self._mon.node_timers[node_name].append(source_time)
        self._mon.node_counts[node_name] += 1
        self.metrics.record_metric(
            f"node_{node_name}_execution_time",
            source_time,
            component="nodes",
            metadata={
                "node_type": "DataSourceNode",
                "execution_count": self._mon.node_counts[node_name],
            },
        )

    def _infer_batch_size(self, batch: Any) -> int:
        """Infer batch size from batch containers used in datarax pipelines."""
        if hasattr(batch, "batch_size"):
            return int(batch.batch_size)
        if isinstance(batch, dict):
            return len(next(iter(batch.values())))
        if hasattr(batch, "shape"):
            return int(batch.shape[0])
        return 1

    def _record_batch_iteration_metrics(self, batch: Any, batch_start: float | None) -> None:
        """Record per-batch pipeline metrics."""
        if not self.metrics.enabled:
            return
        self._batch_count += 1
        batch_size = self._infer_batch_size(batch)

        self.metrics.record_metric(
            "batch_size",
            batch_size,
            component="pipeline",
            metadata={"batch_num": self._batch_count},
        )

        current_batches = self.total_batches_processed.get_value()
        self.total_batches_processed.set_value(current_batches + 1)
        current_elements = self.total_elements_processed.get_value()
        self.total_elements_processed.set_value(current_elements + batch_size)

        if self.track_memory:
            self._record_memory_usage()
        if batch_start is not None:
            batch_time = time.time() - batch_start
            self._mon.batch_times.append(batch_time)
            self.metrics.record_metric(
                "batch_processing_time",
                batch_time,
                component="pipeline",
                metadata={"batch_num": self._batch_count},
            )

        self._mon.notify_counter += 1
        if self._mon.notify_counter >= self._mon.notify_threshold:
            self._notify_observers()
            self._mon.notify_counter = 0

    def _execute(self, node: Node, data: Any, key: jax.Array | None = None) -> Any:
        """Execute node with metrics collection.

        Args:
            node: Node to execute
            data: Input data
            key: Optional RNG key

        Returns:
            Processed data
        """
        if not self.metrics.enabled:
            return super()._execute(node, data, key)

        # Import node types
        from datarax.dag.nodes import Sequential, Parallel, BatchNode, OperatorNode

        if isinstance(node, Sequential | Parallel):
            node_name, result = self._execute_composite_node(
                node, data, key, BatchNode, OperatorNode
            )
        else:
            node_name, result = self._execute_standard_node(node, data, key)
        self._record_data_flow_metrics(node_name, result)
        return result

    def _execute_composite_node(
        self,
        node: Node,
        data: Any,
        key: jax.Array | None,
        batch_node_type: type,
        operator_node_type: type,
    ) -> tuple[str, Any]:
        """Execute a composite node and record container + child placeholder metrics."""
        from datarax.dag.nodes import Sequential

        node_name = type(node).__name__
        start_time = time.time()
        result = super()._execute(node, data, key)
        self._record_node_execution_metric(node_name, node, time.time() - start_time)

        if isinstance(node, Sequential):
            self._record_sequential_child_placeholders(
                node, node_name, batch_node_type, operator_node_type
            )
        return node_name, result

    def _execute_standard_node(
        self, node: Node, data: Any, key: jax.Array | None
    ) -> tuple[str, Any]:
        """Execute a regular node and record execution metric."""
        node_name = node.name if hasattr(node, "name") else type(node).__name__
        start_time = time.time()
        result = super()._execute(node, data, key)
        self._record_node_execution_metric(node_name, node, time.time() - start_time)
        return node_name, result

    def _record_node_execution_metric(
        self, node_name: str, node: Node, execution_time: float
    ) -> None:
        """Record execution time metric and update node counters."""
        if node_name not in self._mon.node_timers:
            self._mon.node_timers[node_name] = []
            self._mon.node_counts[node_name] = 0
        self._mon.node_timers[node_name].append(execution_time)
        self._mon.node_counts[node_name] += 1
        self.metrics.record_metric(
            f"node_{node_name}_execution_time",
            execution_time,
            component="nodes",
            metadata={
                "node_type": type(node).__name__,
                "execution_count": self._mon.node_counts[node_name],
            },
        )

    def _record_sequential_child_placeholders(
        self,
        node: Any,
        parent_name: str,
        batch_node_type: type,
        operator_node_type: type,
    ) -> None:
        """Record placeholder child metrics for sequential containers."""
        for child_node in node.nodes:
            child_type = type(child_node).__name__
            child_name = self._child_metric_name(
                child_node, child_type, batch_node_type, operator_node_type
            )
            self.metrics.record_metric(
                f"node_{child_name}_execution_time",
                0.0,
                component="nodes",
                metadata={
                    "node_type": child_type,
                    "parent": parent_name,
                    "placeholder": True,
                },
            )

    def _child_metric_name(
        self, child_node: Any, child_type: str, batch_node_type: type, operator_node_type: type
    ) -> str:
        """Map node types to stable metric names."""
        if isinstance(child_node, batch_node_type):
            return "Batch"
        if isinstance(child_node, operator_node_type):
            if hasattr(child_node, "name") and child_node.name:
                return child_node.name
            return "Operator"
        return child_type

    def _record_data_flow_metrics(self, node_name: str, result: Any) -> None:
        """Record output shape metrics for data-flow tracking."""
        if result is None:
            return
        if isinstance(result, dict):
            for field_name, value in result.items():
                if not hasattr(value, "shape"):
                    continue
                self.metrics.record_metric(
                    f"node_{node_name}_output_shape",
                    value.shape[0] if len(value.shape) > 0 else 1,
                    component="data_flow",
                    metadata={"field": field_name, "shape": str(value.shape)},
                )
            return
        if hasattr(result, "shape"):
            self.metrics.record_metric(
                f"node_{node_name}_output_shape",
                result.shape[0] if len(result.shape) > 0 else 1,
                component="data_flow",
                metadata={"shape": str(result.shape)},
            )

    def _record_memory_usage(self) -> None:
        """Record current memory usage."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            # Record memory in MB
            self.metrics.record_metric(
                "memory_rss_mb", memory_info.rss / 1024 / 1024, component="system"
            )

            self.metrics.record_metric(
                "memory_vms_mb", memory_info.vms / 1024 / 1024, component="system"
            )

        except ImportError:
            # psutil not available
            pass

    def _notify_observers(self) -> None:
        """Notify all registered observers with current metrics."""
        if self.callbacks and self.metrics.enabled:
            metrics = self.metrics.get_metrics()

            # Add summary statistics
            summary = self._compute_summary_statistics()
            for key, value in summary.items():
                metrics.append(
                    MetricRecord(
                        name=key,
                        value=value,
                        component="summary",
                        timestamp=time.time(),
                        metadata={},
                    )
                )

            self.callbacks.notify(metrics)

    def _compute_summary_statistics(self) -> dict[str, float]:
        """Compute summary statistics for the pipeline.

        Returns:
            Dictionary of summary statistics
        """
        stats = {}

        # Throughput
        total_time = self.total_processing_time.get_value()
        if total_time > 0:
            stats["throughput_batches_per_sec"] = (
                self.total_batches_processed.get_value() / total_time
            )
            stats["throughput_elements_per_sec"] = (
                self.total_elements_processed.get_value() / total_time
            )

        # Batch timing statistics
        if self._mon.batch_times:
            stats["avg_batch_time"] = sum(self._mon.batch_times) / len(self._mon.batch_times)
            stats["min_batch_time"] = min(self._mon.batch_times)
            stats["max_batch_time"] = max(self._mon.batch_times)

        # Node statistics
        for node_name, times in self._mon.node_timers.items():
            if times:
                stats[f"node_{node_name}_avg_time"] = sum(times) / len(times)
                stats[f"node_{node_name}_total_time"] = sum(times)
                stats[f"node_{node_name}_count"] = self._mon.node_counts[node_name]

        # Cache statistics
        if self.enable_caching and self._cache is not None:
            stats["cache_size"] = len(self._cache)

            # Check for cache nodes in graph
            cache_stats = self._get_cache_node_stats(self.graph)
            stats.update(cache_stats)

        return stats

    def _get_cache_node_stats(self, node: Node, prefix: str = "") -> dict[str, float]:
        """Recursively get cache statistics from nodes.

        Args:
            node: Node to check
            prefix: Prefix for metric names

        Returns:
            Cache statistics dictionary
        """
        stats = {}

        from datarax.dag.nodes import CacheNode, Sequential, Parallel

        if isinstance(node, CacheNode):
            cache_stats = node.get_stats()
            for key, value in cache_stats.items():
                stats[f"{prefix}cache_{key}"] = value

        elif isinstance(node, Sequential | Parallel):
            for i, child in enumerate(node.nodes):
                child_prefix = f"{prefix}node{i}_"
                child_stats = self._get_cache_node_stats(child, child_prefix)
                stats.update(child_stats)

        return stats

    def _finalize_metrics(self) -> None:
        """Finalize metrics collection at end of iteration."""
        if self._mon.pipeline_start_time:
            total_time = time.time() - self._mon.pipeline_start_time
            self.total_processing_time.set_value(total_time)

            self.metrics.stop_timer("pipeline_iteration", "pipeline")

            # Record final statistics
            self.metrics.record_metric(
                "total_batches", self.total_batches_processed.get_value(), component="pipeline"
            )

            self.metrics.record_metric(
                "total_elements", self.total_elements_processed.get_value(), component="pipeline"
            )

            self.metrics.record_metric("total_time", total_time, component="pipeline")

            # Final notification
            self._notify_observers()

    def get_performance_report(self) -> dict[str, Any]:
        """Generate performance report.

        Returns:
            Dictionary containing performance metrics
        """
        report = {
            "pipeline_name": self.name,
            "total_batches": int(self.total_batches_processed.get_value()),
            "total_elements": int(self.total_elements_processed.get_value()),
            "total_time": float(self.total_processing_time.get_value()),
            "configuration": {
                "enforce_batch": self.enforce_batch,
                "enable_caching": self.enable_caching,
                "jit_compile": self.jit_compile,
                "metrics_enabled": self.metrics.enabled,
                "track_memory": self.track_memory,
            },
        }

        # Add summary statistics
        report["statistics"] = self._compute_summary_statistics()

        # Add node breakdown
        report["nodes"] = {}
        for node_name in self._mon.node_timers:
            report["nodes"][node_name] = {
                "execution_count": self._mon.node_counts[node_name],
                "total_time": sum(self._mon.node_timers[node_name]),
                "avg_time": sum(self._mon.node_timers[node_name])
                / len(self._mon.node_timers[node_name]),
            }

        return report

    def reset_metrics(self) -> None:
        """Reset all metrics and counters."""
        self.metrics.clear()
        self._mon.node_timers.clear()
        self._mon.node_counts.clear()
        self._mon.batch_times.clear()
        self._mon.notify_counter = 0
        self.total_batches_processed.set_value(0)
        self.total_elements_processed.set_value(0)
        self.total_processing_time.set_value(0.0)

    def __repr__(self) -> str:
        """String representation with monitoring info."""
        base = super().__repr__()
        return base.replace(
            "DAGExecutor",
            f"MonitoredDAGExecutor(metrics={'on' if self.metrics.enabled else 'off'})",
        )


# Convenience function for creating monitored pipelines
def monitored_pipeline(*nodes: Node, **kwargs: Any) -> MonitoredDAGExecutor:
    """Create a monitored pipeline from nodes.

    Args:
        nodes: Nodes to add to pipeline
        **kwargs: Additional arguments for MonitoredDAGExecutor

    Returns:
        MonitoredDAGExecutor instance
    """
    executor = MonitoredDAGExecutor(**kwargs)
    for node in nodes:
        executor.add(node)
    return executor
