"""Pipeline integration for the Datarax monitoring system.

This module provides a MonitoredPipeline class that integrates metrics collection
with the Datarax pipeline.
"""

from typing import Any, Self
from collections.abc import Callable, Iterator

import flax.nnx as nnx

from datarax.core.data_source import DataSourceModule
from datarax.dag.dag_executor import DAGExecutor
from datarax.monitoring.callbacks import CallbackRegistry
from datarax.monitoring.metrics import MetricsCollector
from datarax.typing import Batch, Element


class MonitoredPipeline(DAGExecutor):
    """Pipeline with integrated metrics collection.

    This class extends the standard Pipeline with metrics collection capabilities.

    Attributes:
        metrics: MetricsCollector instance for collecting metrics.
        callbacks: CallbackRegistry for observers that consume metrics.
    """

    def __init__(
        self,
        source: DataSourceModule,
        rngs: nnx.Rngs | None = None,
        metrics_enabled: bool = True,
        **kwargs: Any,
    ):
        """Initialize a MonitoredPipeline.

        Args:
            source: The data source providing the raw data elements.
            rngs: Optional RNG streams for randomness.
            metrics_enabled: Whether to enable metrics collection.
            **kwargs: Keyword arguments to pass to the DAGExecutor constructor.
        """
        # Initialize metrics first
        self.metrics = MetricsCollector(enabled=metrics_enabled)
        self.callbacks = CallbackRegistry()
        self._notify_counter = 0
        self._notify_threshold = 100  # Notify observers after this many metrics

        # Initialize DAGExecutor with proper signature
        super().__init__(rngs=rngs, **kwargs)

        # Add the source to the pipeline
        self.add(source)

    def __iter__(self) -> Iterator[Batch]:
        """Create an iterator with metrics collection.

        Returns:
            An iterator that yields batches while collecting metrics.
        """
        # Reset iteration tracking
        self._iteration_count = 0
        self._epoch_count = 0

        # Fast path: if metrics disabled, use parent's iterator directly
        # This avoids any overhead from generator wrapping
        if not self.metrics.enabled:
            return super().__iter__()

        # Start measuring pipeline iteration time
        self.metrics.start_timer("pipeline_iteration", "pipeline")

        # Get the actual iterator from parent class by calling _create_iterator directly
        base_iterator = self._create_iterator()

        # Return a wrapped iterator that collects metrics
        return self._wrap_iterator_with_metrics(base_iterator)

    def _wrap_iterator_with_metrics(self, iterator: Iterator[Batch]) -> Iterator[Batch]:
        """Wrap an iterator with metrics collection.

        Args:
            iterator: Base iterator to wrap.

        Returns:
            A wrapped iterator that collects metrics.
        """
        if not self.metrics.enabled:
            # If metrics are disabled, return the original iterator
            return iterator

        # Otherwise, wrap the iterator with metrics collection
        def metrics_iterator():
            try:
                # Time each batch production
                batch_count = 0

                for batch in iterator:
                    # Record batch production
                    batch_count += 1
                    self.metrics.record_metric(
                        "batch_produced",
                        1,
                        "pipeline",
                        {"batch_size": len(batch) if hasattr(batch, "__len__") else None},
                    )

                    # Time batch production
                    self.metrics.start_timer("batch_production", "pipeline")

                    # Notify observers periodically
                    self._notify_counter += 1
                    if self._notify_counter >= self._notify_threshold:
                        self.callbacks.notify(self.metrics.get_metrics())
                        self.metrics.clear()
                        self._notify_counter = 0

                    # Stop timing the previous batch production
                    self.metrics.stop_timer("batch_production", "pipeline")

                    # Yield the batch
                    yield batch

                # Record pipeline completion
                self.metrics.record_metric(
                    "pipeline_completed", 1, "pipeline", {"total_batches": batch_count}
                )

            finally:
                # Stop timing pipeline iteration
                self.metrics.stop_timer("pipeline_iteration", "pipeline")

                # Final notification
                if self._notify_counter > 0:
                    self.callbacks.notify(self.metrics.get_metrics())
                    self.metrics.clear()
                    self._notify_counter = 0

        return metrics_iterator()

    def add(self, node: Any, **kwargs: Any) -> Self:
        """Add a node to the pipeline with metrics collection.

        This overrides the add method to collect metrics about node operations.

        Args:
            node: The node to add.
            **kwargs: Additional arguments for the add operation.

        Returns:
            The updated pipeline.
        """
        # Record the node type
        if self.metrics.enabled:
            self.metrics.record_metric(
                "node_added", 1, "pipeline", {"node_type": type(node).__name__}
            )

        # Call the parent implementation
        return super().add(node, **kwargs)

    def filter(self, predicate: Callable[[Element], bool], **kwargs: Any) -> Self:
        """Apply a filter to the pipeline with metrics collection.

        This overrides the filter method to collect metrics about filter operations.
        Since Pipeline doesn't have a filter method, we implement it here.

        Args:
            predicate: The predicate function to apply.
            **kwargs: Additional arguments for the filter operation.

        Returns:
            The updated pipeline.
        """
        # Record the filter addition
        if self.metrics.enabled:
            self.metrics.record_metric(
                "filter_added",
                1,
                "pipeline",
                {"filter_type": getattr(predicate, "__name__", type(predicate).__name__)},
            )

        # Create a filter operator using unified operator architecture
        from datarax.operators import ElementOperator, ElementOperatorConfig
        from datarax.dag.nodes import OperatorNode

        def filter_fn(element: Element) -> Element | None:
            """Filter elements based on predicate."""
            if predicate(element):
                return element
            return None

        config = ElementOperatorConfig(stochastic=False, stream_name="filter")
        filter_operator = ElementOperator(config, fn=filter_fn)
        return self.add(OperatorNode(filter_operator), **kwargs)

    def batch(self, batch_size: int, drop_remainder: bool = False, **kwargs: Any) -> Self:
        """Create batches with metrics collection.

        This method adds a BatchNode to collect metrics about batch operations.

        Args:
            batch_size: Size of the batches to create.
            drop_remainder: Whether to drop incomplete batches.
            **kwargs: Additional arguments for the add operation.

        Returns:
            The updated pipeline.
        """
        # Record the batching operation
        if self.metrics.enabled:
            self.metrics.record_metric("batching_added", 1, "pipeline", {"batch_size": batch_size})

        # Add BatchNode
        from datarax.dag.nodes import BatchNode

        return self.add(BatchNode(batch_size=batch_size, drop_remainder=drop_remainder), **kwargs)
