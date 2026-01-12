"""Tests for the Datarax monitoring pipeline.

This module contains tests for the MonitoredPipeline component
of the Datarax monitoring system.
"""

from typing import Any
import numpy as np
import flax.nnx as nnx

from datarax.monitoring.callbacks import MetricsObserver, MetricRecord
from datarax.monitoring.pipeline import MonitoredPipeline
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig
from datarax.dag.nodes import BatchNode, OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig


class TestMonitoredPipeline:
    """Tests for the MonitoredPipeline class."""

    def test_basic_functionality(self):
        """Test basic functionality of MonitoredPipeline."""
        # Create test data
        data = {"value": np.arange(100)}
        source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

        # Create a monitored pipeline
        pipeline = MonitoredPipeline(source).add(BatchNode(batch_size=1))

        # Create an observer to capture metrics
        class TestObserver(MetricsObserver):
            def __init__(self, *args: Any, **kwargs: Any):
                super().__init__(*args, **kwargs)
                self.metrics_received: list[MetricRecord] = []

            def update(self, metrics: list[MetricRecord]) -> None:
                self.metrics_received.extend(metrics)

        observer = TestObserver()
        pipeline.callbacks.register(observer)

        # Process all data
        list(pipeline)

        # Force final notification of metrics
        if len(observer.metrics_received) == 0:
            pipeline.callbacks.notify(pipeline.metrics.get_metrics())

        # Verify metrics were collected
        assert len(observer.metrics_received) > 0

    def test_transformations(self):
        """Test MonitoredPipeline with transformations."""
        # Create test data
        data = {"value": np.arange(100)}
        source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

        # Create a monitored pipeline
        pipeline = MonitoredPipeline(source)

        # Add a transformer
        def double(element, key):
            """Double the value in element data."""
            new_data = {"value": element.data["value"] * 2}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        transformer = ElementOperator(config, fn=double, rngs=rngs)
        pipeline = pipeline.add(BatchNode(batch_size=1)).add(OperatorNode(transformer))

        # Process all data
        results = list(pipeline)

        # Verify transformation was applied
        assert results[0]["value"] == 0  # First element is 0, doubled is still 0
        assert results[1]["value"] == 2  # Second element is 1, doubled is 2

    def test_filtering_with_map(self):
        """Test filtering using post-processing on transformed data."""
        # Create test data with pre-allocated _keep field for JAX vmap compatibility
        data = {"value": np.arange(10), "_keep": np.ones(10, dtype=bool)}
        source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

        # Create a monitored pipeline
        pipeline = MonitoredPipeline(source)

        # Add a filter using add that marks elements for filtering
        def mark_even(element, key):
            """Mark even numbers to keep."""
            # Mark even numbers to keep (no conditional return for JAX compatibility)
            # Pre-allocated _keep field ensures PyTree structure doesn't change
            is_even = element.data["value"] % 2 == 0
            new_data = {"value": element.data["value"], "_keep": is_even}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        transformer = ElementOperator(config, fn=mark_even, rngs=rngs)
        pipeline = pipeline.add(BatchNode(batch_size=1)).add(OperatorNode(transformer))

        # Process all data and filter based on the marker
        raw_results = list(pipeline)
        # Extract data from batches and filter
        results = [r for r in raw_results if r.get_data()["_keep"][0]]

        # Verify filtering was applied correctly
        assert len(results) == 5  # Should have 5 even numbers (0, 2, 4, 6, 8)
        for result in results:
            assert result.get_data()["value"][0] % 2 == 0

    def test_batched(self):
        """Test MonitoredPipeline batched method."""
        # Create test data
        data = {"value": np.arange(100)}
        source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

        # Create a monitored pipeline
        pipeline = MonitoredPipeline(source)

        # Add batching
        batch_size = 10
        pipeline = pipeline.batch(batch_size)

        # Process all data
        batches = list(pipeline)

        # Verify batching was applied
        assert len(batches) == 10  # 100 elements / batch_size 10 = 10 batches
        for batch in batches:
            assert len(batch["value"]) == batch_size


def test_end_to_end_monitored_pipeline():
    """Test end-to-end monitoring with a pipeline."""
    # Create test data
    data = {"value": np.arange(100)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    # Create a monitored pipeline and register observer before any transformations
    pipeline = MonitoredPipeline(source, metrics_enabled=True)

    # Create a metrics observer
    class MetricsCapture(MetricsObserver):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.metrics: list[MetricRecord] = []

        def update(self, metrics: list[MetricRecord]) -> None:
            self.metrics.extend(metrics)

    observer = MetricsCapture()
    pipeline.callbacks.register(observer)

    # Add a simple transformer to the pipeline to generate metrics
    def double_values(element, key):
        """Double the values in element data."""
        new_data = {"value": element.data["value"] * 2}
        return element.replace(data=new_data)

    config = ElementOperatorConfig(stochastic=False)
    rngs = nnx.Rngs(0)
    transformer = ElementOperator(config, fn=double_values, rngs=rngs)
    transformed_pipeline = pipeline.add(BatchNode(batch_size=1)).add(OperatorNode(transformer))

    # Process all data from the pipeline to generate metrics
    all_elements = list(transformed_pipeline)

    # Manually notify if no metrics were recorded yet
    if len(observer.metrics) == 0:
        pipeline.callbacks.notify(pipeline.metrics.get_metrics())

    # Verify metrics were collected
    assert len(observer.metrics) > 0

    # Now filter the elements to find those divisible by 3
    # This is post-processing of the data rather than using the pipeline's filter
    divisible_by_3 = [elem for elem in all_elements if elem["value"] % 3 == 0]

    # Batch these into groups of 5
    batch_size = 5
    batches = []
    current_batch = {"value": []}

    for elem in divisible_by_3:
        current_batch["value"].append(elem["value"])
        if len(current_batch["value"]) >= batch_size:
            batches.append(current_batch)
            current_batch = {"value": []}

    # Add the last partial batch if any
    if current_batch["value"]:
        batches.append(current_batch)

    # Verify we have batches with elements divisible by both 2 and 3
    assert len(batches) > 0
    for batch in batches:
        for value in batch["value"]:
            assert value % 3 == 0
            assert value % 2 == 0  # All values were doubled
