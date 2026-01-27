# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Monitoring Quick Reference

| Metadata | Value |
|----------|-------|
| **Level** | Beginner |
| **Runtime** | ~5 min |
| **Prerequisites** | Basic Python, Simple Pipeline |
| **Format** | Python + Jupyter |

## Overview

This quick reference demonstrates Datarax's built-in monitoring system for
tracking pipeline metrics. You'll learn to use `MonitoredPipeline` with
`ConsoleReporter` to observe throughput and custom metrics in real-time.

## Learning Goals

By the end of this example, you will be able to:

1. Create a `MonitoredPipeline` that collects metrics automatically
2. Register a `ConsoleReporter` for real-time metric display
3. Record custom metrics during pipeline iteration
4. Understand the metrics collection architecture
"""

# %% [markdown]
"""
## Setup

```bash
# Install datarax
uv pip install datarax
```
"""

# %%
# Imports
import time

import jax.numpy as jnp
from flax import nnx

from datarax.dag.nodes import BatchNode, OperatorNode
from datarax.monitoring.pipeline import MonitoredPipeline
from datarax.monitoring.reporters import ConsoleReporter
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources.memory_source import MemorySource, MemorySourceConfig

# %% [markdown]
"""
## Key Concepts

### MonitoredPipeline

`MonitoredPipeline` extends the standard DAG executor with metrics collection:

- Automatically tracks batch production
- Records timing information
- Provides a `metrics` object for custom metric recording
- Supports callback observers for reporting

### Reporters

Reporters consume collected metrics:

| Reporter | Description |
|----------|-------------|
| `ConsoleReporter` | Prints metrics to console at intervals |
| `FileReporter` | Writes metrics to a file |

### Metric Types

| Metric | Description |
|--------|-------------|
| `batch_produced` | Count of batches yielded |
| `pipeline_iteration` | Total iteration time |
| `batch_production` | Per-batch production time |
| Custom metrics | User-defined via `record_metric()` |
"""

# %% [markdown]
"""
## Step 1: Create Data Source

Start with a simple in-memory data source.
"""

# %%
# Create sample data
data = [{"value": i, "label": i % 5} for i in range(200)]

source_config = MemorySourceConfig()
source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

print(f"Data source: {len(data)} samples")

# %% [markdown]
"""
## Step 2: Create Monitored Pipeline

Use `MonitoredPipeline` instead of the standard `from_source()` API
when you need metrics collection.
"""

# %%
# Create monitored pipeline
pipeline = MonitoredPipeline(source, metrics_enabled=True)

# Register console reporter (reports every 1 second for demo)
reporter = ConsoleReporter(report_interval=1.0)
pipeline.callbacks.register(reporter)

print("Created MonitoredPipeline with ConsoleReporter")

# %% [markdown]
"""
## Step 3: Add Pipeline Stages

Add batching and transformations as usual.
"""


# %%
# Define a simple transformation
def double_value(element, key=None):  # noqa: ARG001
    """Double the value field."""
    del key  # Unused - deterministic operator
    result = dict(element.data)
    result["value"] = result["value"] * 2
    return element.update_data(result)


# Add batching (required before operators)
pipeline.add(BatchNode(batch_size=32))

# Add transformation operator
double_op = ElementOperator(
    ElementOperatorConfig(stochastic=False),
    fn=double_value,
    rngs=nnx.Rngs(0),
)
pipeline.add(OperatorNode(double_op))

print("Pipeline: Source -> Batch(32) -> DoubleValue -> Output")

# %% [markdown]
"""
## Step 4: Process with Custom Metrics

Iterate through the pipeline and record custom metrics.
"""

# %%
# Process data with custom metrics
print("\nProcessing data with monitoring...")
print("(Metrics report will appear periodically)")
print()

batch_count = 0
total_samples = 0

for batch in pipeline:
    batch_count += 1

    # Get batch data
    values = batch["value"]
    labels = batch["label"]
    batch_size = values.shape[0]
    total_samples += batch_size

    # Record custom metrics
    if pipeline.metrics.enabled:
        # Record batch statistics
        pipeline.metrics.record_metric(
            "batch_mean_value",
            float(jnp.mean(values)),
            "custom",
        )
        pipeline.metrics.record_metric(
            "batch_max_value",
            float(jnp.max(values)),
            "custom",
        )

    # Simulate some processing time
    time.sleep(0.05)

    # Print progress every 2 batches
    if batch_count % 2 == 0:
        mean_val = float(jnp.mean(values))
        print(f"Batch {batch_count}: {batch_size} samples, mean={mean_val:.1f}")

print(f"\nCompleted: {batch_count} batches, {total_samples} samples")

# %% [markdown]
"""
## Results Summary

| Component | Value |
|-----------|-------|
| Data Source | MemorySource (200 samples) |
| Batch Size | 32 |
| Transformation | Double value field |
| Reporter | ConsoleReporter (1s interval) |
| Custom Metrics | batch_mean_value, batch_max_value |

The monitoring system automatically tracks:

- Number of batches produced
- Pipeline iteration timing
- Node additions to the pipeline
- Custom metrics you record explicitly
"""

# %% [markdown]
"""
## Next Steps

- **File output**: Use `FileReporter` to persist metrics to disk
- **Custom reporters**: Implement `MetricsObserver` for custom destinations
- **Distributed metrics**: [Distributed](../distributed/01_sharding_quickref.ipynb)
- **API Reference**: [Monitoring module](https://datarax.readthedocs.io/monitoring/)
"""


# %%
def main():
    """Run the monitoring quick reference example."""
    print("Monitoring Quick Reference")
    print("=" * 50)

    # Create data and source
    data = [{"value": i, "label": i % 5} for i in range(100)]
    source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

    # Create monitored pipeline
    pipeline = MonitoredPipeline(source, metrics_enabled=True)
    reporter = ConsoleReporter(report_interval=0.5)
    pipeline.callbacks.register(reporter)

    # Define transformation
    def double_value(element, key=None):  # noqa: ARG001
        del key
        result = dict(element.data)
        result["value"] = result["value"] * 2
        return element.update_data(result)

    # Build pipeline
    pipeline.add(BatchNode(batch_size=16))
    double_op = ElementOperator(
        ElementOperatorConfig(stochastic=False),
        fn=double_value,
        rngs=nnx.Rngs(0),
    )
    pipeline.add(OperatorNode(double_op))

    # Process
    total = 0
    for batch in pipeline:
        total += batch["value"].shape[0]
        # Record custom metric
        pipeline.metrics.record_metric("batch_sum", float(jnp.sum(batch["value"])), "example")
        time.sleep(0.02)

    print(f"\nProcessed {total} samples with monitoring!")


if __name__ == "__main__":
    main()
