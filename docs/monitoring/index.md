# Monitoring

Metrics collection and observability for data pipelines. Track throughput, latency, and custom metrics during training and inference.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Metrics** | Collect measurements | Throughput, latency, custom |
| **Callbacks** | Event hooks | Log on batch/epoch end |
| **Reporters** | Output backends | Console, file, TensorBoard |
| **DAG Monitor** | Pipeline visualization | Debug DAG execution |

`★ Insight ─────────────────────────────────────`

- Metrics are lightweight - minimal overhead
- Use callbacks for automated logging
- Reporters support multiple backends simultaneously
- DAG monitor helps debug pipeline structure

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.monitoring import MetricsCollector, ConsoleReporter

# Create collector with reporter
collector = MetricsCollector()
collector.add_reporter(ConsoleReporter())

# Track metrics during training
for step, batch in enumerate(pipeline):
    loss = train_step(batch)
    collector.record("loss", loss)
    collector.record("throughput", batch_size / elapsed)

# Print summary
collector.summary()
```

## Modules

- [metrics](metrics.md) - Core metrics collection and storage
- [callbacks](callbacks.md) - Event-driven monitoring callbacks
- [reporters](reporters.md) - Output backends (console, file, etc.)
- [dag_monitor](dag_monitor.md) - DAG-specific monitoring and visualization
- [pipeline](pipeline.md) - Pipeline-level monitoring integration

## Callback Example

```python
from datarax.monitoring import LoggingCallback

callback = LoggingCallback(
    log_every=100,  # Log every 100 batches
    metrics=["loss", "accuracy"],
)

for step, batch in enumerate(pipeline):
    loss = train_step(batch)
    callback.on_batch_end(step, {"loss": loss})
```

## See Also

- [Benchmarking](../benchmarking/index.md) - Performance measurement
- [Monitoring Tutorial](../examples/advanced/monitoring/monitoring-quickref.md)
