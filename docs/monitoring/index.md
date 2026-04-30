# Monitoring

Metrics collection and observability for data pipelines. Track throughput, latency, and custom metrics during training and inference.

## Components

| Component | Purpose | Use Case |
|-----------|---------|----------|
| **Metrics** | Collect measurements | Throughput, latency, custom |
| **Callbacks** | Event hooks | Log on batch/epoch end |
| **Reporters** | Output backends | Console, file, TensorBoard |

`★ Insight ─────────────────────────────────────`

- Metrics are lightweight - minimal overhead
- Use callbacks for automated logging
- Reporters support multiple backends simultaneously

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.monitoring import MetricsCollector

collector = MetricsCollector()

# Track metrics during training
for step, batch in enumerate(pipeline):
    loss = train_step(batch)
    collector.record_metric("loss", float(loss))
    collector.record_metric("throughput", batch_size / elapsed)

# Inspect collected records
for record in collector.get_metrics():
    print(record.name, record.value)
```

## Modules

- [metrics](metrics.md) - Core metrics collection and storage
- [callbacks](callbacks.md) - Event-driven monitoring callbacks
- [reporters](reporters.md) - Output backends (console, file, etc.)

## Callback Example

```python
from datarax.monitoring import (
    CallbackRegistry,
    ConsoleReporter,
    MetricsCollector,
    MetricsObserver,
)
from datarax.monitoring.metrics import MetricRecord

collector = MetricsCollector()
registry = CallbackRegistry()
registry.register(ConsoleReporter(report_interval=1.0))


class LossPrinter(MetricsObserver):
    def update(self, metrics: list[MetricRecord]) -> None:
        for record in metrics:
            if record.name == "loss":
                print(f"step {record.metadata.get('step')}: loss = {record.value:.4f}")


registry.register(LossPrinter())

for step, batch in enumerate(pipeline):
    loss = train_step(batch)
    collector.record_metric("loss", float(loss), metadata={"step": step})
    registry.notify(collector.get_metrics())
```

## See Also

- [Benchmarking](../benchmarking/index.md) - Performance measurement
