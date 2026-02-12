# Benchmark Monitor

Real-time pipeline monitoring with GPU profiling and alerting.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Monitoring](../monitoring/index.md) - General monitoring
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

This module provides three monitoring components:

- **AdvancedMonitor** — Real-time pipeline monitoring with configurable thresholds, background metric collection (CPU, memory, GPU), and trend analysis. Integrates with `AlertManager` for threshold-based alerts.
- **ProductionMonitor** — Extends `AdvancedMonitor` with production-specific features: performance baseline tracking, per-pipeline execution recording, error rate monitoring, and health reports.
- **AlertManager** — Manages threshold-based alerts with configurable severity levels and pluggable alert handlers.

## Quick Start

```python
from datarax.benchmarking import AdvancedMonitor

monitor = AdvancedMonitor()

# Configure thresholds
monitor.set_threshold("gpu_memory_utilization", 0.85)
monitor.set_threshold("memory_usage_mb", 4000)

# Start background monitoring (collects metrics every 5 seconds)
monitor.start_monitoring(interval=5.0)

# ... run your pipeline ...

# Stop and check results
monitor.stop_monitoring()
summary = monitor.get_monitoring_summary()
print(f"Alerts triggered: {summary['total_alerts']}")
print(f"Critical alerts: {summary['critical_alerts']}")
```

---

::: datarax.benchmarking.monitor
