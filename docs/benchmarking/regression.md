# Regression Testing

Detect performance regressions over time.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Comparative](comparative.md) - Compare configurations
- [Testing Guide](../contributing/testing_guide.md) - Test infrastructure
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

The `detect_regressions()` function compares a current `Run` against a baseline `Run` and flags metrics that degraded beyond a configurable threshold (default 5%). Detection is **direction-aware**: throughput decreases are regressions, but latency decreases are improvements. Metrics with direction `info` are skipped.

Points are matched between runs using a composite key of `(name, tags)`, ensuring that "CV-1/small for Datarax" is compared against the correct baseline even when multiple frameworks share the same point name.

## Quick Start

```python
from calibrax.analysis import detect_regressions
from calibrax.core import Metric, MetricDef, MetricDirection, Point, Run

baseline = Run(
    points=(
        Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax"},
            metrics={"throughput": Metric(value=20000.0)},
        ),
    ),
    metric_defs={
        "throughput": MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
        ),
    },
)

current = Run(
    points=(
        Point(
            name="CV-1/small",
            scenario="CV-1",
            tags={"framework": "Datarax"},
            metrics={"throughput": Metric(value=18000.0)},
        ),
    ),
    metric_defs=baseline.metric_defs,
)

regressions = detect_regressions(current, baseline, threshold=0.05)
for r in regressions:
    print(f"  {r.metric} on {r.point_name}: {r.delta_pct:+.1f}%")
    print(f"    baseline={r.baseline_value:.0f} -> current={r.current_value:.0f}")
```

### CI Integration

The `calibrax check` CLI command wraps `detect_regressions()` for CI pipelines:

```bash
calibrax check --data benchmark-data/ --threshold 0.05
```

Exits with code 1 if any regressions exceed the threshold.

---

::: calibrax.analysis.regression
