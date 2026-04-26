# Comparative Benchmarking

Compare performance across configurations or versions.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Regression Testing](regression.md) - Track performance changes
- [Profiler](profiler.md) - Detailed profiling
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

calibrax provides comparative analysis through `Run` objects containing multiple `Point` entries (one per framework/configuration). The `rank_table()` function ranks entries by any metric with direction-aware sorting, while `compare_configurations()` produces a full comparison report between two runs.

## Quick Start

```python
from calibrax.analysis import rank_table, compare_configurations
from calibrax.core import Metric, MetricDef, MetricDirection, Point, Run

# Build a run with results from multiple configurations
run = Run(
    points=(
        Point(
            name="CV-1/baseline",
            scenario="CV-1",
            tags={"framework": "baseline"},
            metrics={"throughput": Metric(value=15000.0)},
        ),
        Point(
            name="CV-1/optimized",
            scenario="CV-1",
            tags={"framework": "optimized"},
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

# Rank by throughput (direction-aware: higher is better)
rankings = rank_table(run, "throughput")
for row in rankings:
    marker = " (best)" if row.is_best else ""
    print(f"  {row.rank}. {row.label}: {row.value:.0f} elem/s{marker}")
```

---

::: calibrax.analysis
