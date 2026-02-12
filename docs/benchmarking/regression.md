# Regression Testing

Detect performance regressions over time.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Comparative](comparative.md) - Compare configurations
- [Testing Guide](../contributing/testing_guide.md) - Test infrastructure
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

`RegressionDetector` compares current benchmark results against historical baselines using statistical analysis. It uses Welch's t-test when enough samples are available (per Section 9.2 of the benchmark report) and falls back to percentage thresholds for small sample sizes. Configurable thresholds control regression (default 10%) and critical (default 25%) severity levels.

The detector is **direction-aware**: throughput decreases are regressions, but latency decreases are improvements.

## Quick Start

```python
from datarax.benchmarking import RegressionDetector, BenchmarkResult

detector = RegressionDetector(baseline_dir="baselines/")

# Add historical baselines (run this multiple times over different runs)
detector.add_baseline(baseline_result, "my_pipeline")

# Later: detect regressions against baselines
report = detector.detect_regressions(current_result, "my_pipeline")

if report.has_critical_regressions():
    print("CRITICAL regressions detected!")
    worst = report.get_worst_regression()
    print(f"  {worst.description}")

for regression in report.regressions:
    print(f"  [{regression.severity.value}] {regression.description} (p={regression.p_value})")
```

---

::: datarax.benchmarking.regression
