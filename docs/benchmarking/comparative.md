# Comparative Benchmarking

Compare performance across configurations or versions.

## See Also

- [Benchmarking Overview](index.md) - All benchmarking tools
- [Regression Testing](regression.md) - Track performance changes
- [Profiler](profiler.md) - Detailed profiling
- [Benchmarking Guide](../user_guide/benchmarking.md)

## Overview

`BenchmarkComparison` collects `BenchmarkResult` objects from multiple configurations, automatically identifies the best and worst performers by throughput, and computes performance ratios relative to the best. Results can be serialized to JSON for archival.

## Quick Start

```python
from datarax.benchmarking import BenchmarkComparison, BenchmarkResult

comparison = BenchmarkComparison()
comparison.add_result("baseline", baseline_result)
comparison.add_result("optimized", optimized_result)

# Identify fastest
print(f"Best: {comparison.best_config}")
print(f"Worst: {comparison.worst_config}")

# Performance ratios (1.0 = best)
ratios = comparison.get_performance_ratio()
for name, ratio in ratios.items():
    print(f"  {name}: {ratio:.2%} of best")

# Save for later analysis
comparison.save("comparison_results.json")
```

---

::: datarax.benchmarking.comparative
