# Performance

Performance analysis and optimization tools. Understand your pipeline's performance characteristics and apply optimizations.

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| **Roofline** | Performance modeling | Compute vs memory bound |
| **XLA Optimization** | JAX/XLA tuning | Compilation hints |

`★ Insight ─────────────────────────────────────`

- Roofline model reveals if you're compute or memory bound
- XLA optimizations require understanding JAX compilation
- Profile before optimizing - measure, don't guess
- Most pipelines are I/O bound, not compute bound

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.performance import roofline_analysis

# Analyze pipeline performance characteristics
result = roofline_analysis(
    pipeline,
    flops_per_sample=1e6,  # Estimated FLOPs
    bytes_per_sample=1e4,  # Data size
)

print(f"Arithmetic intensity: {result.intensity:.2f}")
print(f"Bottleneck: {result.bottleneck}")  # 'compute' or 'memory'
```

## Modules

- [roofline](roofline.md) - Roofline model analysis for performance characterization
- [xla_optimization](xla_optimization.md) - XLA-specific optimization utilities

## Roofline Model

The roofline model helps identify your performance bottleneck:

```
Performance (FLOPS)
     ^
     |     ______ Peak Compute
     |    /
     |   /   <- Memory bound region
     |  /
     | /_______ <- Compute bound region
     +-------------------> Arithmetic Intensity (FLOPS/byte)
```

## XLA Optimization Tips

```python
from datarax.performance.xla_optimization import (
    enable_persistent_cache,
    set_memory_fraction,
)

# Cache JIT compilations to disk
enable_persistent_cache("/tmp/jax_cache")

# Limit GPU memory usage
set_memory_fraction(0.9)  # Use 90% of GPU memory
```

## See Also

- [Benchmarking](../benchmarking/index.md) - Measure performance
- [NNX Best Practices](../user_guide/nnx_best_practices.md) - JAX optimization
- [Troubleshooting](../user_guide/troubleshooting_guide.md) - Common issues
