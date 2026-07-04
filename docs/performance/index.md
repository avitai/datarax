# Performance

Performance analysis and optimization tools. Understand your pipeline's performance characteristics and apply optimizations.

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| **Roofline** | Performance modeling | Compute vs memory bound |
| **XLA Optimization** | JAX/XLA tuning | Compilation hints |
| **Goodput** | Effective-time tracking | Useful vs stalled time |
| **Synchronization** | Host/device sync | Blocking + async copy helpers |

!!! note "Key points"

    - Roofline model reveals if you're compute or memory bound
    - XLA optimizations require understanding JAX compilation
    - Profile before optimizing - measure, don't guess
    - Most pipelines are I/O bound, not compute bound

## Quick Start

```python
from datarax.performance import RooflineAnalyzer

# Analyze a JAX operation against the detected hardware's roofline
analyzer = RooflineAnalyzer(hardware="auto")
result = analyzer.analyze_operation(my_fn, sample_input)

print(f"Arithmetic intensity: {result['arithmetic_intensity']:.2f}")
print(f"Bottleneck: {result['bottleneck']}")  # 'compute' or 'memory'
```

## Modules

- [roofline](roofline.md) - Roofline model analysis for performance characterization
- [xla_optimization](xla_optimization.md) - XLA-specific optimization utilities
- [goodput](goodput.md) - Effective-training-time tracking
- [synchronization](synchronization.md) - Host/device synchronization helpers

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
from datarax.performance import XLAOptimizer

optimizer = XLAOptimizer(target_hardware="auto")

# Apply hardware-tuned XLA flags
optimizer.setup_xla_flags()

# Cache JIT compilations to disk (persistent compilation cache)
optimizer.setup_compilation_cache()
```

To limit GPU memory usage, set the `XLA_PYTHON_CLIENT_MEM_FRACTION`
environment variable (for example `export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`)
before importing JAX.

## See Also

- [Benchmarking](../benchmarking/index.md) - Measure performance
- [NNX Best Practices](../user_guide/nnx_best_practices.md) - JAX optimization
- [Troubleshooting](../user_guide/troubleshooting_guide.md) - Common issues

## Choosing an Iteration Path

Each pipeline consumption pattern has a distinct performance profile:

| Pattern | Per-batch cost | Use when |
|---|---|---|
| ``for batch in pipeline:`` | One compiled dispatch (the module graph is split once per session) | Data-only loops, or feeding a training framework that owns its own step |
| ``pipeline.step()`` inside your ``nnx.jit`` train step | Absorbed by the outer trace | Fused data+train steps you orchestrate yourself |
| ``pipeline.scan(step_fn, ...)`` | One XLA call per epoch | Whole-epoch training with datarax managing the loop |
| Bare ``pipeline.step()`` in a Python loop | NNX graph traversal per call | Single batches, debugging, interactive use |

The iterator path keeps module state live at every yield boundary (so
mid-epoch checkpointing just works) and additionally exposes
``get_state()``/``set_state()`` for JSON-serializable data checkpoints
(see the checkpointing guide).
