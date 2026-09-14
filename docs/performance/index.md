# Performance

Performance analysis and optimization tools. Understand your pipeline's performance characteristics and apply optimizations.

## Tools

| Tool | Purpose | Output |
|------|---------|--------|
| **Goodput** | Effective-time tracking | Useful vs stalled time |
| **Synchronization** | Host/device sync | Blocking + async copy helpers |

!!! note "Key points"

    - calibrax's roofline analyzer reveals if you're compute or memory bound
    - Profile before optimizing - measure, don't guess
    - Most pipelines are I/O bound, not compute bound

## Quick Start

```python
from calibrax.profiling import RooflineAnalyzer

# Analyze a JAX operation against the detected hardware's roofline
analyzer = RooflineAnalyzer()
result = analyzer.analyze_operation(my_fn, [sample_input])

print(f"Arithmetic intensity: {result.arithmetic_intensity:.2f}")
print(f"Bottleneck: {result.bottleneck}")  # 'compute' or 'memory_bandwidth'
```

## Modules

- [goodput](goodput.md) - Effective-training-time tracking
- [synchronization](synchronization.md) - Host/device synchronization helpers

## Roofline Model

The roofline model, implemented by
[`calibrax.profiling.RooflineAnalyzer`](https://calibrax.readthedocs.io/en/latest/api-reference/profiling/),
helps identify your performance bottleneck:

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

## JAX Process Settings

XLA flags, the persistent compilation cache and accelerator memory are settings of the JAX
process. Declare them once with
[`substrax.runtime`](https://github.com/avitai/substrax/blob/main/docs/api/runtime.md), before
importing JAX:

```python
from pathlib import Path

from substrax.runtime import JaxRuntime, apply_runtime

apply_runtime(
    JaxRuntime(
        compilation_cache_dir=Path("~/.cache/jax").expanduser(),
        memory_fraction=0.9,
    )
)

import jax  # noqa: E402 - the settings above must come first
```

JAX reads the memory fraction when its backends start, from `XLA_CLIENT_MEM_FRACTION`;
`XLA_PYTHON_CLIENT_MEM_FRACTION` is its deprecated name.

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
