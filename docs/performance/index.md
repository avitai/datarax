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

A pipeline is a data loader first: iterate it and pass each batch to your train or inference
step, written exactly as the Flax and JAX examples write it.

```python
@nnx.jit
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(model, grads)
    return loss

for batch in pipeline:
    train_step(model, optimizer, batch)
```

Measured on one RTX 4090 with a 1 GiB shuffled source, batches of 256, and a linear model trained
with SGD:

| Pattern | Cost | Peak device memory | Use when |
|---|---|---|---|
| ``for batch in pipeline:`` + your ``nnx.jit`` step | 0.31-0.34 ms per train step (0.08 ms per batch for the data alone) | 1.05 GiB | Training and inference, in any framework that takes batches |
| ``pipeline.scan(step_fn, modules=..., length=...)`` | 4.6 ms per 100 train steps | 1.03 GiB | Whole-epoch training fused into one XLA call, datarax running the loop |
| ``pipeline.step()`` in a Python loop | 0.35 ms per batch | - | Single batches, debugging, interactive use |
| ``pipeline.step()`` inside your default ``nnx.jit`` step | 3.4-3.6 ms per train step | 2-3 GiB | Avoid: see below |

The session behind ``for batch in pipeline`` and ``scan`` uploads NumPy source data to the device
once and reads device data in place, so neither copies the dataset per batch. The iterator keeps
module state live at every yield boundary (mid-epoch checkpointing works) and exposes
``get_state()``/``set_state()`` for the iteration state (see the checkpointing guide).

A pipeline passed into your own jitted step is an argument of that step, so the dataset goes
with it into every call:

- The default ``nnx.jit`` returns the pipeline's state, dataset included, from every call: each
  step copies the dataset on the device (the 3.4 ms and 2-3 GiB above).
- ``nnx.jit(..., graph=True, graph_updates=False)`` returns only what changed, so device data
  stays in place (0.54 ms per train step), but NumPy data is an argument JAX transfers to the
  device on every call (65 ms per train step).
- ``nnx.cached_partial`` does not accept the arrays a source holds
  ([google/flax#5109](https://github.com/google/flax/issues/5109)).

Flax plans to make tree mode the default for its transforms
([FLIP 5310](https://github.com/google/flax/blob/main/docs_nnx/flip/5310-tree-mode-nnx.md)).
Tree mode refuses a Variable reached twice, as a pipeline built with one ``nnx.Rngs`` for its
source and itself reaches its RNG counts (``ValueError: Duplicate RngCount``). datarax's own
transforms (iteration, ``step()``, ``scan``, checkpointing) pin graph mode and are unaffected. A
transform of your own that takes such a pipeline needs ``graph=True``, or the pipeline's parts
need separate ``nnx.Rngs``.

Inside ``nnx.grad`` or ``nnx.value_and_grad``, pass the pipeline as an argument: a function that
closes over it raises ``TraceContextError``, because ``step()`` writes the pipeline's position.

One pipeline serves one consumer at a time: an open iterator carries its own copy of the position,
so ``step()`` called on the same pipeline while it is open serves records the iterator serves
again. Records are immutable once given to a source: a NumPy array is
uploaded when first used, so an in-place edit after that is not served, while assigning a new
array is. Two pipelines over the same NumPy array each upload their own copy.
