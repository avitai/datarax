# Interpreting Comparative Throughput

The comparative benchmark suite measures throughput across data-loading
frameworks on identical scenarios. Raw `elem/s` ratios are useful for
positioning, but they don't always reflect equivalent work being done.
This page documents the cross-framework asymmetries readers should keep
in mind when reading the gap classifier output.

## Three classes of throughput gap

The "below jax-dataloader" entries in the comparison report fall into
distinct categories. Each requires a different interpretation:

### Category A — Iteration-bound, no real work

Datarax pays a fixed per-batch overhead (~600 µs) for NNX module
marshalling around its JIT'd step body. On scenarios with zero or
trivial transforms, this overhead exceeds the actual data-movement
cost, so a numpy-host iterator wins on raw throughput because there
is no work to amortize against the overhead.

| Scenario | Transforms | Real work? |
|----------|-----------:|:----------:|
| NLP-1 | 0 | No |
| IO-1 / IO-2 | 0–1 | Minimal |
| TAB-1 / TAB-2 | 1 (Normalize) | Minimal |

Where readers see a 50× – 200× gap on these scenarios, the gap
reflects pipeline overhead rather than computational disadvantage.
Mitigation: use the **scan-mode** adapter (`Datarax-scan`), which
amortizes NNX marshalling across whole epochs via `Pipeline.scan`.

### Category B — Asymmetric transform sets

Several adapters limit themselves to a small `BASIC_TRANSFORMS`
registry (typically `Normalize` and `CastToFloat32`) and silently
skip any transform outside that set. When a scenario's transform list
includes operations beyond that registry, the adapters do not perform
equivalent work:

| Scenario | Datarax stages | Limited adapters' stages | Asymmetry |
|----------|---------------:|-------------------------:|----------:|
| PC-1 (depth_5) | 5 | 1 (Normalize) | 5× |
| PC-1 (depth_10) | 10 | 2 | 5× |
| CV-1 / CV-2 | 4–5 image ops | 1 | 4–5× |
| DIST-1, PR-1 | 3+ ops | 1 | 3× |

The throughput gap on these scenarios mixes architectural cost
against workload mismatch. Reports should annotate which transforms
each adapter actually executed before drawing conclusions.

### Category C — Datarax-only territory

Differentiable, self-supervised, and gradient-flowing scenarios
require a JIT-compiled pipeline with traceable parameters. Adapters
that yield raw arrays without operator state cannot implement these
scenarios and are not in the comparison.

| Scenario | Datarax | Best alternative | Note |
|----------|--------:|----------------:|:----:|
| HPC-1 | leads | `tf.data` (limited) | 22× lead |

Datarax has no peer here; the throughput value is the floor of what
the differentiable pipeline costs, not a relative position.

## What the JAX backend means for each adapter

Even on a CUDA-enabled host with `JAX_PLATFORMS=cuda`, not every
adapter exercises the GPU during iteration:

| Adapter | Iterates on | Notes |
|---------|------------|-------|
| `Datarax`, `Datarax-scan` | GPU | Pipeline stages compile to CUDA via `nnx.jit` |
| `jax-dataloader` | CPU | `dataset.asnumpy()` at setup; iterator is numpy fancy-indexing on host arrays |
| `grain` | CPU | Numpy slicing |
| `tf.data` | CPU/GPU (varies) | Depends on op fusion |
| `dali` | GPU | Explicit `.cpu().numpy()` at materialize |
| `pytorch_dl`, `ffcv`, `mosaic` | CPU | Numpy-host iteration |

**`jax-dataloader` does not iterate on GPU regardless of the JAX
backend.** Its `DataLoaderJAX` calls `dataset.asnumpy()` once at
setup, then every `next()` is `data[indices]` against host memory.
The "JAX" in its name refers to the consumer (a JAX program can eat
the arrays it yields), not the producer. Compare against it as a
numpy-host baseline, not as a peer GPU loader.

## How the synchronization barrier is enforced

All adapters force end-of-batch synchronization, but the mechanism
differs:

| Adapter | Sync mechanism |
|---------|----------------|
| `Datarax`, `Datarax-scan` | Explicit `jax.block_until_ready` |
| `jax-dataloader`, `grain`, `tf.data`, etc. | Implicit via `np.asarray()` (forces host readback) |
| `pytorch_dl`, `dali`, `ffcv` | Framework-native CPU pull |

The sync barrier is not a fairness issue — every adapter completes
its dispatch before reporting batch time. The cost difference between
adapters is in *what was dispatched*, not in *whether the dispatch
finished*.

## Two Datarax adapters: iter-mode vs scan-mode

The benchmark suite ships two measurement dimensions for the same
Datarax pipeline:

| Adapter | Execution mode | Best for |
|---------|----------------|----------|
| `Datarax` | Python iterator over `Pipeline.step()` | Interactive use, debugging, small datasets |
| `Datarax-scan` | Whole-epoch persistent `@nnx.scan` body | Training loops, throughput-critical paths |

In iter mode every batch pays the full NNX `split` / `merge` cost on
the host. In scan mode an `@nnx.scan`-decorated body consumes
`length` batches inside one XLA graph; the marshalling cost is paid
once at compile time and amortized across all iterations.

Measured GPU throughput on representative iteration-bound scenarios
(A100-class hardware, no transforms or one trivial transform):

| Scenario | iter mode | scan mode | speedup |
|----------|----------:|----------:|--------:|
| NLP-1 (0 transforms) | 70 ms / 100 batches | 2.6 ms / 100 batches | ~27× |
| TAB-1 (1 transform) | 30 ms / 100 batches | 2.7 ms / 100 batches | ~11× |
| PC-1 depth_1 (1 MB batches) | 140 ms / 100 batches | ~140 ms / 100 batches | ~1× |

Category A scenarios (NLP-1, TAB-1, IO-1, IO-2) are dominated by
iterator-overhead and recover dramatically. Category B/C scenarios
(PC-1, CV-1) are compute- or memory-bandwidth-bound and show little
benefit from scan-mode because the iterator overhead is already a
small fraction of total time.

### Implementation note: cached scan bodies

`@nnx.scan`'s JIT cache is keyed on the decorated function's
identity. Naively rebuilding the decorated body on every call
creates a fresh closure each time and defeats caching — every call
re-traces and re-compiles, which is slower than iter mode.

`Pipeline.scan` caches its compiled body on the Pipeline instance,
keyed on `(step_fn identity, length, n_modules, has_init_carry)`.
Calls with the same signature reuse the cached graph; first call
pays the trace+compile cost (~150 ms for a small workload), every
subsequent call costs only the wall-clock dispatch (~2-5 ms for the
same workload). User code does not need any special pattern to get
this behaviour — calling `pipeline.scan(step_fn, length=N)` in a
training loop hits the cache automatically as long as `step_fn` is a
stable function reference.

## Reading the comparison report

When reviewing the auto-generated `comparison_report.md`:

1. **Strengths** are scenarios where Datarax leads by >1.2×. These
   are typically Category C (no peer) or compute-bound where JIT
   fusion pays off.
2. **Gaps** list scenarios where another framework leads by >1.2×.
   Cross-reference each gap entry against the categories above:
   - Category A: expect substantial recovery from `Datarax-scan`.
   - Category B: verify both adapters ran the same transform set.
   - Category C-adjacent: compare Datarax-iter against Datarax-scan
     for the same workload to confirm whether the gap is iterator
     overhead or something deeper.
3. **Interpretation Notes** at the end of the report point to this
   page; treat the raw ratios as positioning data, not as a
   capability ranking.

## Related pages

- [Comparison framework overview](comparison.md)
- [Methodology](methodology.md)
- [Cloud benchmark workflow](cloud.md)
- [Live W&B dashboard](dashboard.md)
