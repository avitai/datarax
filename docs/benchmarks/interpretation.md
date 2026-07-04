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

On scenarios with zero or trivial transforms, per-batch dispatch and
NNX module marshalling around the JIT'd step body dominate the cost:
there is little data-movement work to amortize that fixed overhead
against. On such scenarios a numpy-host iterator can report higher raw
throughput simply because it does even less per batch, so the ratio
reflects pipeline overhead rather than a computational disadvantage.

| Scenario | Transforms | Real work? |
|----------|-----------:|:----------:|
| NLP-1 | 0 | No |
| IO-1 / IO-2 | 0–1 | Minimal |
| TAB-1 / TAB-2 | 1 (Normalize) | Minimal |

After the iteration-path optimization (baselines regenerated
2026-07-03), Datarax iter-mode itself already reaches high throughput
on these scenarios — for example, on an RTX 4090: NLP-1/small
≈194k elem/s, TAB-1/small ≈1.44M elem/s, IO-1 (memory source)
≈360k elem/s, IO-2/10k ≈290k elem/s (source:
`benchmarks/baselines/*.json`). Where a numpy-host iterator still edges
ahead, prefer the **scan-mode** adapter (`Datarax-scan`), which amortizes
NNX marshalling across whole epochs via `Pipeline.scan`.

### Category B — Asymmetric transform sets (historical, no longer occurs)

!!! note "Resolved by fail-fast transform resolution"
    This category described an older behavior and no longer applies.
    Earlier, some adapters limited themselves to a small
    `BASIC_TRANSFORMS` registry and silently skipped any transform
    outside that set, so a measured run could execute a lighter
    pipeline than the scenario requested.

    `resolve_transforms` (`benchmarks/adapters/_utils.py`) now **raises
    `ValueError` on any unimplemented transform**. An adapter that
    cannot express a scenario's full transform set is **excluded from
    that scenario** rather than measured with less work, and a
    determinism test asserts this for every installed adapter. As a
    result, every measured cross-framework comparison runs the identical
    transform set on both sides — there is no asymmetric-measurement
    situation to annotate around anymore.

### Category C — Datarax-only territory

Differentiable, self-supervised, and gradient-flowing scenarios
require a JIT-compiled pipeline with traceable parameters. Adapters
that yield raw arrays without operator state cannot implement these
scenarios and are not in the comparison.

| Scenario | Datarax | Best alternative | Note |
|----------|--------:|-----------------:|:----:|
| HPC-1, PC-3, PC-4, PC-5, XFMR-1 | only implementation | none (excluded) | capability-exclusive |

Datarax has no peer on these scenarios; the throughput value is the
floor of what the differentiable/JIT-compiled pipeline costs, not a
relative position. These are capability-coverage differentiators, not
head-to-head throughput wins (see
[Coverage Matrix](https://github.com/avitai/datarax/blob/main/benchmarks/COVERAGE_MATRIX.md)).

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

Qualitatively (measured 2026-07-03; RTX 4090 for iter-mode baselines,
A100 for the cloud profile):

- **Iteration-bound scenarios with zero or one trivial transform**
  (NLP-1, TAB-1, IO-1, IO-2) are dominated by per-batch iterator
  overhead. Scan-mode gives the largest relative gain here because it
  pays the NNX marshalling cost once per epoch instead of once per
  batch. For reference, current iter-mode throughput on these
  scenarios is already high (RTX 4090: NLP-1/small ≈194k elem/s,
  TAB-1/small ≈1.44M elem/s; source `benchmarks/baselines/*.json`).
- **Compute- or memory-bandwidth-bound scenarios** (PC-1, CV-1) show
  little benefit from scan-mode because the per-batch iterator
  overhead is already a small fraction of total time — the batch cost
  is dominated by the actual transform/data-movement work (RTX 4090:
  PC-1 depth_1 ≈370k elem/s, CV-1/small ≈334k elem/s).

Use throughput ratios as positioning data, not as a fixed speedup
figure — the exact multiplier depends on hardware, batch size, and
transform set.

### Implementation note: cached scan bodies

`@nnx.scan`'s JIT cache is keyed on the decorated function's
identity. Naively rebuilding the decorated body on every call
creates a fresh closure each time and defeats caching — every call
re-traces and re-compiles, which is slower than iter mode.

`Pipeline.scan` caches its compiled body on the Pipeline instance,
keyed on `(id(step_fn), length, n_modules, has_init_carry)`.
Calls with the same signature reuse the cached graph; the first call
pays the one-time trace+compile cost, and every subsequent call costs
only the wall-clock dispatch. User code does not need any special
pattern to get this behaviour — calling `pipeline.scan(step_fn,
length=N)` in a training loop hits the cache automatically as long as
`step_fn` is a stable function reference.

## Reading the comparison report

When reviewing the auto-generated `comparison_report.md`:

1. **Strengths** are scenarios where Datarax leads by >1.2×. These
   are typically Category C (no peer) or compute-bound where JIT
   fusion pays off.
2. **Gaps** list scenarios where another framework leads by >1.2×.
   Cross-reference each gap entry against the categories above:
   - Category A: expect substantial recovery from `Datarax-scan`.
   - Category B: no longer applies — fail-fast transform resolution
     guarantees both adapters ran the identical transform set.
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
