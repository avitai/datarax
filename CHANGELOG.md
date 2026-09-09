# Changelog

All notable changes to datarax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Depends on `substrax>=0.1.2`, the shared infrastructure package below calibrax, and
  `calibrax>=0.1.3`.
- **Floors match what is tested.** `jax>=0.11.1`, `flax>=0.12.9`,
  `orbax-checkpoint>=0.11.33` and `numpy>=2.1` (and `jax[cuda12]>=0.11.1` in the cuda12
  extra) are what every lock since 0.1.5 has resolved and run CI against; the previous
  floors promised compatibility nothing checked.
- `prefetch_to_device` lives in `datarax.control.prefetcher` (still exported from the
  package root) and no longer takes the `cpu_buffer_size` argument, which was accepted
  and discarded.

### Removed

- The `datarax.distributed` package. Its device placement, mesh, SPMD and metric
  utilities moved to substrax, which datarax now depends on; the names are unchanged
  and `docs/distributed/index.md` maps each to its new module: `DevicePlacement`,
  `HardwareType`, `BatchSizeRecommendation`, `place_on_device`, `distribute_batch` and
  `get_batch_size_recommendation` in `substrax.devices`; `DeviceMeshManager`,
  `MeshRules`, `data_parallel_rules`, `fsdp_rules`, `create_named_sharding` and
  `partition_spec_for_names` in `substrax.mesh`; `create_data_parallel_sharding`,
  `place_batch_on_shards`, `place_nnx_state_on_shards`, `spmd_train_step`,
  `reduce_gradient_tree`, the `reduce_*` and `*_collective` functions, `all_gather`
  and `collect_from_devices` in `substrax.spmd`. `DevicePlacement.prefetch_to_device`
  is the `prefetch_to_device` function. `data_parallel_train_step`,
  `place_model_state_on_shards` and `reduce_gradients_across_devices` had no consumer
  and are gone; `spmd_train_step`, `place_nnx_state_on_shards` and
  `reduce_gradient_tree` cover them.
- `datarax.performance.roofline` (`RooflineAnalyzer`, `HardwareSpecs`),
  `CompilationProfiler` and `DistributedUtils` from `datarax.performance.xla_optimization`.
  calibrax owns roofline analysis, compilation profiling and the hardware table
  (`calibrax.profiling`); datarax's copies had drifted from it and had no consumer in
  this package or its dependants. The hardware figures datarax's tests pinned now live in
  calibrax's tests.
- The eight `docs/benchmarking/*` API pages and the three `tests/benchmarks` files that
  documented and tested calibrax's profiler, monitor and adaptive-operation classes
  rather than any datarax code; the benchmarking page now links calibrax's reference.

## [0.1.5] - 2026-08-29

### Changed

- **Requires Python 3.12 or later.** jax 0.11.0 dropped 3.11, and this release
  takes that jax line.
- **The `gpu` extra is renamed `cuda12`.** JAX names its own extras for the CUDA
  major version and publishes no `gpu` extra, and this package also ships
  `metal`, which is a GPU. The extra no longer restates the jaxlib pin, which
  every jax cuda extra already enforces at its own version.
- Resolves to jax 0.11.1, jaxlib 0.11.1, flax 0.12.9, optax 0.2.8 and grain
  0.2.18, matching the sibling packages.
- `flax>=0.12.1` is now required: it is the first release with
  `nnx.Variable.set_value`, which the pipeline iteration path calls.
- `optax>=0.2.8` is now required. Below it, optax sets a jax config option
  removed in jax 0.10, so importing flax raises `AttributeError` and takes out
  collection for the whole suite.
- calibrax is resolved from PyPI at 0.1.2 or later rather than from a pinned git
  tag, so the tested environment matches what this package ships against.
- TensorFlow moved to the `tfds` extra, and the automation extra is isolated
  through a uv conflict so its exact pins no longer govern ordinary installs.
- `click` is declared in the `benchmark` extra, where `benchmarks/cli.py`
  imports it.

### Added

- A security policy, issue templates, and a security workflow.

## [0.1.4] - 2026-07-04

### Added

- `PipelineIterator`: `iter(pipeline)` over a random-access source now
  returns a compiled iteration session. The module graph is split once per
  session and batches run through a cached `jax.jit` step, removing
  per-batch Python graph traversal. Iteration throughput improves 2.4-9x
  across benchmark scenarios; peak memory on large in-memory datasets
  roughly halves.
- `PipelineIterator.get_state()` / `set_state()`: JSON-serializable
  iterator state (position and RNG counts) for exact mid-epoch resume.
- `datarax.pipeline.nodes`: DAG node library — `RebatchNode`
  (differentiable within-batch regrouping), `SplitField` (field routing),
  and `CachingIterator` (iteration-boundary memoization).
- Streaming iteration path: sources without random access iterate through
  a sequential batch loop with the operator DAG applied as a compiled call.
- Per-record RNG determinism: stochastic operators derive per-record keys
  from global record indices, so augmentation is invariant to batch size,
  shard count, shuffle order, and resume point.
- Benchmark capability model: scenarios declare required capabilities and
  adapters advertise what they support; the datarax adapter covers all 37
  scenarios. Empirical coverage recorded in `benchmarks/COVERAGE_MATRIX.md`.
- Real-data benchmark variants (CIFAR-10, WikiText-103, Criteo, COCO)
  behind a framework-neutral provider with opt-in downloads via
  `DATARAX_BENCH_DOWNLOAD=1`.
- A100 cloud baselines under `benchmarks/baselines/gpu_a100/`, alongside
  local RTX 4090 baselines; regression guard is hardware-gated.
- Cloud benchmark orchestration: `--region` and `--accelerators` placement
  controls, raised launch/stall watchdogs, spot-fallback teardown, and
  `setup.sh --with-benchmarks` bootstrap.
- Gap-ranked optimization backlog auto-generated from comparative runs
  (`benchmarks/OPTIMIZATION_BACKLOG.md`), with architecture-probe adapters
  excluded from ranking by default.

### Changed

- Checkpointing semantics: pipeline module state is synced at every yield
  boundary during iteration sessions, so `nnx.split`/Orbax checkpoints
  taken inside or after a training loop capture exactly the batches
  consumed.
- grain dependency raised to `>=0.2.18` (nightly pin dropped); shuffle
  samplers use Feistel-based `index_shuffle` with O(1) memory.
- Benchmark adapters fail fast on unimplemented scenario transforms
  instead of silently filtering them; a fairness test asserts this for
  every installed adapter.
- All benchmark baselines regenerated on the compiled iteration path.
- Example scripts honor `DATARAX_EXAMPLES_OUTPUT_DIR` for plot output, so
  test runs no longer modify committed documentation images.

### Removed

- The config-driven `datarax run` and `datarax profile` CLI subcommands
  (previously non-functional stubs). Construct and profile pipelines
  directly in Python.

## [0.1.3] and earlier

Releases prior to the changelog's introduction; see the git tag history
(`git log v0.1.2...v0.1.3`) for details.
