# Changelog

All notable changes to datarax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `datarax.core.spec.validate_batch` checks a batch against a declared element spec: tree
  structure, per-element shapes, dtypes and one shared record count, with a short final
  batch allowed when `batch_size` is given. Every problem is reported with its field path
  in one `SpecMismatchError` (a `ValueError`). Only shapes and dtypes are read, so nothing
  is copied or cast, and the check runs on tracers without adding to the compiled graph.
- `device_spec` states a spec as JAX arrays hold its data under the active x64 setting,
  `validate_device_dtypes` refuses declared dtypes the device cannot hold as declared, and
  `spec_mismatches` and `batch_length` expose the field-level comparisons.

### Changed

- `array_to_spec` and `array_to_spec_strip_leading` describe a value exactly as given.
  They called `jnp.asarray`, which copied the value to the device (the whole dataset for
  `MemorySource.element_spec`: 12 ms for 200,000 x 64 `float64` rows on CPU) and turned
  host `float64`/`int64` into `float32`/`int32` while x64 was off. Sources whose batches
  are converted JAX arrays now state it with `device_spec`: `MemorySource` and
  `EagerSourceBase` declare `device_spec` of their storage, the same values as before.
  Code that derives a spec with `array_to_spec` from data it later converts must wrap the
  result in `device_spec`.
- Streaming iteration (`for batch in pipeline` over a source without indexed access)
  validates every source batch against `source.element_spec()` before it reaches the DAG,
  and refuses a declared dtype the device would narrow when the pass starts. It used to
  check only the first leaf's leading axis, so a wrong trailing shape, a `float64` batch
  declared `float32` (narrowed silently inside the compiled DAG), an undeclared or missing
  field, or leaves with different record counts all ran, and the position advanced by the
  first leaf's length.
- Streaming iteration runs the stage DAG as one compiled step per batch shape instead of
  calling `nnx.jit(Pipeline.__call__)` on every batch. The step covers the stage modules
  and the position counter only, so the module graph is no longer traversed per batch
  and a source that replaces its backend iterator between passes no longer forces a
  recompile. Measured on CPU with 64-record batches over repeated epochs: 26.7 µs per
  batch instead of 445.8 µs without stages, and 45.4 µs instead of 817.8 µs with one
  stochastic stage. Pipeline state is still read before and written back after every
  batch, and every Variable a stage writes, whatever its type, reaches the live module. A
  stage that adds or removes state while it runs is refused with a `ValueError`.
- Streaming iteration reads `source.element_spec()` once per source and x64 setting,
  because the TFDS and HuggingFace streaming sources open their backend to answer it. A
  streaming source must implement `element_spec()`, and its declaration must stay fixed
  after construction.
- `MixDataSourcesNode` names every differing field when it rejects incompatible sources.
- `DataSourceModule.supports_indexed_access()` is true for any source whose class implements
  `get_batch_at`, which the base class documents as stateless and JAX-traceable, so
  `MemorySource`, `EagerSourceBase` and `MixDataSourcesNode` no longer restate it. Iterating
  a source that implements neither `get_batch_at` nor `get_batch` raises `TypeError` when
  iteration starts; it used to fail inside the loop with `AttributeError: get_batch`.
- `ArrayRecordSourceModule` takes `decode`, a function turning one `bytes` record into a
  dict of arrays, and implements `get_batch`: records of the current epoch are decoded and
  stacked on the host, and each `for batch in pipeline` pass covers one epoch. Its
  `element_spec` describes a decoded record. The host-only `get_batch_at`, which needed a
  concrete Python int and returned raw records, is removed; Pipeline iteration over the
  source used to fail.

### Fixed

- Iterating a random-access source (`for batch in pipeline`) kept only RNG counts and
  plain `nnx.Variable` state between batches, so a stage that updated `nnx.BatchStat` or
  any other Variable type lost the update silently: a running total stayed 0.0 where
  `Pipeline.step()` reached -109.6. The compiled session now finds, while tracing, every
  Variable the step writes and returns exactly those, and a step that adds or removes
  state is refused with a `ValueError`. The session calls the step body directly instead
  of nesting `nnx.jit`, whose write-back rebinds every Variable. Measured against
  651389f with paired interleaved rounds and one stochastic stage: 120.2 to 45.9 us per
  batch on CPU and 150.9 to 62.6 us on GPU.
- `HFStreamingSource.element_spec` and `TFDSStreamingSource.element_spec` describe the
  records the source emits: the first record is filtered and converted exactly as
  iteration does it, so `include_keys` and `exclude_keys` apply. The spec used to list
  every dataset column.
- `StreamingDiskSource` iterates through `Pipeline`. It implements a traceable `get_batch_at`
  but reported no indexed access, so `for batch in pipeline` failed with
  `AttributeError: get_batch`, and `step()` failed with `UnexpectedTracerError`: the
  memory-map was NNX state, traced into the step and read back by the host callback. The
  memory-map now stays on the host, outside module state.
- Test runs honour `DATARAX_TEST_JAX_PLATFORMS`. The test configuration applied it and
  then its CPU device emulation reset `JAX_PLATFORMS` to `cpu`, so a run asking for CUDA,
  `scripts/run_gpu_tests.sh` included, used eight emulated CPU devices. The backend is now
  resolved once, before JAX is imported: an accelerator request disables emulation, a
  `JAX_PLATFORMS` inherited from the shell still leaves tests on the CPU, and asking for
  CUDA without a JAX CUDA plugin fails. `scripts/run_gpu_tests.sh` activates the project
  environment and requests CUDA that way.
- The development docs, the root and benchmark Dockerfiles and the Sky templates target Python 3.12,
  the minimum this package requires. They named Python 3.11, the `gpu` extra (now
  `cuda12`), a `setup.sh --force` flag that does not exist, and `uv pip install -e` for a
  checkout whose tooling syncs with `uv sync`.

## [0.1.8] - 2026-09-09

### Fixed

- A shuffled epoch through `Pipeline` is a permutation. `step()` drew a fresh key from
  the pipeline's rng stream for every batch and `get_batch_at` builds a full permutation
  from its key, so consecutive batches came from different permutations and an epoch
  repeated some records and skipped others (measured: 4 batches of 16 from 64 shuffled
  records visited 42 distinct records). Every batch of an epoch now receives the same
  epoch key, derived from a base key drawn once at construction and the epoch counter,
  on both the iterator and the `scan` paths.

### Added

- `Pipeline.reset()` starts the next epoch (position 0, epoch counter advanced, a new
  permutation for shuffled sources); iterating an exhausted pipeline yields nothing until
  it is reset. `Pipeline.epoch_key()` is the key the current epoch passes to the source.
  `PipelineIterator.get_state()` / `set_state()` carry `epoch` alongside `position` and
  `rng_counts`, so a mid-epoch resume reproduces the same slices.

## [0.1.7] - 2026-09-09

### Changed

- The helpers a data source is built from are public: `datarax.sources.source_ops`
  (formerly the private `datarax.sources._eager_source_ops`) now also holds
  `resolve_wrapped_indices` (formerly on the private `datarax.sources._source_base`), and
  `datarax.sources` exports `eager_iter`, `eager_get_batch`, `eager_reset` and
  `resolve_wrapped_indices`. DiffAV and DiffBio imported the private paths; the private
  module path is gone.

## [0.1.6] - 2026-09-09

### Added

- `scripts/derive_status.py` measures the benchmark scenario count, the peer-framework
  count and every adapter's scenario coverage from the adapter registry and checks the
  README and `benchmarks/COVERAGE_MATRIX.md` against them; CI runs it with `--check`.
- A weekly `Upstream Compatibility` workflow runs the fast test subset against the newest
  Grain and Orbax releases, ahead of the lock.

### Changed

- Depends on `substrax>=0.1.2`, the shared infrastructure package below calibrax, and
  `calibrax>=0.1.3`.
- **Floors match what is tested.** `jax>=0.11.1`, `flax>=0.12.9`,
  `orbax-checkpoint>=0.11.33` and `numpy>=2.1` (and `jax[cuda12]>=0.11.1` in the cuda12
  extra) are what every lock since 0.1.5 has resolved and run CI against; the previous
  floors promised compatibility nothing checked.
- `IteratorCheckpoint` is the one checkpoint API and is built on
  `substrax.checkpoint.OrbaxCheckpointStore`: `save(target, step, metadata=)`,
  `save_if_due(target, step, interval=)`, `restore(target, step=None)`, `all_steps()`,
  `latest_step()`, `has_checkpoint()`. Any `Checkpointable` (module, pipeline, iterator)
  is accepted; retention is `max_to_keep` on the store; restore validates the
  checkpoint's identity fields against the target before applying it.
- `prefetch_to_device` lives in `datarax.control.prefetcher` (still exported from the
  package root) and no longer takes the `cpu_buffer_size` argument, which was accepted
  and discarded.

### Fixed

- `DataraxModule.set_state` restores the module's `nnx.Rngs` streams. They were skipped
  without a word because `nnx.Rngs` is not an `nnx.Module`, so a stochastic module resumed
  from a checkpoint drew fresh random numbers instead of continuing its stream.

### Removed

- `OrbaxCheckpointHandler` and `PipelineCheckpoint`. The handler re-implemented what
  Orbax's `CheckpointManager` does (step discovery, pruning, a marker codec for strings
  and PRNG keys that `PyTreeSave` carries natively) and substrax's store is its home;
  `PipelineCheckpoint.save_to_step` is `IteratorCheckpoint.save_if_due`. The `keep`,
  `overwrite` and `async_checkpointing` options are gone with them.
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
