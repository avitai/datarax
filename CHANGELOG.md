# Changelog

All notable changes to datarax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CompositeOperatorModule.mixture_weights()` returns the weights a `WEIGHTED_PARALLEL` composite
  applies: its static weights, or `softmax(weight_logits / temperature)` for learnable weights. A
  composite that reads its weights from each record (`weight_key`) has no fixed mixture and
  raises `ValueError`, as does a composite with another strategy.
- `examples/comparison/01_grain_datarax_quickref.py`, with its notebook and docs page: one job
  run with Grain and with Datarax side by side — reading records, adding per-record noise,
  batching, and resuming an interrupted epoch from saved iterator state — with what each
  library's checkpoint holds and where its randomness comes from. The numbered files under
  `examples/comparison` are now discovered and checked like every other tutorial.
- `examples/comparison/02_randomness_and_learnable_operators_tutorial.py`, with its notebook and
  docs page: what a record's randomness depends on in Grain (`Philox(seed + draw_index)`, which
  belongs to the draw) and in Datarax (`fold_in(fold_in(base_key, epoch), record_index)`, which
  belongs to the record), each reproduced and checked under another shuffle order and batch
  size; then a learnable stage and a `WEIGHTED_PARALLEL` composite of image operators with
  `learnable_weights=True`, both fitted by differentiating an epoch of `Pipeline.scan`.
- `examples/comparison/03_sharding_guide.py`, with its notebook and docs page: which records
  each process serves under Grain's `ShardOptions` (contiguous ranges, what `ShardByJaxProcess()`
  builds from the JAX process) and Datarax's `MemorySourceConfig(shard_id, num_workers)` (every
  `n`-th record); that a Datarax record's randomness is the same on one process and on two,
  because its key is a function of the global record index; and the same
  `substrax.spmd.place_batch_on_shards` and `spmd_train_step` applied to both libraries' batches
  under `jax.set_mesh`, giving the same losses on the same records.
- `examples/comparison/04_resumed_training_guide.py`, with its notebook and docs page: model,
  optimizer and loader state saved together through one `substrax.checkpoint.OrbaxCheckpointStore`
  at a mid-epoch step (Grain's JSON bytes as a string leaf, Datarax's iterator state as a dict),
  restored into fresh objects, and both resumed runs reproducing the uninterrupted runs loss for
  loss, with `calibrax.metrics.functional.mse` as the loss.
- An operator can override `compute_statistics(batch_data)` to fit the statistics it applies to
  each batch, as batch normalization does. The batch path calls it once per batch, before the
  batch is vectorized, and gives every record the result. An operator that stores fixed
  statistics with `set_statistics` is unaffected: the default `compute_statistics` returns
  exactly those, so what such an operator produces does not change.

### Changed

- `SelectorOperatorConfig.normalized_weights` is a tuple of floats instead of a `jax.Array`.
  Comparing two equal configs raised `ValueError: The truth value of an array with more than one
  element is ambiguous`, which also reaches `nnx.jit` dispatch, since a config is graphdef
  metadata that dispatch compares.
- `BatchMixOperator.apply` raises `NotImplementedError` instead of returning its input unchanged,
  which looked like a successful mix. Batch mixing combines each record with another record of the
  same batch, so it has no per-record form: call `apply_batch(batch)`, or the operator itself.
- datarax requires substrax 0.1.7, whose `place_batch_on_shards` places the NumPy leaves of a host
  batch and assembles a global batch from each process's slice.
- `CompositeOperatorConfig` for `WEIGHTED_PARALLEL` takes `mix_fields`, the dotted paths of the
  data fields to combine. It defaults to the fields the child operators declare they write
  (`target_key` or `field_key`) and must be given when a child declares none, such as an
  `ElementOperator` or `MapOperator`. Static weights still form a linear combination.
  `learnable_weights=True` now stores logits in `weight_logits`, initialized to
  `log(weights / sum(weights))` and mixed with `softmax(logits / temperature)` (new
  `temperature`, default 1.0; initial weights must be positive), as DARTS and Faster
  AutoAugment learn operation mixtures. The DDSP and DADA guides name their mixed field.
- `ModalityOperator` reads and writes dotted field paths through the new
  `datarax.core.field_paths.get_field` and `set_field`, which weighted-parallel composites use too.
- The sharding examples and their notebooks, the sharding docs pages, the distributed scaling
  benchmark and the comparison example build meshes with `substrax.mesh.DeviceMeshManager` and place
  batches with `substrax.spmd.create_data_parallel_sharding` and `place_batch_on_shards`, replacing
  their own `Mesh(...)` construction and the `create_sharding_spec` and `distribute_batch` helpers;
  `benchmarks/core/sharding.py`, which duplicated them, is removed. Meshes are entered with
  `with jax.set_mesh(mesh):` instead of `with mesh:`, which jax deprecates in favour of it, in those
  files and the sharding tests, and the TensorFlow comparison tables map `strategy.scope()` to
  `jax.set_mesh(mesh)`. Contracts fail any `with mesh:` in source, tests, benchmarks, examples,
  notebooks or docs, and any raw `Mesh(...)` or `jax.make_mesh(...)` in examples, benchmarks or
  docs pages.
- The generated `.datarax.env`, the pytest environment, the Dockerfiles, the SkyPilot template,
  the GPU run scripts and the docs set `XLA_CLIENT_MEM_FRACTION`, the name jaxlib reads, instead
  of the deprecated `XLA_PYTHON_CLIENT_MEM_FRACTION`. jax refuses a process that sets both;
  re-running `source activate.sh` unsets the old name from a shell activated before.
- Examples write their figures through `substrax.artifacts.resolve_output_dir("examples")`
  instead of reading `DATARAX_EXAMPLES_OUTPUT_DIR`. They default to a per-run temporary
  directory, never the working tree. Set `AVITAI_OUTPUT_DIR` to an absolute directory to choose
  where `examples/` is written; `scripts/run_all_examples_on_gpu.sh` points it at
  `docs/assets/images`. The example tests run each example in its own interpreter through
  `substrax.testing.run_example`, and an example that runs past its time budget now fails
  instead of being skipped.
- The cloud benchmark launchers (SkyPilot template, Vast orchestrator, Vertex job script) select
  the GPU backend with `JAX_PLATFORMS=cuda,cpu` alone. jax makes the first listed platform the
  default, so the deprecated `JAX_PLATFORM_NAME` they also set is no longer exported.
- CI's long-running tier restores CIFAR-10 from a cache that a new `prepare_example_datasets` job
  fills with `scripts/prepare_example_datasets.py`, so the examples that load it no longer
  download it inside their time budget. The dataset's host sends each connection about 100 KB/s,
  so the script fetches the TFDS and keras archives in byte ranges over eight connections,
  verifies their SHA-256, and places them where both loaders reuse them instead of downloading.

### Changed

- An operator holds the statistics it applies. `set_statistics` stores them, `get_statistics` reads
  them back, `reset_statistics` clears them, and `compute_statistics(batch_data)` returns the stored
  statistics, which an operator that fits statistics to each batch overrides. The store is a plain
  `nnx.Variable`, so statistics are module state: they round-trip through a checkpoint, and two
  operators with equal configurations share one compiled trace whatever their statistics hold.
- Two operators with equal configurations share one compiled trace. Every operator used to carry a
  unique integer identity, and a configuration is graphdef metadata, so each instance traced again and
  a rebuilt pipeline recompiled. A wrapper whose configuration holds child modules — the composite, the
  selector, the probabilistic wrapper — still traces again when it is rebuilt over freshly constructed
  children, because flax compares modules by identity.
- An operator called on the raw path with empty states and then on a batch carrying a state field, or
  in the other order, no longer raises `vmap out_axes specification must be a tree prefix of the
  corresponding value`. The output axes come from the call itself rather than from a cache keyed on the
  input structure alone.
- `cacheable` moves from `DataraxModuleConfig` to a new `SamplerConfig`, and the result cache with its
  hashing helpers and `reset_cache` move from `DataraxModule` to `SamplerModule`. A sampler memoizes each
  sampled list by request size, which is the only module kind with something to key a cache on; every other
  module carried an empty dict through each compiled step. The six sampler configs derive from
  `SamplerConfig`, `eager_reset` and `reset_streaming_state` no longer take the cache they only ever cleared
  when it was empty, and the modality and cross-modal docstrings stop promising a caching system that never
  ran. A config that set `cacheable` on an operator or another module must drop it.
- An operator's `apply` takes the record's PRNG key as its fourth argument and draws whatever
  randomness it applies from that key. `generate_random_params` is gone, along with
  `datarax.operators._random_params`: an operator no longer produces a batch of parameters for
  `_vmap_apply` to distribute, so the two halves of every stochastic operator — drawing and
  applying — are one function of one record. A stochastic operator handed no key raises through the
  new `datarax.core.operator.require_key`, naming itself, rather than silently applying a fixed
  value. An operator that overrode `generate_random_params` moves its draw into `apply`; one that
  read the fourth argument as parameters reads it as a key.
- A composition passes each child a key folded from the record's own, by the child's position, in
  place of the `{"operator_i": params}` dictionaries it used to build. `StrategyContext` carries
  `key` instead of `random_params`.
- A `probability=1.0` wrapper around a stochastic child is itself stochastic, so the child receives
  a key and draws. Previously such a wrapper was classified deterministic, was handed nothing, and
  passed nothing down, which silently turned its stochastic child into a fixed one; the outputs of
  those compositions change. `CompositeOperatorConfig` already derived its mode from its children,
  and `SelectorOperatorConfig` is always stochastic, so neither changes. `add_operator` refuses a
  stochastic operator on a deterministic composite, which would otherwise have no key to give it.
- The stable per-record index that `OperatorModule._vmap_apply` and `_apply_on_raw` take is named
  `record_indices`, which is what the pipeline layer calling them already called it (`run_dag`,
  `PipelineIterator`, `per_record_keys`). One concept carried two names across the seam between the two
  layers, and the executor passed `record_indices` into a parameter named `global_indices`. Both methods
  are private and every call site passes the index positionally, so only a caller naming it as a keyword
  has anything to change; the helper reading the indices out of batch metadata is
  `_record_indices_from_metadata`.
- An operator no longer keeps the `nnx.Rngs` its caller passes. The base class reads it once, to
  draw the operator's stable base key, and leaves `rngs` as `None`; a module that is not an
  operator keeps its `Rngs` as before, and a subclass may still store one after
  `super().__init__`. A call carrying record indices — every pipeline batch, `apply_batch` and
  `_apply_on_raw` — still keys each record on the base key, so pipeline outputs do not change. A
  direct call carrying no record identity draws from a private per-operator stream instead, so
  two such calls now differ where they used to repeat one draw. That stream hangs below
  `datarax.core.operator.DIRECT_CALL_STREAM`, one level deeper than any key a pipeline derives,
  so the two families cannot coincide; `BatchMixOperator`'s direct calls draw from it too.
- `PipelineIterator.get_state()` carries a `version`, and `rng_counts` holds one count per
  stochastic operator rather than one per stream of the `Rngs` each operator held. A
  deterministic operator contributes none and operators that shared one `Rngs` no longer share a
  count, so the list changes length: a pipeline with one stochastic operator reads `[0, 1, 0]`
  where it read `[1, 1, 0]`. A state saved without a `version` is upgraded on load — the counts
  outside operators keep their values and order, and every operator count restores to 0. A
  pipeline that holds an operator's count after one no operator owns, such as an operator inside
  a source, refuses the state instead of resuming from counts placed wrongly.
- Module checkpoints written before these changes are upgraded on load: an operator's `rngs`
  subtree is dropped, `_rng_stream` is rebuilt from the saved `_base_key`, and statistics saved
  as `_computed_stats` are read as `_statistics`. `DataraxModule._upgrade_saved_state` is the
  hook a module overrides to carry its own layout change, applied to each module's own subtree
  before the state is validated.
- A wrapper's children receive the statistics they computed on the wrapper's input rather than
  the wrapper's own. A composite, a selector and a probabilistic wrapper each compute one entry
  per child, once per batch, and give each child its own; a wrapper whose children all compute
  none passes none. Composite children used to receive no statistics at all, and selector and
  probabilistic children received the wrapper's. A wrapper applies its children inside one
  vectorized call, so a child cannot compute statistics of its own while it runs: a sequential
  composition's later children see the statistics of the composition's input, not of the
  previous child's output. Exact per-stage statistics come from separate `Pipeline` stages.
- `BrightnessOperatorConfig` and `ContrastOperatorConfig` refuse the parameter their mode does not
  use and name the one it does. A deterministic operator applies `brightness_delta` or
  `contrast_factor` and refuses a range; a stochastic operator draws from `brightness_range` or
  `contrast_range` and refuses a fixed value. All four now default to `None` and resolve for the
  mode: a stochastic operator draws from `(-0.2, 0.2)` or `(0.8, 1.2)`, a deterministic one
  applies `0.0` or `1.0`. Previously a deterministic operator given `brightness_range=(0.2, 0.2)`
  returned the image unchanged, and a `brightness_delta` given with `stochastic=True` was ignored
  the same way, because that mode draws from the range; neither was reported. The check both
  configs share lives in `datarax.operators.modality.image._validation.resolve_mode_parameters`.

### Removed

- `DataraxModuleConfig.batch_stats_fn` and `precomputed_stats`, with the validation that made them
  mutually exclusive, and the statistics system on `DataraxModule`: `compute_statistics`,
  `get_statistics`, `set_statistics`, `reset_statistics`, `_computed_stats` and `_is_stats_reset`.
  Statistics are fitted values, and a configuration is static metadata that every transform
  compares, so holding them there made each fitted number part of what decided whether two
  operators could share a compiled trace. They now live on the operator that applies them. A config
  that passed either field must set the statistics on the operator instead. A module whose only
  variables were these two now has an empty state, which Orbax refuses; `IteratorCheckpoint.save`
  rejects it by name rather than letting Orbax raise its own `Found empty item.`
- `SamplerModule._compute_statistics`, `_maybe_update_statistics` and `_last_computed_stats`. The
  sampler recomputed statistics after every call into an attribute that nothing in datarax or any
  dependent read.
- `OperatorModule.get_output_structure`, the module-level output-structure cache with its size bound,
  and the per-operator identity that keyed it. The batch path vectorizes with `out_axes=0`, which is a
  tree prefix of whatever an operator returns, so nothing discovers the output structure before the
  call and no operator declares it. The four operators that did — the selector, the external adapter,
  the CREPE f0 operator and the loudness operator — no longer carry an override. A dependent that
  reads an operator's `_unique_id`, such as to wrap it as `nnx.static` for a transform, must drop that
  code: the attribute no longer exists.
- `DataraxModule.copy`. It rebuilt a module as `type(self)(config=..., rngs=..., name=...)`, which 17 of the
  42 module classes cannot accept: nine operators take no `name` (the composite, the selector, the
  probabilistic wrapper and the six image operators), `PureJaxAdapter` takes no `rngs`, and eight classes
  require an argument of their own (a mapped function, the data, the paths, the sources, an element spec).
  Construct the module with the arguments its constructor takes, or use `nnx.clone` for a structural copy.
- `CompositeOperatorModule.operator_statistics` and `StrategyContext.stats_callback`, with the
  `_emit_operator_statistics` hook the sequential and parallel strategies called. The hook
  forwarded an operator's `statistics` attribute, which no operator defines, so the variable
  every composite carried through each compiled step stayed empty.
- The pytest `--device` option and the `gpu`, `gpu_required`, `cuda`, `cpu` and `tpu` markers, which
  only a keyword filter behind that option acted on. A test that needs a GPU backend or several
  devices declares the substrax plugin's `accelerator(kind="gpu")` or `devices(count)` marker, which
  skips it from the backend and devices the run selected; `DATARAX_TEST_JAX_PLATFORMS=cuda` selects
  the GPU. CI, `scripts/run_tests.sh`, `scripts/run_gpu_tests.sh` and the contributing docs no longer
  pass `--device`. The unused `tests/test_common/device_detection.py` and `hardware_fixtures.py`, the
  device-skip fixtures and decorators in `tests/test_common`, and `benchmarks/core/platform.py`'s
  `required_devices` are removed.
- The `datarax` command no longer reads `DATARAX_DEVICE`. It set jax's deprecated
  `jax_platform_name` option after jax was imported, which does not choose the backends jax
  starts. Set `JAX_PLATFORMS` before running the command instead.
- `datarax.performance.xla_optimization`: `XLAOptimizer`, `get_xla_flags`, `apply_xla_flags`,
  `SmartCompilation` and `MemoryEfficientCompilation`. JAX process settings move to
  `substrax.runtime` (`JaxRuntime`, `apply_runtime`, `merge_xla_flags`), and the compilation
  wrappers give way to `jax.jit`, `jax.shard_map`, `jax.jit(donate_argnums=...)` and
  `jax.checkpoint`.
- `DataraxModule.get_operation_stats`, `reset_operation_stats`, the applied and skipped operation
  counters and the `IterationCount` variable type. Nothing outside the tests incremented the
  counters, so they always read zero while every module carried them through each compiled step.
- `datarax.sharding.ArraySharder`, `datarax.core.SharderModule` and `SharderModuleConfig`, which
  duplicated substrax. Place a batch on a mesh with `substrax.spmd.place_batch_on_shards`, map
  logical axis names with `substrax.mesh.MeshRules` and `partition_spec_for_names`, build a named
  sharding with `substrax.mesh.create_named_sharding`, apply a function per shard with
  `flax.nnx.shard_map` and create a partitioned parameter with `flax.nnx.with_partitioning`.
  `JaxProcessSharderModule` and `JaxProcessSharderConfig` now derive from `DataraxModule` and
  `DataraxModuleConfig`, `datarax.sharding` exports both, and the component registry has no
  `sharder` type.
- The `examples/comparison` mock-ups `quick_demo.py`, `02_stateful_transformations.py`,
  `03_distributed_memory_efficient.py` and `04_end_to_end_pipeline.py`, with their notebooks.
  They imported neither datarax nor grain, illustrating a stateful-versus-stateless design in
  plain flax and numpy, and `validate_examples.py` excluded the directory from every example
  check, so none of them had to pass one. Its tutorials now run both libraries on one job and
  are checked like every other numbered example.

### Fixed

- `PipelineIterator.get_state()` returns `position` as an `int`. It was an `np.int64`, so the
  state the method documents as JSON-serializable failed `json.dumps`, and a checkpoint store
  building its restore template from the state refused the NumPy scalar.
- `functional.adjust_contrast`, which `ContrastOperator` and `color_jitter` apply, returned a
  one-channel image with three channels, and changed a uniform image by a fabricated per-channel
  offset. It added a hardcoded three-element offset to any uniform three-channel image whose factor
  was not 1.0, inside a `jnp.where` whose broadcast reached the output even when its condition was
  false. The offset is gone: each channel is scaled about its spatial mean, the output has the
  input's shape, and a uniform image, which has no contrast, comes back unchanged for every factor.
- `CrepeF0Operator` could only run in eager eval mode. It computed its pad width with
  `jnp.maximum`, which `jnp.pad` cannot read under a trace, so every `jit` path and the
  `batch_strategy="scan"` its own config recommends raised `ConcretizationTypeError`, and in
  train mode `vmap` raised `TraceContextError` because BatchNorm wrote its statistics from
  inside the mapped function. The pad width is a Python `int`, and pitch extraction reads the
  stored batch statistics, so the operator runs under `jit`, `vmap` and `scan` in both modes.
  `CrepeModel.__call__` takes `use_running_average` for that choice.
- `PipelineIterator.set_state` accepted a negative `position` or `epoch`. An epoch of `-1` becomes
  `2**32 - 1` where it is folded into a record key, so the restored iterator would draw a different
  stream than the one that was saved, and a negative position would place the iterator before the
  start of its epoch. Both now raise `ValueError`.
- The composition strategies and advanced operators tutorials built their fixed brightness and
  contrast operators with `brightness_range=(d, d)` and `contrast_range=(f, f)` in deterministic
  mode, which applies `brightness_delta` and `contrast_factor`, so those operators returned images
  unchanged. They set `brightness_delta` and `contrast_factor`.
- A `WEIGHTED_PARALLEL` composite took the weighted sum of every field of its operators'
  outputs, so fields no operator wrote were scaled whenever the weights did not sum to one and
  integer fields became floats (with weights `[1.0, 0.1]`, an untouched `f0_hz` of 440 became
  484 and an int32 label of 3 became 3.3). Only the `mix_fields` are combined now; every other
  field passes through from the input unchanged.
- Stochastic operators in a `Pipeline` keyed each record's randomness on its position in the
  epoch, and that position restarts every epoch. A record's augmentation therefore changed when
  the order was shuffled, and repeated identically every epoch (every record with
  `shuffle=False`, every slot with `shuffle=True`). Keys are now
  `fold_in(fold_in(base_key, epoch), record_index)`, where `record_index` is the stable index
  that `DataSourceModule.record_indices_at` names for each record `get_batch_at` serves. Within
  an epoch a record keeps its augmentation across batch size, shuffle order, worker split and
  resume point, and every epoch draws fresh augmentation. The default `record_indices_at` names
  records by position; `MemorySource`, the eager sources and `MixDataSourcesNode` name the
  records they shuffle, partition or mix. `per_record_keys` and the operators' raw batch path
  take the epoch.
- `Pipeline` iteration ignored `MemorySourceConfig(num_workers, shard_id)`, so every worker
  served every record. `get_batch_at`, `record_indices_at` and `len()` now follow the worker's
  partition, positions `[shard_id::num_workers]` of the global order, with global record
  indices.
- `scripts/run_tests.sh` runs its GPU pass on the GPU. It exported `JAX_PLATFORMS=cuda`, which the
  test environment ignores, so both passes ran on emulated CPU devices; it now sets
  `DATARAX_TEST_JAX_PLATFORMS=cuda`.
- `scripts/check_sync.py --fix`, `scripts/validate_examples.py --execute` and
  `scripts/distributed_test_runner.py` no longer need a `python` on `PATH`: they run jupytext,
  examples and pytest with the interpreter that runs the script. `check_sync.py --fix` used to
  report "fix failed" without saying why when `python` was missing. Importing
  `distributed_test_runner.py` no longer configures logging; its `main()` does.
- `SharderModule.parallel_transform` no longer enters the deprecated `with mesh:` context, which
  made jax warn on every call; `nnx.shard_map` already receives the mesh. The sharding guide's
  meshes name `axis_types=(jax.sharding.AxisType.Auto,)`, because `jax.make_mesh` builds
  Explicit axes without it, and a contract test fails any documented `make_mesh` call that
  does not name its axis types.
- `HFEagerSource` builds each column once from the dataset's numpy format and moves it to JAX in
  one array. It used to convert every row to its own JAX array and stack them on the device,
  which compiled a program whose input count was the row count: building MNIST's training
  split took tens of minutes (8,000 rows took 31 s). The full 60,000-row split now builds in
  about 3 s. Text and ragged columns keep the per-row conversion.

## [0.1.9] - 2026-09-10

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
