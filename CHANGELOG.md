# Changelog

All notable changes to datarax are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Pipeline.raw_batches(chunk=None, *, max_chunk_bytes=None, with_provenance=False)`: unprocessed
  batches read by the host stage (`datarax.pipeline.host_stage`) and placed on the default device,
  uncommitted, as each is taken (reads run ahead): the form whose DAG (`pipe.dag`) runs inside the
  caller's differentiated step. An indexed source's batches are named on the CPU device and read
  with its stateless host read (`datarax.core.data_source.IndexedHostRead`) by Grain threads; a
  stream's run is read and decoded on one producer thread ahead of the consumer, through one
  sliceable Grain dataset for a TFDS stream and pass by pass for any other. `chunk=K` serves `(K, B, ...)` chunks of full batches, one host
  read and one transfer each, then the rest singly; `with_provenance=True` serves
  `(batch, provenance)` pairs. Nothing but the batches reaches the device, nothing transfers
  implicitly, and one Grain iterator serves a run across calls (`Pipeline.close()` ends it). Each
  returned iterator serves only the run it was created for: once `reset()`, `set_state()`,
  `close()` or a call with other options ended that run, its next `next()` raises `RuntimeError`
  naming the call, and once the run served its last batch it stops. A read error reaches the
  caller unchanged and ends the run at the batches delivered, so iterating again resumes at the
  batch whose read failed, losing and repeating nothing. A run lives as long as the pipelines it served and the iterators serving it, though a compiled-step
  cache keyed by the pipeline's graph keeps its host stage. Reads run in the caller's precision
  mode. `Pipeline.host_stage` holds where host iteration stands; its `read_threads` and
  `read_buffer` are the run's read options.
- `TFDSStreamingSource.run_dataset(schedule, key)`: a run of passes as one Grain dataset of decoded
  units numbered from the run's start (`datarax.core.data_source.BatchSchedule`), so Grain's
  process prefetch starts its workers once per run and `k` slices interleaved from the first
  serve a resumed run in order.
- `datarax.pipeline.epochs.batch_records(source, plan, *, key_base, start, epoch, size)`: the
  records a batch of an indexed source holds, by the epoch plan, each row by its own epoch's order
  (the rows past an epoch's end are the next epoch's head). The compiled session names its batches
  with it. `datarax.pipeline.epochs.HostNaming(source, plan, *, shuffled)` runs it as one
  `jax.jit` program on the CPU device for the host stage, taking the start as two uint32 words,
  so positions past `2**31` are named exactly; the source is closed over (none of its records is
  transferred), one program is compiled per batch shape, and nothing transfers implicitly.
- `datarax.pipeline.epochs.Run`: the batches a run serves, batch `ordinal` found without walking
  to it, until a recorded end epoch, so a run resumed anywhere ends where the uninterrupted run
  would. `Run.starting(plan, position, epoch)` counts the run as a session from that position.
- `record_indices_at(start, ...)` of every source (the default, `EagerSource`, `MemorySource`
  sharded or not, `MixDataSourcesNode`, `resolve_wrapped_indices`) takes the start as its two
  uint32 words `(hi, lo)`, NumPy or traced, beside a Python int and a traced int32
  (`index_words.is_word_start`, `index_words.wrapped_positions`).
- `datarax.core.prng.key_words(key)`, `fold_on_host(words, data)` and `host_device()`: a key's
  data as host words, read once with an explicit `jax.device_get`, and `fold_in` computed on the
  CPU device from them, with no implicit transfer and no read back from an accelerator.


- `mixed_source.grain_proportions(weights)`: Grain's integer proportions for mixing weights (the
  smallest scaled to 100, the others alike and truncated), the rule of `grain.MapDataset.mix`.
  `MixDataSourcesConfig.weights` keeps the weights as given, so a mix's proportions are the ones
  Grain computes from the same weights; `MixDataSourcesConfig.normalized_weights` reads them
  divided by their sum. A mix's length is Grain's rule computed exactly in integers, which can
  exceed Grain's float64 result by one.
- `DataSourceModule.index_space()`: how many record indices a source names, `len` by default.
  An in-memory source names every stored row (a worker's shard included) and a mix every record
  of its children, though an epoch serves fewer. A mix's offsets count its children's index
  spaces, so a nested mix that does not cover its own children keeps one index per record.
- `MixDataSourcesNode.get_batch(indices, *, epochs=0, contiguous=False)`: the host read of
  mixed records with the indexed sources' signature, a `Batch` named by the given words and
  epochs with draws 0, in the order named. It reads each source once with that source's own
  `get_batch`, creates no device array and changes no state; a run declared contiguous is read
  as each source's views.
- Sources with different fields mix: a mixed record carries the union of its sources' fields
  (`MixDataSourcesNode.element_spec()`, `mixed_source.union_spec`). A field some source lacks is
  `Maybe(value, present)` in every batch of the mix, zeros where a record has none, and a
  source's own `Maybe` keeps its `present`. Host columns of one field stored in different dtypes
  of one kind join at NumPy's lossless promotion. A field whose shape, device dtype or nesting
  differs between sources is refused naming both. The traced `get_records` refuses a mix whose
  sources' fields differ, naming `get_batch`.
- `datarax.core.index_words.divmod_word(a, divisor)`: the quotient and remainder of a two-word
  (64-bit) value by a word in uint32 arithmetic (Hacker's Delight `divlu`), the same code for
  NumPy and traced arrays, with x64 off. A jitted caller compiles once per divisor.
- `datarax.sources.StreamingSourceBase` (public) and `StreamChunk`: the base every stream
  builds on. A subclass reads one pass in its order (`_open_pass(pass_index, key, read_size)`, a
  generator of host columns, provenance and ids); the base serves
  `get_batch(batch_size, *, key=None, with_provenance=False, read_size=None)`, a host `Batch`
  named by the stream (`STREAM_IDS`: the ids it reports as two words; `ARRIVAL`: arrival
  ordinals, never reset), `epochs` the pass from 0, an empty `Batch` at a pass's end, and with
  `with_provenance=True` the records' strings and objects beside it. A pass is read `read_size`
  records at a time, the pipeline's batch size, whatever a pull asks for. An error raised while
  a pass is read reaches the pull that met it, and later pulls are refused naming it until
  `reset()`. `clone()` gives a stream between passes a position of its own; a stream in the
  middle of a pass refuses it. Where a stream is lives in a
  host holder outside NNX state, so no Variable holds a Python value and the graph definition
  does not move as it advances. `element_spec()` is the first record's array part as the device
  holds it. A pass's reader stopped midway is closed by `reset()`, when the stream is collected,
  and at interpreter exit before module teardown (a backend reader left suspended until teardown
  hangs there, as HuggingFace's Parquet reader does).
- `datarax.pipeline.epochs.stream_batches(pull, batch_size, *, drop_last, num_epochs)`: the
  epoch rule over a stream's passes, the rule `EpochPlan` applies to an indexed source.
- `DataSourceModule.provenance(indices)` and `DataSourceModule.record_keys(batch)`: a source
  naming records stably (`INDEXED`, `STREAM_IDS`) serves each record's strings and objects by
  its index, one immutable mapping per index (empty for a record carrying nothing but arrays),
  and hands out a batch's indices as the keys of a per-record table; a source naming records by
  arrival (`ARRIVAL`) refuses both with a `TypeError` naming its kind. `EagerSource` serves the
  provenance it holds, `MixDataSourcesNode` each record's from the source that owns it. The row
  check of host reads, `host_rows`, and the word check, `record_words`, move to
  `datarax.core.data_source`.
- `StreamingDiskSource.get_batch(indices, *, epochs=0, contiguous=False)`: a host read of the
  memory map with the eager sources' signature and checks, returning a `Batch` named with the
  given index words and epochs and creating no device array (a run declared contiguous is a
  view of the map). The traced `get_records` still serves the compiled session.
  `StreamingDiskSource` and `StreamingDiskSourceConfig` are exported from `datarax.sources`.
- `datarax.sources.eager_source.read_host_batch(columns, length, indices, ...)`: the one host
  read of indexed sources whose records sit in host arrays (`EagerSource` and
  `StreamingDiskSource` use it).
- `datarax.sources.tfds_source.open_prepared_split(name, split, data_dir)` opens a TFDS split
  prepared as ArrayRecord through TFDS's random-access reader, reading no record and importing
  no TensorFlow, and refuses a copy that is not prepared or is prepared in another format with a
  `FileNotFoundError` naming the call that prepares it.
- `datarax.sources.eager_source.parts_of_records(records)`, the one path from records to host
  columns and provenance (`MemorySource` and `TFDSEagerSource` use it), and `HostValue`, the
  holder of a host value NNX keeps out of module state and every program (`HostProvenance` is
  one).
- `datarax.core.RecordIdentity` (`INDEXED`, `STREAM_IDS`, `ARRIVAL`) and the abstract
  `DataSourceModule.record_identity`: every source declares what its record index means (a
  stable position, an id the stream reports, or the arrival ordinal), and a source that does
  not is refused at construction. The in-memory sources, `StreamingDiskSource`,
  `ArrayRecordSourceModule` and `MixDataSourcesNode` are `INDEXED`, `TFDSStreamingSource`
  `STREAM_IDS`, `HFStreamingSource` `ARRIVAL`. A source in another package declares its kind
  with a `record_identity` property returning it.
- `datarax.sources.EagerSource`, the public base of every in-memory source (`MemorySource`,
  `TFDSEagerSource`, `HFEagerSource`). It holds a record's array part as host NumPy columns and
  its non-array part (strings, bytes, Python objects) as the record's provenance, an immutable
  mapping per record in a host holder NNX keeps out of module state and out of every trace,
  never part of a batch. Its host read, `get_batch(indices, *, epochs=0, contiguous=False)`,
  gathers the records named by uint32 `(n, 2)` index words with one NumPy gather (a run the
  caller declares contiguous as views) and returns a `Batch` named with those indices and
  epochs; it reads and changes no state, creates no device array, and refuses the padding index
  and indices outside the source by name. A subclass stores its columns with `_store`.
- `datarax.core.Maybe(value, present)`: a data field a record may lack, at any depth in `data`.
  `value` holds zeros where a record has no value and `present` is a bool per record. It is a
  frozen registered pytree node of the two arrays, so batch operations, placement, specs and
  every transform carry it with its rows, and batches differing only in presence compile once.
  It has no arithmetic: `*`, `+`, `==` against a number, `jnp` reductions and `np.asarray` raise
  `TypeError`; `value_or(fill)` reads it, broadcasting `present` over the value's trailing axes.
- `state_keys.MASKED` (`state[MASKED][field]`: a present value an operator hides and keeps as the
  target) and `state_keys.IMPUTED` (`state[IMPUTED][field]`: a missing value an operator filled,
  with the field's `present` set).
- `RandomCropOperator` pads each record and crops it at an offset drawn from the record's own
  key, with torchvision's `RandomCrop` semantics: padding as one int, (left/right, top/bottom) or
  (left, top, right, bottom); modes constant (with `fill`), edge, reflect and symmetric; top and
  left drawn independently and uniformly over every valid offset. A crop larger than the padded
  record is refused. In eval mode, and with `stochastic=False`, it takes the centre crop of the
  padded record; `output_spec` reports the cropped shape.
- `FlipOperator` mirrors each record left to right or top to bottom. A flip with probability p,
  torchvision's `RandomHorizontalFlip(p)`, is `ProbabilisticOperator(probability=p)` around it.
- `functional.pad`, `functional.flip_left_right` and `functional.flip_up_down`.
- `HFEagerConfig`, `HFStreamingConfig` and `from_hf` take `cache_dir`, passed to
  `datasets.load_dataset`.
- `datarax.config` exports `load_config_from_path_with_includes`, `deep_merge_dict`,
  `SchemaField`, `ValidationError` and `CONFIG_ENV_PREFIX`.
- Benchmarks: a structural gate, `benchmarks/tests/test_structural_counters.py`, runs every
  Tier-1 scenario through the Datarax adapter at its own batch size, element shape and
  transforms, over a dataset of exactly the batches iterated, and requires that nothing compiles
  after warmup. It runs in the unit lane on every pull request. `ScenarioVariant.generate_data()`
  generates a variant's data at its configured size and `ScenarioVariant.with_dataset_size(n)`
  sets that size; every `data_generator` takes the number of records to generate.

### Changed

- `for batch in pipeline` runs on the host stage: each batch is read by Grain threads, named on the
  CPU device, placed on the device as it is taken and run through `pipeline.dag` in one compiled
  call (`datarax.pipeline.dag_call.compile_dag`), split once per iteration, compiled once per
  batch shape, the state the stages write written back. Iteration serves the records and draws the
  compiled session served, uploads no dataset, and continues where the last batch taken ended
  until the run's `num_epochs`; `reset()` starts the next run. `iter(pipeline)` is no longer a
  `PipelineIterator`: `pipeline.session()` is the compiled session. `step()`, `scan` and the
  session keep their place in the pipeline's Variables, apart from iteration's. A pipeline without
  stages serves the placed batches as they are, with no DAG call: no compile and no copy.
- `Pipeline.get_state()` / `set_state(state, /)` are the host stage's cursor, version 3, typed
  `substrax.typing.CheckpointState`: `version`, `kind`, `epoch`, `position`, `run_end_epoch`,
  `stream` (a stream's pass, records, arrivals and passes left) and `fingerprint` (batch size,
  length, `drop_last`, `num_epochs`, whether it shuffles, the seed's words, the order). They no
  longer hold stage parameters or the position Variables: stage parameters and statistics are
  checkpointed as `nnx.state(pipeline.dag)`. A state of any other layout (an unversioned
  `module_state`, a session's version 2) or of another configuration is refused, naming it; a
  refused state changes nothing. `Pipeline.batches_left()` counts the host stage's run.
- The Tier-A DAG call moves from `datarax.pipeline.iteration` to `datarax.pipeline.dag_call`, its
  helpers public (`is_per_batch_state`, `cached_step`, `state_leaves`, `run_tracking_writes`,
  `apply_writes`, `Writes`); `compile_streaming_dag` is `compile_dag`.
- `datarax.pipeline.epochs.stream_batches(pull, ...)` takes a pull returning `(Batch, provenance)`
  and yields `(Batch, provenance)` pairs, the provenance joined with its rows.
- A `TFDSStreamDataset` element is `(columns, provenance, ids, epochs)`, each record's pass beside
  its id; `pass_dataset` is a run of one pass.
- `StreamingSourceBase.read_from(cursor, batch_size, *, key, read_size)` is the read `get_batch`
  makes at the stream's own position, open to a caller keeping a position of its own
  (`StreamCursor`).
- `StreamingDiskSource` pickles as its path and opens its memory map at first use, once, under
  concurrent readers.
- The GPU test recipe is `DATARAX_TEST_JAX_PLATFORMS=cuda,cpu`: the host stage names records on the
  CPU device beside the GPU.
- A stream's pass key is folded on the CPU device from the key's host words:
  `StreamingSourceBase._open_pass(pass_index, key, read_size)` and
  `TFDSStreamingSource.pass_dataset(pass_index, key, batch_size)` take `key` as uint32 words on the
  host (`datarax.core.prng.key_words`), `pass_seed(key, pass_index)`, `pass_generator` and
  `key_integer` take the words, and `get_batch(..., key=...)` reads a typed or raw key's words once
  per pass. A pass opens under `jax.transfer_guard("disallow")`. `_source_base.typed_key` is
  removed.
- `datarax.sources._grain_bridge` is `datarax.sources._index_validation`, named for the one
  function it holds, `validate_index_batch`.
- `MixDataSourcesNode` mixes on Grain's mix. Position `k` of an epoch belongs to the source
  `grain.MapDataset.mix` selects for `k`, at the source position Grain reads there; the weights
  become Grain's integer proportions. The selection runs in uint32 words, so it traces and runs
  on the host alike, with no table of Grain's period. `len(mix)` is Grain's length, the most
  positions that serve each source record at most once (`min(len(source) * S / p)`, exact in
  integers), where it was the sum of the source lengths: an epoch no longer repeats a record or
  gives two rows one key. Each source serves its records in its own `record_indices_at` order,
  keyed by `fold_in(key, c)` when the pipeline shuffles and in order when it does not, so a
  `Pipeline(shuffle=False)` over a mix serves one fixed interleave every epoch instead of being
  refused. Mixed record indices and the source offsets are 64-bit words; `provenance` checks
  indices against the mix's index space (the sum of the source lengths).
- `MixDataSourcesNode` refuses, at construction, a source that is not `INDEXED`, one worker's
  shard of a `MemorySource` (`num_workers > 1`), a source with no records, a source whose
  `element_spec()` is not implemented, an index space reaching the padding index, an epoch past
  `sys.maxsize` and weights whose Grain proportions sum past `2**32 - 1`. A source without a host
  read, `get_batch(indices, *, epochs, contiguous)`, is refused too.
  `MixDataSourcesConfig` refuses a zero weight.
- **Streams name their own records, take the pipeline's key and honour the epoch rule.** The
  pipeline keeps no counter for a stream: it passes its key to the stream's `get_batch` when it
  shuffles (`None` otherwise), applies `drop_last` and `num_epochs` to the stream's passes as to
  an indexed source (a pass's short tail is dropped or completed from the next pass's head; a
  stream used to ignore both and serve one pass), checks each batch against the declared spec
  as the device will hold it, and runs the DAG over the stream's `Batch`.
- **`TFDSStreamingSource` streams a TFRecord copy without TensorFlow.** An offset index of each
  shard file, built once from its frame headers, locates every record; each payload is read at
  its offset, checked against the frame's CRCs as tf.data checks them (a damaged frame raises
  `DamagedRecordError`, naming the file and record; `google-crc32c` joins the `data` extra), and
  decoded by TFDS's NumPy decoder; records are named by `tfds_id`
  (shard, offset) and their text is provenance, served beside the batch and looked up by id
  (`provenance(indices)`). A shuffling pipeline orders each pass as TFDS's training read does:
  the shard files in a keyed order, interleaved 16 at a time in blocks of 16, then tf.data's
  buffer shuffle of `shuffle_buffer_size` records, every draw from a NumPy Philox generator keyed
  by `fold_in(key, pass)`; the order is computed over record ids, so the buffer holds ids, not
  records, and each record's payload is read once. A record offset an id cannot hold is refused
  when the source is built. A copy prepared as ArrayRecord is refused, naming `TFDSEagerSource`; an unprepared copy is refused naming the call that prepares a TFRecord copy.
  A pass is a Grain `IterDataset` of decoded batches (`TFDSStreamingSource.pass_dataset`,
  `TFDSStreamDataset`) that pickles with its offset index and implements Grain's `set_slice`:
  each of k worker processes computes the same order over ids and reads and decodes only its own
  batches (every k-th), so the order does not depend on k and the workers read the data once
  between them.
- **`HFStreamingSource` always streams** (`load_dataset(..., streaming=True)`); a map-style
  HuggingFace dataset is `HFEagerSource`'s. Records are named by arrival; a shuffling pipeline
  seeds HuggingFace's buffer shuffle from its key, `set_epoch(pass)` ordering each pass (it used
  a literal seed of 42). Reads are batched NumPy columns in their features' dtypes, text and
  objects as provenance beside the batch.
- `from_tfds(name, split, *, in_memory=True, ...)` picks the source by the copy's prepared
  format: an ArrayRecord copy is decoded into memory by `TFDSEagerSource`, or with
  `in_memory=False` read and decoded per batch by an `ArrayRecordSourceModule` over the split's
  files with TFDS's decoder, for a split larger than RAM; a TFRecord copy is streamed by
  `TFDSStreamingSource`; `from_hf(name, split, *, streaming=False,
  ...)` builds `HFEagerSource`, or `HFStreamingSource` with `streaming=True`.
- `ArrayRecordSourceModule` is an `INDEXED` source: a record's index is its position in the files,
  and the pipeline orders, batches and resumes it. `get_batch(indices, *, epochs=0,
  contiguous=False)` reads the named records with one batched read of ArrayRecord's
  `ArrayRecordDataSource` (the reader Grain and TFDS use) and decodes them with one call of `decode`, which now takes a batch's
  `bytes` records and returns one mapping per record; numeric values are the batch's columns and
  the rest the records' provenance (`provenance(indices)`). `paths` may be TFDS
  `FileInstruction`s. It pickles without open file handles and has no traced read, so `step()`
  over it is refused naming `get_records`.
- `source_ops.validate_eager_source_settings`, `validate_eager_config` and
  `finalize_eager_config_validation` are `validate_source_settings`, `validate_source_config` and
  `finalize_source_config_validation` (eager and stream
  configs share it).
- The HuggingFace examples that read raw text (the IMDB quick reference and the SST-2 training
  example) load with `HFEagerSource` and read each record's text with `provenance(indices)`.
- **`TFDSEagerSource` reads TFDS without TensorFlow.** It reads a split TFDS has prepared as
  ArrayRecord, through `builder.as_data_source`, in one batched read, and never imports
  TensorFlow, so a training process that reads TFDS data holds none (TensorFlow in a JAX process
  breaks JAX's NCCL collectives). It never prepares a dataset: preparing imports TensorFlow, so
  it runs once, in a process of its own, as
  `tfds.builder(name, data_dir=..., file_format="array_record").download_and_prepare()`, and a
  split that is not prepared, or is prepared only as TFRecord, is refused naming that call. The
  records become columns and provenance through the path every eager source shares: a text
  feature, such as CIFAR-10's `id`, is the record's provenance rather than a `TypeError`, so the
  examples no longer exclude it. `as_supervised=True` keeps the dataset's supervised features
  (`info.supervised_keys`) under their own names. Values keep the dtype TFDS stores: a class
  label is int64 on the host (int32 on a device while 64-bit types are off, as before). The
  50,000 CIFAR-10 training records load in about 4.5 s, where the TensorFlow loader took over
  1,000 s, and loading places nothing on a device.
  Reading needs the `data` extra; preparing needs the `tfds` extra.
  `tensorflow-datasets` is now `>=4.9.8`, the first release whose `as_data_source` takes a file
  format.
- `get_dataset_info()` of the eager TFDS and HuggingFace sources is held in a `HostValue`, out of
  the module's graph: TFDS's `DatasetInfo` compares by identity, so every `TFDSEagerSource` was a
  graphdef of its own and a jitted step over a newly built source compiled again.
- `scripts/prepare_example_datasets.py` prepares every TFDS dataset the examples read (CIFAR-10,
  Fashion-MNIST, MNIST) as ArrayRecord in TFDS's data directory. It lists first, then deletes
  just before preparing again, any copy of those three datasets prepared in another format;
  nothing else in the directory is touched. CI's dataset cache key names the format
  (`example-datasets-array_record-...`).
- The TFDS tests read an offline dataset prepared by `python -m tests.test_common.tfds_fixture`
  (with TensorFlow) and run in CI's long-running job after a step that prepares it; every other
  lane deselects them (`-m "not tfds"`), and a missing fixture fails them instead of skipping.
- IO-1's `tfds_eager` and `tfds_streaming` variants read separate data directories, an
  ArrayRecord copy and a TFRecord copy; the benchmark fixture reads CIFAR-10 through
  `TFDSEagerSource` and no longer downloads it.
- **The record order lives in core.** `datarax.samplers.index_shuffle` is now
  `datarax.core.index_shuffle` (`shuffle_positions`, `shuffle_positions_host`, `index_shuffle`),
  beside `datarax.core.index_words`; the samplers, the sources and core import it from there.
  The old path is gone, with no alias: import from `datarax.core.index_shuffle`.
- `Pipeline` routes a source on its declared kind: an `INDEXED` source through the compiled
  session, any other through the streaming path. `Pipeline.session()` refuses a stream naming
  its kind; an `INDEXED` source that does not implement `get_records` is refused at its first
  pull, naming it.
- **In-memory sources hold their records on the host.** `MemorySource`, `TFDSEagerSource` and
  `HFEagerSource` store NumPy columns: device arrays given to them are copied to the host once,
  at construction, and `HFEagerSource` builds its columns without any device array. A list of
  records is turned into columns once: numbers (Python and NumPy scalars, numeric arrays) are
  stacked per field, strings and other objects become the record's provenance, an `Element`
  record contributes its `data` (one carrying an index or state is refused), and a field whose
  shape differs between records is refused naming the field, both shapes, padding and packing.
  A dict's text or object column, and `HFEagerSource`'s text columns, are provenance, never
  refused. Indexing, iteration (in order, stateless) and Grain's batched reads return a
  record's array part as NumPy. `index_words.low_words` keeps NumPy indices on the host.
- **The pipeline owns the shuffle.** `Pipeline(..., shuffle=True)` (and `Pipeline.from_dag`)
  serves each epoch in a new order, keyed by the pipeline's epoch key; `shuffle=False`, the
  default, serves the sequential order. `record_indices_at(start, size, key)` keeps its
  signature and serves the order `key` selects, or the sequential order when `key` is `None`:
  the pipeline passes its epoch key exactly when it shuffles. The default `record_indices_at`
  shuffles a sized source by the key (so `StreamingDiskSource` now shuffles) and refuses a key
  for a source without a length. Orders are unchanged: a fixture recorded on the previous
  revision, every batch that iteration, `step()` and `scan` served over in-memory, mixed and
  memory-mapped sources, is reproduced bit for bit. `Pipeline.from_arrays(shuffle=...)` passes
  the flag to the pipeline. `resolve_wrapped_indices(start, size, length, key, *, num_workers,
  shard_id)` drops `is_random_order` and shuffles iff a key is given. A pipeline over
  `MixDataSourcesNode`, which draws its mix from the key, is built with `shuffle=True`; without
  it the first pull is refused naming `Pipeline(shuffle=True)`. The session state's
  `fingerprint["shuffled"]` is the pipeline's flag.
- In-memory sources (`MemorySource`, `TFDSEagerSource`, `HFEagerSource`) take no `rngs`: nothing
  in them is random. `Pipeline.get_state()` therefore holds no source RNG or seed state, and an
  iterator state's `rng_counts` is the pipeline's count alone. A state saved while the source
  held them is refused on restore, naming what the pipeline lacks; nothing converts it.
- **Record indices and the shuffled order are 64-bit.** `record_indices_at` (the
  `DataSourceModule` default and every source's override) and `resolve_wrapped_indices` return
  uint32 `(size, 2)`, each index as its words `(hi, lo)` (the layout of `Batch.indices`), in
  place of int32 `(size,)`; `get_records` and `name_records` take those words, and
  `record_positions` returns them. `shuffle_positions` takes and returns uint32 `(..., 2)` words
  and refuses any other positions; `shuffle_positions_host` returns uint64 in place of int64. The
  order covers lengths up to `2**64 - 1`, so the all-ones padding index is never a record, where
  it refused lengths past `2**31 - 1`; orders of shorter lengths are unchanged (a regression
  fixture recorded from the previous release's orders pins them). `index_shuffle` keeps its
  signature. The new `datarax.core.index_words` holds the conversion (`to_words`,
  `from_words`) and the two-word arithmetic host and device share with x64 off. A traced gather
  (`MemorySource`, the eager sources, `MixDataSourcesNode`) addresses rows with the low word and
  refuses a source of more than `2**32` rows; `StreamingDiskSource` reads every 64-bit index on
  the host.
- `EpochPlan.run_extent` counts batches with integer division, exact past `2**53` records, and
  `partition_length` past `2**63 - 1`, where `len(range(...))` overflowed.
- Operators that treat a field as an array refuse a `Maybe` field with a `TypeError` naming it:
  `MapOperator` over the field (a `Maybe` outside its subtree passes through), the parallel
  merges and ensemble reductions of `CompositeOperatorModule` (a `merge_fn` receives the outputs
  as they are), `WEIGHTED_PARALLEL`'s mixed fields, `BatchMixOperator`'s mixed field, the image
  operators and `RandomCropOperator.output_spec`, `LoudnessOperator`, `CrepeF0Operator` and
  `CrossModalOperator` inputs.
- Configuration environment overrides are `DATARAX_CONFIG__<KEY>[__<KEY>...]`, apart from the
  operational `DATARAX_*` variables, and replace only values the configuration already has: the
  string is read as the replaced value's type (only `true`/`false` are booleans, so
  `DATARAX_CONFIG__BATCH_SIZE=1` is the integer 1; a list reads a TOML array), and a name the
  configuration does not have, a table, or a string that does not read as the type is refused.
  `apply_environment_overrides` returns a deep copy and takes `environ=` to read a given mapping;
  `prefix` and `separator` are keyword-only. Migration: rename `DATARAX_<KEY>` overrides to
  `DATARAX_CONFIG__<KEY>`, and put every overridden key in the configuration file.
- `create_component_from_config` re-raises a constructor's `TypeError` or `ValueError` as itself,
  with a note naming the component, instead of wrapping every exception in `TypeError`; `seed`
  seeds the derived `rngs` and is not passed to a constructor that does not take it.
- Benchmarks: a batch's time spans the iterator's `next()`, where the framework produces the
  batch, as well as its materialization, so per-batch times and their percentiles include the
  pipeline's own work and add up to the timed wall clock. Every reported per-batch time moves
  up by the `next()` share; compare timings only against runs of the same instrument.
- Benchmarks: `run_scenario` refuses a compile inside the timed iteration with
  `substrax.testing.compiles.CompileCountError`, noted with the adapter, scenario and variant,
  and tears the adapter down whether or not the repetition succeeds. `PipelineAdapter.warmup`
  takes `timed_batches`, the count the next `iterate` serves: the Datarax-scan adapter compiles
  its scan at that length in warmup, where a timed call of another length compiled inside the
  timed region on every repetition. The Datarax adapter serves full batches only
  (`drop_last=True`), so a timed run reaching the end of an epoch stops there instead of serving
  a short final batch compiled inside the timed region.
- Tests count compiles with `substrax.testing.compiles` only; `tests/test_common/compiles.py`,
  which parsed JAX's compile log, is removed.
- An operator implements one method, `apply(element, key, stats) -> Element`: one record, its
  key and the batch's statistics. `__call__(batch)` is the one entry: it computes the statistics
  (`compute_statistics(batch)`, which takes the `Batch`), derives one key per record, applies the
  mode (a stochastic operator in eval mode maps `apply_deterministic`, the record unchanged by
  default) and calls `apply_batch(batch, keys, stats)`, which maps `apply` by the operator's
  `batch_strategy`. A whole-batch operator overrides `apply_batch` alone and draws from its first
  record's key. `apply_record(element, stats)` applies one record in the operator's mode; an
  `apply` that writes module state (a BatchNorm in training) or changes the record's identity is
  refused, naming the fix. Removed: the positional `apply(data, state, metadata, key, stats)`,
  `_vmap_apply`, `extract_batch_size`, `StrategyContext` and
  `ProbabilisticOperatorConfig.for_child`; a strategy is applied as
  `strategy.apply(operators, element, stats)`, and `WeightedParallelStrategy` takes its weights
  in its constructor.
- An operator keys a record from its own base key and the record's identity wherever it sits
  (`prng.record_key`, mapped as `per_record_keys`); no wrapper derives a key for a child, and a
  wrapper's own key serves only its own decision. A child draws the same values at top level, in a
  sequential composite and inside a per-record wrapper. Draws of every wrapper with a stochastic
  child change. The same operator instance twice in one wrapper is refused.
- A SEQUENTIAL or DYNAMIC_SEQUENTIAL composite calls each child on the whole batch in turn, so it
  holds whole-batch children and each child fits its statistics on its own input. Every other
  composite strategy, `ProbabilisticOperator` and `SelectorOperator` decide or merge per record and
  refuse a child that works on the whole batch, or a chain whose later child fits statistics per
  batch. A composite draws nothing itself: it is deterministic and takes no `rngs`; a
  `ProbabilisticOperator` is stochastic only when `0 < probability < 1`.
- `BatchMixOperator` mixes `data_field` only and leaves labels untouched: it writes each record's
  partner row to `states[MIX_PARTNER]` and the mixing ratio (MixUp's `lam`, CutMix's fraction of
  the image kept) to `batch_state[MIX_LAMBDA]` (`datarax.core.state_keys`), and the loss reads
  them, `lam * CE(y) + (1 - lam) * CE(y[partner])`. Integer class labels are no longer averaged.
  `label_field` is removed; a batch without `data_field`, or a CutMix field that is not
  `(B, H, W, C)`, is refused.
- `SamplerModule` is generic in the index it emits (`SamplerModule[int]`,
  `SlidingWindowSampler` a `SamplerModule[jax.Array]`); a data source's `__iter__`, `__next__`
  and `__getitem__` are declared as yielding the PyTrees they return. Pyright's
  `reportIncompatibleMethodOverride` is enabled.

- A pipeline's stage graph is `pipe.dag`, an `OperatorDag` (`datarax.pipeline.dag`): an
  `nnx.Module` holding the nodes and a static plan, mapping a `Batch` to a `Batch`, with no source,
  position or `Rngs`, so it runs inside a differentiated train step and its operators' parameters
  train with the model. Every node takes and returns `Batch`es; a node returning anything else is
  refused. `Pipeline.step()`, `scan` and iteration yield `Batch`es whose rows carry their records'
  indices and epochs; `Pipeline.__call__(data, records)` names the gathered records and runs the
  DAG, and a subclass overriding it returns a `Batch`.
- One operator path: `OperatorModule._apply_on_raw` is removed; an operator is called on a
  `Batch`. `BatchMixOperator` mixes through `apply_batch` alone.
- Removed: `RebatchNode` (a `(K, B, ...)` chunk is `batch_ops.stack(batch_ops.split(batch, K))`;
  the benchmark's rebatching scenario serves each batch as `batch_ops.split` parts),
  `datarax.pipeline.dag.run_dag` and `record_count`, and `datarax.typing.PipelineBatch`.
  `SplitField` takes and returns a `Batch`.

- `Element` and `Batch` are frozen dataclasses registered as pytrees, every field an array: they
  pass through `jax.jit`, `nnx.jit` in graph and tree mode, `nnx.jit_partial`, `vmap`, `scan`,
  `checkpoint`, `cond` and `shard_map`, and batches differing only in their values share one
  compiled program. A `Batch` holds `data`, `states`, `indices` (uint32 `(B, 2)`, each record's
  64-bit index as two words), `epochs`, `draws` and `batch_state`; `batch["image"]`,
  `batch.get(...)` and `in` read its data, and `dict`, `**`, iteration, `len` and `batch[0]` raise.
  An `Element` holds `data`, `state`, `index`, `epoch` and `draw`; `Element(x)` names no index,
  and records stacked without one are keyed by their rows. `Batch` is no longer an `nnx.Module`:
  its getters (`get_data`, `get_element(s)`, `get_states`, `get_batch_state`,
  `get/set_batch_metadata`, `update_batch_state`), `Batch.from_parts`, `slice`,
  `split_for_devices` and `compute_stats` are removed.
- Batch operations are pure functions in `datarax.core.batch_ops`: `from_arrays`, `from_stacked`,
  `stack`, `element`, `slice_rows`, `take`, `split`, `concatenate`, `mask`, `compact`,
  `record_count` and `shardings`. A padding row carries `PADDING_INDEX` (both words all ones) and
  `state_keys.WEIGHT` 0; `mask` and `compact` keep the batch's shape.
- A record's key is `per_record_keys(base_key, indices, epochs, draws)`: its epoch, its draw and
  both words of its index folded into the operator's base key, so indices past `2^32` keep
  distinct keys and a record served twice in one epoch draws twice. Every stochastic operator's
  draws change; `BatchMixOperator` mixes with its batch's first record's key.
- `DefaultBatcher` yields `Batch`es built with `batch_ops.from_stacked`, each record keeping its
  identity; a record holding a string is refused.
- Removed: `datarax.core.metadata` (`Metadata`, `RecordMetadata`, `MetadataManager` and their
  helpers), `MemorySourceConfig.track_metadata` and `MemorySource.get_with_metadata` /
  `get_batch_with_metadata` / `has_metadata`, `BatchOps`, `conditional_transform`,
  `iterative_transform`, `while_transform`, `create_element`, `create_batch_from_arrays`,
  `datarax.utils.pytree_utils`, `records_to_batch`, `record_to_element`,
  `datarax.core.spec.batch_length` (a stream's batch is checked by `validate_batch`), and from
  `datarax.typing` the re-exports of `Element`, `Batch` and `Metadata` (import them from
  `datarax`) and the aliases `StateDict`, `MetadataDict`, `ElementTransform`, `BatchTransform`,
  `DataProcessor`, `StateProcessor`, `MetadataProcessor`, `ScanFn`, `CondFn`, `WhileBodyFn`.
- Requires substrax 0.1.20. The version has one source, `pyproject.toml`; `datarax.__version__`
  reads the installed package's metadata.
- CI runs only what a change needs. A merge whose tree its pull request already tested, with every
  check of that pull request succeeded, stands the repeating jobs down through substrax's
  `already-tested` action (a pending or cancelled check keeps the merge tested). macOS runs in
  `macos.yml` nightly when `main` has moved, on demand, and on the release commit before its tag,
  never on a push or pull request. A pull request's changed lines need 80% coverage
  (`diff-cover`); coverage counts bare `pass` statements; the ruff hooks run the lock's ruff.
- The Security job audits the lockfile through substrax's `audit-lock` action, pinned by commit:
  every extra the lock resolves, in groups that respect `[tool.uv] conflicts`, each audited by a
  pinned pip-audit run in isolation with a fresh advisory cache. The ignored advisories and their
  reasons move to `[tool.substrax.audit-lock.ignore]` in `pyproject.toml`; the run fails on an
  advisory that table does not name and on an entry no advisory matches. The `dev` extra no
  longer installs pip-audit.
- A stochastic operator's only RNG state is its base key, an `nnx.RngKey` drawn once from the
  caller's stream; the private direct-call stream and `datarax.core.operator.DIRECT_CALL_STREAM`
  are removed, so applying an operator mutates nothing. A call that names no records keys on
  batch positions and repeats exactly; two operators differing only in their key share one
  compiled trace.
- Checkpoints saved under an earlier module layout are refused: the upgrade hook
  (`DataraxModule._upgrade_saved_state`) is removed, and an operator state carrying a stream
  no longer restores.
- The pipeline iterator's `rng_counts` are the pipeline's and the source's; operators contribute
  none.
- Operators follow flax's mode convention: `eval()` / `train()` and `nnx.view(module,
  deterministic=True)` set a `deterministic` flag, and a stochastic operator in that mode applies
  `apply_deterministic` (the record unchanged by default; a composite runs its deterministic
  children). Every framework path, wrappers included, applies a record through
  `OperatorModule.apply_record`, which hands a deterministic operator no key.
- `NoiseOperatorConfig`, `DropoutOperatorConfig` and `PatchDropoutOperatorConfig` are stochastic
  by default (stream `augment`) and refuse `stochastic=False`: the fixed-key "deterministic noise"
  mode is removed; evaluation mode turns them off.
- Wrapper children are constructor arguments held as graph children, never configuration:
  `ProbabilisticOperator(config, operator=...)`, `SelectorOperator(config, operators=...)`,
  `CompositeOperatorModule(config, operators=...)`. The configs keep hyperparameters only (their
  sequences stored as tuples), `resolved_for` / `for_child` / `normalized_weights(n)` complete them
  for the children, and a composite derives its strategy from the configuration on demand. Wrappers
  built the same way share one compiled trace in graph and tree mode; they compiled once per
  instance before.
- `BrightnessOperator` and `ContrastOperator` take `rngs` only when stochastic.
- Operator statistics are fixed-shape state given at construction:
  `OperatorModule(config, ..., statistics={...})` stores them as arrays at their real shape, and an
  operator built without them holds no statistics leaf. `set_statistics` replaces values of the
  same structure, shapes and dtypes and refuses anything else; `reset_statistics` is removed.
- Every operator family's input and parameter gradients are checked against finite differences
  (`tests/operators/test_gradients.py`); a Poisson noise draw passes no gradient to its input,
  as its docstring and the README state.
- A deterministic `ElementOperator`, `MapOperator` or external adapter hands its function `None`
  as the key instead of a fixed key; the function type is the keyed or the keyless shape.

### Removed

- `ArrayRecordSourceConfig`'s `seed`, `num_epochs` and `shuffle_files`, and
  `ArrayRecordSourceModule`'s own iteration and state: `rngs`, `grain_source`, `current_index`,
  `current_epoch`, `total_records`, `prefetch_cache`, `iterator_initialized`, `shuffled_indices`,
  `__iter__`, `__next__`, `__getitem__`, `get_state`/`set_state` and the stream
  `get_batch(batch_size, *, key, read_size)`. Use `Pipeline(..., shuffle=True, num_epochs=...)`
  and the pipeline's state.
- `MixDataSourcesConfig.num_sources` (the weights give the count), the `rngs` argument of
  `MixDataSourcesNode` (nothing in a mix is random), its `index` and `epoch` Variables, its
  iteration (`__iter__`, `__next__`), `reset()` and `to_grain_iter_dataset()`. A pipeline state
  saved over a mix holds those Variables and is refused.
- `datarax.sources._grain_streaming` (`mix_streaming_sources`, `data_source_to_iter_dataset`,
  `interleave_streaming_sources`, `repeat_streaming_records`, `limit_streaming_records`,
  `ensure_iter_dataset`) and the `_grain_bridge` adapters `DataraxRandomAccessAdapter` and
  `DataraxMapDatasetAdapter`, which nothing calls once the mix leaves Grain's `IterDataset.mix`.
- The stream configs' `shuffle` (the pipeline's `shuffle` decides), `HFStreamingConfig.streaming`
  and `HFStreamingSource.is_iterable_mode`/`random_order_buffer_depth`, and
  `TFDSStreamingConfig`'s `try_gcs`, `download_and_prepare_kwargs`, `beam_num_workers`,
  `prefetch_buffer` and `local_files_only` (the stream neither prepares a dataset nor runs
  tf.data). The streams' `rngs` argument, `epoch` Variable, per-record iteration (`__iter__`,
  `__next__`) and `is_random_order`.
- `from_tfds(eager=)` and `from_hf(eager=)`, and `from_tfds`'s size-based choice.
- `datarax.sources._conversion` (`tf_to_jax`, `hf_to_jax`, `convert_batch_to_jax`,
  `stack_batch_sequence`), and the stream helpers in `source_ops`
  (`streaming_apply_batch`, `batch_elements_to_dict`, `converted_filtered_record`,
  `reset_streaming_state`, `configure_stochastic_from_shuffle`,
  `validate_streaming_source_settings`, `finalize_streaming_config_validation`,
  `validate_positive_optional_int`); `datarax.pipeline.dag.record_positions`.
- `TFDSEagerConfig.try_gcs`, `download_and_prepare_kwargs`, `beam_num_workers` and
  `local_files_only`, and `from_tfds`'s `try_gcs`, `download_and_prepare_kwargs` and
  `beam_num_workers`: the eager source never prepares a dataset. Passing them raises
  `TypeError`; a stream that needs them is built with `TFDSStreamingConfig`.
- `DataSourceModule.supports_indexed_access` and `supports_streaming`, and every override:
  `record_identity` is the one declaration of a source's kind.
- The stateful host read and its state: `get_batch(batch_size, key=None)`, `reset()` and the
  `index`/`epoch` Variables of `MemorySource` and the eager sources (use the stateless
  `get_batch(indices, epochs=...)`; a pipeline owns the position and epoch),
  `DataSourceModule.get_batch_at` (use `get_records(record_indices_at(start, size, key))`), the
  private `EagerSourceBase` (use `EagerSource`), the eager helpers `eager_get_batch`,
  `eager_iter`, `eager_reset`, `eager_get_batch_default`, `eager_iter_default`,
  `build_eager_element`, `get_eager_item` and `gather_eager_batch` with their exports from
  `datarax.sources`, `source_ops.record_count`, and the epoch argument of
  `source_ops.format_source_repr`.
- Source-level shuffling, which the pipeline now owns: `MemorySourceConfig.shuffle`,
  `TFDSEagerConfig.shuffle`/`seed`, `HFEagerConfig.shuffle`/`seed`, the `shuffle`, `seed` and
  `rngs` arguments of `from_tfds` and `from_hf`, `is_random_order`/`set_random_order` on
  `MemorySource` and the eager sources, `MemorySource`'s drawn host-shuffle seed (and its
  `_shuffle_seed`/`_shuffle_seeded` state), `source_ops.shuffled_index_for_position` and
  `source_ops.validate_seed_range`, and the `shuffle`/`seed` parameters of the eager helpers
  (`eager_iter`, `eager_get_batch` and their defaults). Use `Pipeline(..., shuffle=True)`; a
  removed config field raises `TypeError`. Iterating a source directly serves its records in
  order. The streaming sources keep their own shuffle.
- `PipelineSchema` and `NNXComponentSchema`: no field of either was ever read, so neither
  validated anything. Pipelines are built in Python; define a `ConfigSchema` for the parameters
  you configure. `examples/config/config_example.py` and its notebook, which exited on a
  configuration file the repository never had, are removed with them.
- `scripts/audit_lock.py` and its tests: the audit and its tests live in substrax's
  `audit-lock` action.

### Fixed

- TFDS records decode on the protobuf runtime a plain install selects (`upb`). The managed env
  file, the test conftest and the TFDS sources each chose protobuf's pure-Python runtime, which
  nearly doubles TFDS's per-record decode (CIFAR-10: about 130 us against 72 us). The sources
  chose it after protobuf had loaded, so their own process stayed on `upb` and only the child
  processes it started ran pure Python. Rerun `./setup.sh` to regenerate `.datarax.env`.
- `MixDataSourcesNode.get_records` refuses a mix whose index space passes `2**31 - 1`, which its
  int32 gather cannot address, naming `get_batch`; such indices reached the wrong child.
- The docs no longer link to `performance/synchronization.md`, a page removed with the
  synchronization helpers; `mkdocs build --strict` warned on both links.
- `HFEagerSource` calls `load_dataset` once, taking its info and its columns from the one
  dataset (it loaded the dataset twice). The HuggingFace quick reference no longer suggests an
  `HFEagerConfig(subset=...)` field, and the data-sources guide shows a dataset configuration
  passed as `download_kwargs={"name": ...}`.
- `ConfigSchema` reads the fields of its base classes (a subclass lost them), gives each
  validation its own copy of a default, validates a nested schema into its defaults (including an
  absent optional one) and reports its failure by field path, widens an integer to a float field,
  and refuses a boolean for an integer or float field. An override no longer changes the caller's
  nested dictionaries, and `deep_merge_dict` returns dictionaries that share no container with its
  inputs.
- `LoudnessOperator` computes DDSP's loudness (magenta/ddsp `compute_loudness` / `power_to_db`):
  the A-weighted power averaged over frequency, then converted to dB and clamped at `-range_db`.
  It averaged per-bin dB before clamping, which gave near-silent bins hundreds of dB down the
  weight of the whole spectrum and an ill-conditioned input gradient. Framing, window and the
  A-weighting floor follow the reference too: center padding with `n_samples // hop + 1` frames,
  a periodic Hann window, A-weighting floored at -80 dB. The defaults are DDSP's (`n_fft=512`,
  `ref_db=0.0`, `range_db=80.0`).

- A restore refused for a mismatched state no longer changes the module: validation now covers
  every level of the state (dictionaries of stages included) before any value is written.
  Before, a pipeline whose stage subtree did not match kept the position and the earlier
  stages' parameters it had already written.

- Operators follow JAX's and Flax NNX's dtype rules, with `jax_enable_x64` on or off. An operator
  without parameters returns each field in the field's dtype: a value it draws or a constant it
  holds is applied in the data's dtype. Under x64, a float32 input came back float64 from 32 of
  the 51 operator kinds the per-record fixture records, and 5 raised (patch dropout; the branches
  of `ProbabilisticOperator`, CONDITIONAL and BRANCHING composites); `MemorySource`'s shuffle
  seed raised `OverflowError`. `LoudnessConfig` and `CompositeOperatorConfig` (for learnable
  weights) take Flax's `dtype` and `param_dtype`: parameters are created in `param_dtype`
  (float32) and the computation runs in `dtype`, or the promotion of input and parameters, so
  bfloat16 audio computes loudness in float32 (JAX's FFT needs it) and float64 in float64.
  `functional.rotate` computes its geometry in at least float32 and in float64 for a float64
  image. With x64 off, every value is unchanged.

- `scripts/check_sync.py` compares markdown cells as well as code cells, reading the script
  with jupytext. A notebook whose prose was stale reported as synced, and `--fix` left it
  stale.

- The HuggingFace sources describe `data_dir` as `datasets.load_dataset` uses it: a folder inside
  the dataset's repository, not a storage location (`cache_dir` is). `examples/hf_datasets_test.py`,
  which passed a cache folder as `data_dir` and so failed every load while exiting 0, is
  replaced by an integration test that streams six Hub datasets and checks their fields.
- `functional.random_crop` takes `padding`, `padding_mode` and `fill`, and refuses a crop larger
  than the (padded) image, which it cropped to the image's size before; `functional.center_crop`
  refuses one too and rounds its offset as torchvision's `CenterCrop` does. The random flips take
  a traced probability. The benchmark adapter's `RandomCrop` (torchvision's
  `RandomCrop(size, padding=size // 8)`) and `RandomResizedCrop` draw their offsets through
  `functional.random_crop`: their top and left offsets were always equal and the last valid
  offset was never drawn. Its `RandomHorizontalFlip` leaves fields with fewer than two axes
  alone. The `ElementOperator` docs flip images on their width axis (`[:, ::-1]`), not their
  channels.
- The examples flip images with `FlipOperator` inside `ProbabilisticOperator` instead of
  hand-written flip functions (the operators tutorial keeps one as its lesson in writing a
  stochastic `ElementOperator`), and the operators tutorial lists `RandomCropOperator`. Their
  docs pages carry a new GPU run. Output quoted in prose on the DAG fundamentals, sampling,
  advanced operators and resumable training pages is the run's.
- Example docs pages show each script's code verbatim with the output a GPU run of it printed
  (an NVIDIA L40S; the sharding quickref on two), and their figures come from that run: 136
  output lines on 21 pages had come from older runs or had been typed. Scripts' expected-output
  comments carry the values the run printed. The HuggingFace quickref describes its eager load,
  the tutorial compares an `HFStreamingSource` with an eager source, and the IMDB quickref
  shuffles its pipeline source (the train split starts with negative reviews).

## [0.1.17] - 2026-09-25

### Added

- `Pipeline.from_arrays(data, *, batch_size, seed, shuffle=False, drop_last=False,
  num_epochs=1)`: a pipeline over a dict of in-memory arrays in one call, the `MemorySource` and
  the stage-less linear constructor seeded with `seed`. It serves the batches the two-step
  construction serves, resumes from `get_state`, and a training step over it compiles once.

### Changed

- Requires substrax 0.1.18 and calibrax 0.1.14, the latest releases.
- The checkpoint-state protocol is substrax's `substrax.typing.Checkpointable`, the same
  `get_state()`/`set_state(state)` contract; `datarax.typing.Checkpointable` is removed, so a
  training loop checkpoints a pipeline, its callbacks and extensions through one protocol.
  `CheckpointableIterator` builds on it.

### Fixed

- The distributed-training guide no longer imports `substrax.spmd.reduce_gradient_tree`, which
  substrax 0.1.17 removed; it says what replaces it: no reduction under `nnx.jit` with a mesh,
  and the loss averaged before differentiating inside `jax.shard_map`.

## [0.1.16] - 2026-09-24

### Fixed

- An in-memory source's length is read from its data. `MemorySource` and the eager HF and TFDS
  sources copied it at construction while `data` stays a public attribute: after the data was
  replaced (8 records by 12), `len(source)` stayed 8 and a pipeline served 8 records and
  stopped. `MixDataSourcesNode` froze its total and per-source offsets while it sampled each
  child's current length, so a child that grew produced record indices colliding with another
  source's, and per-record randomness is keyed on them. All read the length now through
  `datarax.sources.source_ops.record_count`, which replaces three differing implementations
  (MemorySource's check, HF's first-column read, TFDS's first-key shape). Sources over storage
  that cannot change while they exist (the Grain adapter's per-pass snapshot, memmaps, split
  metadata) keep reading it once. An `EagerSourceBase` subclass no longer sets `length`.
- A module's checkpoint (`DataraxModule.get_state`) holds its `nnx.Variable` state only. It held
  every leaf `nnx.state` reaches, including a `MemorySource`'s data arrays, so each checkpoint
  carried the whole dataset (25.6 MB for a 25.6 MB source) and `set_state` could not restore it
  ("Cannot restore state at data.x").
- A restored `MemorySource` resumes the same records. The host shuffle's seed lived in a plain
  attribute outside the checkpoint and each epoch drew a new one, so a restored source drew a
  different seed; it is now drawn once, on first use, into a checkpointed Variable, and epoch `e`
  serves the order of `(seed, e)`. The last batch of an epoch in stateful `get_batch` is taken
  from its own epoch's order; the epoch advanced before its records were read. `reset()` returns
  to epoch 0 and serves that epoch's order again.
- `Pipeline.scan` reads NumPy source data from the device copy `step()` and iteration use,
  uploaded once. It gave `nnx.scan` the live pipeline, whose NumPy arrays were call arguments, so
  every call uploaded the whole dataset (65.6-70.7 ms per call at 1 GiB on an RTX 4090, now
  2.1-2.3 ms). The scan now runs on a view of the pipeline that shares its Variables. With
  device data a call costs about 0.3 ms more, for building that view.
- A data column that is itself a mapping is refused with a `TypeError` naming the column. Its
  key count was read as its length, so the refusal named that count as a length, and a mapping
  with as many keys as there were records was accepted.
- The performance page and `Pipeline`'s tiers carry measured costs, and lead with the data-loader
  loop (`for batch in pipeline` into a train step written as Flax writes it). They state that a
  pipeline passed into your own `nnx.jit` step copies its dataset on every call, which the page
  previously described as absorbed by the outer trace. They also state what tree mode (FLIP 5310)
  changes for your own transforms, and why a pipeline inside `nnx.grad` must be an argument.

### Changed

- **No batch holds padding; epochs end the way tf.data and Grain end them.** A random-access
  batch crossing the end of an epoch was padded with the epoch's first records and flagged in
  a `valid_mask` leaf. Now:
  - `drop_last=False` (the default) is `repeat().batch()`: the batch is completed from the head
    of the next epoch's order, so every epoch serves each record exactly once, and a batch larger
    than the source spans several epochs. A session serving `num_epochs` epochs stops after
    exactly `num_epochs * len(source)` records; its final batch is short and compiled at its own
    size.
  - `drop_last=True` is `batch(drop_remainder=True).repeat()`: the records short of a full batch
    are skipped and the next batch starts the next epoch.
  - `step()` and `scan(length)` never run out: the compiled step starts the next epoch itself.
    `scan` no longer refuses a length beyond the epoch.
  - The `valid_mask` leaf is removed; drop any masking of pipeline batches. `Batch.valid_mask`,
    `Batch.from_parts(valid_mask=)` and the `valid_mask` leaf of `batched_spec` /
    `BatcherModule.batch_spec` go with it: their only producer was the pipeline's padding. A
    non-dict element spec's batch spec is now the batched spec itself, no longer wrapped under
    `"data"`.
  - `len(pipeline)` counts the batches a whole run serves (`ceil(k*N/B)`, or `k*floor(N/B)`
    under `drop_last`), the same number as before for one epoch. A stream (`num_epochs=None`)
    has no length. `batches_left()` counts what a session started now would serve, and is
    host-only.
  - Refused at construction: a source without records, and `drop_last=True` with a
    `batch_size` above the source's length, which serves no batch.
  - A stochastic operator keys each record on its own epoch: `per_record_keys` takes a
    per-record epoch array, and `BatchMixOperator` keys a batch on its first record's epoch.
- **Indexed sources implement `get_records(indices)`, replacing `get_batch_at` as the method a
  source defines.** `DataSourceModule.get_batch_at(start, size, key)` is now the base-class
  composition `get_records(record_indices_at(start, size, key))`, and a source implementing
  `get_records` is what `supports_indexed_access()` reports. The pipeline names each batch's
  records once and hands the same indices to the gather and to the stages: before, the source's
  shuffle ran inside both `get_batch_at` and `record_indices_at`, and only an XLA merge of the
  two copies kept it to one evaluation per step, a merge the epoch-boundary branch defeats.
  `StreamingDiskSource.read_batch` is renamed `get_records`.
- The device shuffle runs faster on a GPU with the same orders. XLA drives a GPU `while_loop`
  from the host, one predicate read per iteration, and the shuffle's cycle-walk needed about five
  per batch: more than half of a batch's time. Off the CPU, `shuffle_positions` now runs its first
  passes in a `fori_loop` of static trip count, enough that the loop's predicate is read once but
  for a 1e-3 chance, chosen by `jax.lax.platform_dependent`; the CPU keeps the loop. The cipher's
  24 rounds run in `lax.scan(unroll=True)` instead of a Python loop, so they are traced once. GPU
  per batch: a session 140 -> 110 us, a stream 210 -> 115 us, `scan` 208 -> 101 us per step; first
  batch trace 62 -> 41 ms and compile 713 -> 593 ms; peak device memory unchanged.
- The step names each batch's records with one `record_indices_at` vmapped over every epoch the
  batch can touch, with no conditional: `step()`, `scan` and iteration sessions run one program and
  serve bit-identical batches.
- **A `MemorySource` batches dict-of-arrays data only.** A list of records (dicts, `Element`s,
  strings of any length) has no columns to gather a batch from: through a pipeline a list of dicts
  raised `ValueError`, a list of numbers became a bare-array "batch", and `element_spec()` declared
  a dict the batches never were. List data stays a record store for indexing, iteration and the host
  `get_batch`; `supports_indexed_access()` is `False` for it, `get_records` refuses it naming the
  dict form, and a pipeline over it is refused when iteration starts. The user guide, installation
  page and checkpointing guide examples that put list data into a `Pipeline` use dict data.
- **Streaming is a declared capability**, `DataSourceModule.supports_streaming()`, like
  `supports_indexed_access()`. The pipeline inferred it from any callable `get_batch`, and
  `MemorySource.get_batch` (a host record API that wraps around instead of ending) would have
  streamed forever; `MemorySource` declares `False`. A streaming source whose `get_batch` returns a
  non-mapping is refused naming it.
- Batch types: sources return `DataDict` (`get_records`, `get_batch_at`); pipeline outputs are
  `PipelineBatch` (`datarax.typing`, field names to arrays or pytrees of arrays): `step()`,
  iteration, `Pipeline.__call__` and the compiled step body.
- **A pipeline is checkpointable**: `Pipeline.get_state()`/`set_state()` implement the
  `Checkpointable` protocol, so `IteratorCheckpoint` saves and restores a tuned pipeline -- every
  stage's parameters, the source's state and where iteration stands, no data -- for inference or
  further training. The guides documented `checkpoint.save(pipeline, ...)` while `Pipeline` had no
  `get_state`. The state logic moves out of `DataraxModule` into
  `datarax.core.module.module_state` / `restore_module_state`, which both call; each
  `DataraxModule` in a pipeline still upgrades its own earlier layout.
- `CheckpointableIteratorModule` is removed. It was the base of data sources that iterate while
  sources were iterators; `DataSourceModule`, also a checkpointable `DataraxModule`, took that role,
  and nothing subclassed it. What it added worked against a checkpoint: four nullable fields, one
  (`current`) holding a data item, one (`idx`) duplicating `position`, and a `reset` to `None`. A
  resumable host iterator is a `DataSourceModule` whose position is an `nnx.Variable`; the
  checkpointing guide shows it, replacing an example that raised `TypeError` (no config) as
  written. The `CheckpointableIterator` protocol is unchanged.
- `Pipeline.__call__(batch, records=None)`: `records` (`datarax.pipeline.dag.Records`: each
  row's index and epoch) is what the step served; a direct call without it names the records at
  the current position. A subclass overriding `__call__` accepts the argument.
- `MixDataSourcesNode` names each record by its child's record index and gathers exactly that
  record. It named the child's position, while a shuffled child served a different record at
  that position, so per-record randomness was keyed on another record.
- A pipeline's epoch rule lives in one place, `datarax.pipeline.epochs.EpochPlan`: exhaustion,
  where the next batch starts, how a position advances, the most epochs a batch can hold, and
  how many batches a run serves.
  `Pipeline.epoch_plan` builds it from the source's current length, so `len(pipeline)`,
  `batches_left()`, `reset()`, the compiled step and iteration sessions agree and follow a length
  that changed after construction; the pipeline no longer caches the length. Iteration sessions
  held a second copy of the exhaustion and rollover rules and a length frozen at session start.
- `datarax.pipeline.iteration` no longer knows `Pipeline`: `PipelineIterator(module, body=,
  plan=, position=, epoch=, shuffled=)`, `next_batch(module, body, size)` and
  `compile_streaming_dag(stages, position, epoch, plan)` take what they use, and
  `Pipeline.session()` returns a typed, checkpointable session (`iter(pipeline)` returns it for a
  random-access source). The two modules imported each other, the cycle hidden behind a
  `TYPE_CHECKING` import; an import-linter contract now keeps `pipeline` above `iteration` above
  `epochs`. The compiled step runs `type(pipeline)._next_batch`, so a subclass's override is
  honored. `declared_spec` moves to `datarax.core.spec`. The persisted iterator state is
  unchanged.
- The examples, benchmarks and scripts start a new epoch with `Pipeline.reset()` instead of
  writing the pipeline's private position.
- `DataraxModule.get_state`/`set_state`/`clone` and `Pipeline.scan` name graph mode
  (`graph=True`; `graph_updates=True` for `nnx.scan`, whose `StateAxes` need it), so they keep
  working when Flax makes tree mode the default: a module built from one `nnx.Rngs` shares
  Variables, which tree mode rejects.
- `Pipeline.step()` runs the compiled step iteration sessions use instead of `nnx.jit`, and copies
  none of the source's arrays. The `nnx.jit` form wrote every Variable back on every call, so each
  step copied the whole dataset on the device, and replaced a NumPy source's arrays with device
  copies. Now device data is read in place, NumPy data is uploaded once per array and stays NumPy
  in the source, and `step()` and iteration share one device copy and one compiled step. A
  structural change between calls (batch size, a replaced stage, `train()`/`eval()`) is honored;
  a stage adding state while it runs is refused, as iteration already refused it. `step()` inside
  `nnx.jit`, `nnx.grad`, `nnx.scan`, `nnx.vmap` and a functional `jax.jit` traces into the caller's
  program. Per batch of 256 64x64 images, before / after: GPU 0.57 / 0.47 ms at 64 MiB, 3.6 /
  0.62 ms at 1 GiB, 3.6 / 1.24 ms with ten stages; CPU 87.8 / 4.2 ms at 1 GiB.
- The compiled iteration and `step()` paths split the pipeline in graph mode explicitly
  (`graph=True`), so they keep working when Flax makes tree mode the default: a pipeline whose
  source, stages and itself are built from one `nnx.Rngs` shares Variables, which tree mode
  rejects.
- A shuffled source's order is a keyed bijection computed per record,
  `datarax.samplers.index_shuffle.shuffle_positions`: CCCL's Feistel bijection (the
  VariablePhilox cipher of Mitchell et al., "Bandwidth-optimal random shuffling for GPUs", ACM
  TOPC 2022, as in `thrust::shuffle`) ported to JAX, with cycle-walking into the dataset's range.
  A shuffled batch costs O(batch) at every dataset size and a step stores and writes back no
  order. The permutation it replaces cost O(N) per batch: a continuous stream asked a one-key
  cache for two alternating epoch keys and recomputed the permutation four times per batch
  (135 ms per batch of 256 at 65,536 records on CPU), and even a cache hit wrote the whole order
  back from every step (2.0 ms per batch at 4M records). Shuffled orders differ from earlier
  releases for the same key. `resolve_wrapped_indices` and the stateless `get_batch(key=...)` of
  the eager sources use it.
- The host-side shuffles run the same cipher: `shuffle_positions_host(positions, length, seed,
  epoch)` in NumPy, and `index_shuffle(index, seed, num_elements, epoch)` for per-element
  callers, served from cached blocks. They replace Grain's `index_shuffle`, which extends Simon's
  rotation constants to word sizes the cipher does not define (at 14-bit words every round is
  linear in parity) and is not a bijection when `num_elements - 1` is a power of two (65,537
  records serve one record twice). An epoch's order is that of `fold_in(key(seed), epoch)` --
  the device path's key, so host and device serve one order -- where Grain's `seed + epoch`
  made one seed's second epoch the next seed's first. `ShuffleSampler`,
  `EpochAwareSamplerModule`, the eager sources' iteration and stateful `get_batch`, and
  `MemorySource` use it; `MemorySource`'s stateful `get_batch` no longer materializes the
  epoch's order.

### Removed

- `EpochOrderCache` and the `order=` parameter of `resolve_wrapped_indices`: no order is stored.
- `datarax.core.element_batch.BatchView`: nothing has produced one since the executor it served
  was replaced by `Pipeline`, which yields plain dicts.
- The `grain` dependency of `datarax.samplers.index_shuffle`.

## [0.1.15] - 2026-09-21

### Security

- The lock moves anyio from 4.13.0 to 4.14.2 for CVE-2026-63374 and CVE-2026-64847; nothing else moves. 4.14.2 is the first fixed release; 4.15.1 needs typing-extensions 4.16.0, which a single-package upgrade does not allow to move.

### Added

- The `wandb` extra, which the benchmark suite's W&B export path needs. calibrax 0.1.10 turned
  its exporter into one that raises `ImportError` naming its own extra, so `benchmarks/export.py`
  and everything importing it stopped loading without wandb. `setup.sh` and the CI jobs that
  collect `tests/` install it, which also makes the W&B export test run rather than skip, as it
  had in every CI run to date.

### Changed

- Requires `calibrax>=0.1.11` and `substrax>=0.1.16`; the lock moves calibrax from 0.1.10 and
  adds wandb with its dependencies, and nothing else moves. calibrax 0.1.11 replaces
  `StatisticalAnalyzer` with `summarize()`, which draws no random number, so the benchmark
  stability report takes its coefficient of variation from a function that needs no PRNG key;
  record metadata is typed, so the benchmark charts read their configuration through
  `calibrax.core.read_metadata`; and `WandBExporter.log_figures` is `log_images`, which takes
  `wandb.Image`. substrax 0.1.15 moves `discover_examples` to `substrax.examples`.
- From calibrax 0.1.7 `TimingCollector` waits for each batch's arrays with
  `jax.block_until_ready` when no `sync_fn` is given, so `datarax-benchmark` takes that
  default instead of passing the same wait itself, and the benchmarking guides no longer
  say to pass `sync_fn` for GPU timing. calibrax 0.1.7 also removed
  `profiling.measure_execution_time`, which datarax never used, and changed the
  macro-averaged F-scores and the FID and BERTScore values, which datarax does not compute.
- The benchmark suite's `PipelineAdapter` no longer inherits calibrax's `BenchmarkAdapter`. That
  base wraps a target handed to its constructor and is selected by a type predicate over it,
  while a pipeline adapter is built with no arguments, keyed by name, and constructs the pipeline
  under test from a scenario; nothing in this repository called `can_adapt`, `adapt`,
  `AdapterRegistry` or `.target`.
- The distributed scaling benchmark merges `--xla_force_host_platform_device_count` into
  `XLA_FLAGS` by flag name through `substrax.runtime.merge_xla_flags` instead of appending it, so
  a caller's own device count is not silently duplicated.
- Ruff resolves first-party imports from `src` and the packages named in `known-first-party`,
  rather than from whatever directories exist at the repository root, where a generated one
  sorted an import differently on a developer's machine and in CI.

### Removed

- The test helper `measure_pipeline_throughput`, which nothing called and whose default sync
  waited on a new scalar instead of the batch.
- `ITERATOR_STATE_FORMAT2`, and reading a checkpoint root written by datarax 0.1.11 or earlier.
  substrax reads one format, so the layout that split an older payload into items has nothing to
  describe. With it go the fixture generator, the pinned environment that installed an old
  substrax to write one, the CI step that ran that before four jobs, and the tests over them.
- `CUDA_VISIBLE_DEVICES_FOR_TF` from every example, notebook and documentation page that set it.
  TensorFlow reads `CUDA_VISIBLE_DEVICES`; nothing reads the name with the suffix. Each example
  already called `tf.config.set_visible_devices([], "GPU")`, which is what keeps TensorFlow off
  the GPU.

### Fixed

- The `local_files_only` and auto-detect source tests no longer read a dataset to assert that an
  argument reaches a call. Each patched one call on the path and left the one that loads data
  real, so on a machine with the named dataset prepared they materialised a whole split — minutes
  of I/O inside a unit test, and instant everywhere else.

## [0.1.14] - 2026-09-18

### Changed

- Requires `substrax>=0.1.11`; the lock moves it from 0.1.10 and nothing else. 0.1.11 caps
  jax below 0.11.2, whose renamed `jax.experimental.hijax.HiPrimitive` flax 0.12.9 imports
  at module load, and a resolver given `substrax>=0.1.10` keeps jax 0.11.2 and picks 0.1.10
  instead, so a fresh install of datarax failed on `import datarax` until the floor moved.

### Security

- The lock moves GitPython (skypilot's, in the `automation` extra) from 3.1.50 to 3.1.62 for
  GHSA-284h-m62q-gf8w, GHSA-7833-fr7j-v32q, GHSA-8mcc-hrx5-hvxc, GHSA-3wxw-xv34-2frg and
  GHSA-5xxx-qhh7-9287, and mkdocs-material (the `docs` extra) from 9.7.6 to 9.7.7 for
  GHSA-xvg9-69gf-fjrf. Nothing else moves.
- The lock moves soupsieve to 2.8.4, mistune to 3.3.3, pymdown-extensions to 11.0.1 and ray
  to 2.56.0, the fixed releases of their open advisories; nothing else moves. Five advisories
  stay open with an upstream cause: aiohttp 3.10.1 and wheel 0.45.1 are held by the
  `automation` extra, where vastai-sdk 0.2.x pins aiohttp exactly and every 1.x release of
  the underlying `vastai` pins cryptography 49.0.0, which the `cryptography>=50` constraint
  refuses; transformers 4.57.6 is capped below 5 by mosaicml-streaming (the `benchmark`
  adapters); torch stays below 2.11 for CUDA 12 wheels; paramiko has no fixed release. None
  of those packages is imported under `src/`.

## [0.1.13] - 2026-09-17

### Changed

- `IteratorCheckpoint` writes substrax's checkpoint format 3: the state is the checkpoint's
  `data_iterator` item, `save` and `save_if_due` take the record's `epoch` as a keyword, and
  `metadata` is the record's `extra` (a key naming a record field, `epoch` among them, is
  refused); `save` returns the checkpoint's directory as a `Path`, and restoring a step the
  directory does not hold raises `substrax.checkpoint.CheckpointNotFoundError`. A root
  written by datarax 0.1.11 or
  earlier restores unchanged through `ITERATOR_STATE_FORMAT2`, which
  `substrax.checkpoint.upgrade_checkpoints` also takes to rewrite such a root.
- The resumed-training guide saves the model, the optimizer and the loader state as named
  items and reads them back through `Checkpoint.items` and `Checkpoint.metadata`.
- Requires `substrax>=0.1.10`; the lock moves it from 0.1.8.

## [0.1.12] - 2026-09-17

### Added

- `Pipeline(..., drop_last=False, num_epochs=1)` settles the last batch of an epoch. A
  random-access source keeps every batch at `batch_size`; under the default the last batch is
  padded with rows from the start of the order and every batch carries a top-level
  `valid_mask` leaf (`(batch_size,)`, bool) marking the rows past the epoch's end invalid,
  attached after the stages run so a masked loss ignores the padding and the stages never
  see the mask (a mask a batcher stage already set is combined with it). `drop_last=True`
  serves floor(N / B) batches, PyTorch's rule. `num_epochs=k` makes `iter(pipeline)` serve `k`
  epochs, advancing the epoch at each boundary as `reset()` does; `num_epochs=None` is a
  continuous stream whose boundary batches hold the tail of one epoch and the head of the
  next, no row padding, and whose iterator state reports the crossed epoch. `len(pipeline)`
  is the batches per epoch, `batches_left()` what the epoch still holds, and `scan(length)`
  refuses a `length` beyond it (a continuous stream scans any length). A streaming source
  carries an all-true mask over the rows it yields.
- `EpochOrderCache`, the permutation one epoch serves, computed once per epoch key and
  reused by every batch: `MemorySource` and the eager sources (`EagerSourceBase` subclasses
  build one over their length) read it through `resolve_wrapped_indices(..., order=...)`,
  so the O(N) permutation runs once per epoch instead of once per batch, and the served
  order for a given key is unchanged. A source built on `resolve_wrapped_indices` without
  the cache keeps its behaviour.

### Changed

- Iterator state is version 2: `get_state()` adds `fingerprint` (`batch_size`, `length`,
  `drop_last`, `num_epochs`, `shuffled`), and `set_state()` refuses a state produced under a
  different configuration, naming the first field that disagrees; a version-1 state, which
  has no fingerprint, is accepted and its counts upgraded as before.
- The last batch of an epoch no longer wraps silently: the rows past the end are still served
  from the start of the order, so batch shapes are unchanged, but they are marked invalid
  in `valid_mask`, and a batch is one leaf richer.

### Removed

- `datarax.performance.synchronization` (`block_until_ready_tree`, `copy_to_host_async_tree`)
  and the `performance` exports of both; `jax.block_until_ready` and `jax.copy_to_host_async`
  do the same over a pytree. The CLI benchmark and the benchmark scripts wait through
  `jax.block_until_ready`.

## [0.1.11] - 2026-09-17

### Changed

- A component built from a seeded configuration receives its `nnx.Rngs` from
  `substrax.rng.rngs_from_seed(seed, DEFAULT_RNG_STREAMS)`: each of the `augment`, `dropout`,
  `params`, `shuffling` and `default` streams is derived from the seed and the stream's name,
  so a stream's key no longer depends on the other streams present. The keys differ from the
  split-derived ones of earlier releases, so a recorded run seeded through the registry does
  not reproduce bit for bit across this change. A configuration with neither `rngs` nor
  `seed` is seeded with zero, as before. Requires `substrax>=0.1.8`.

### Removed

- `datarax.utils.prng`, with `create_rngs`. Build streams with
  `substrax.rng.rngs_from_seed(seed, DEFAULT_RNG_STREAMS)`; `DEFAULT_RNG_STREAMS` is a tuple
  in `datarax.core.prng`, beside `per_record_keys`.

## [0.1.10] - 2026-09-16

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

- `RotationOperatorConfig` uses one parameter per mode, as the brightness and contrast configs
  do: a stochastic operator draws each record's angle from `angle_range` (default
  `(-15.0, 15.0)`), a deterministic operator applies `angle` (default `0.0`), and each config
  refuses the parameter its mode does not use. A deterministic operator given `angle_range`
  used to rotate every image by the range's midpoint, so the symmetric ranges the augmentation
  examples passed without `stochastic=True` rotated by 0° while the examples called the result
  a random rotation; those examples and their docs pages now build the operator stochastic.
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
- CI's integration, end-to-end and performance jobs depend on the lint job alone and run beside
  the unit-test matrix instead of after it. They built their own environments and shared no
  artifact with the unit jobs, so the ordering only delayed their verdicts by the slowest unit
  cell; `coverage` still joins every tier.
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

- `BrightnessOperator` and `ContrastOperator` honour `clip_range=None`. `functional.adjust_brightness`,
  `adjust_brightness_delta` and `adjust_contrast` clipped to `[0, 1]` unconditionally, so the
  option documented as "no clipping" changed nothing for these operators; the functions now
  return the raw adjustment and the operators clip to their `clip_range`, `(0.0, 1.0)` by default,
  so their default output is unchanged. `color_jitter` still clips its final result.
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
