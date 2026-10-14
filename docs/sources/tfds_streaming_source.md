# tfds_streaming_source

`TFDSStreamingSource` streams a split TFDS has prepared as TFRecord (TFDS's default format), for
datasets too large for host memory, read without TensorFlow. `from_tfds` picks it for a TFRecord
copy; preparing a dataset and choosing between the two TFDS sources are on the
[TFDS source page](tfds_source.md).

```python
from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource

source = TFDSStreamingSource(
    TFDSStreamingConfig(name="imagenet2012", split="train", shuffle_buffer_size=10_000)
)
```

A pass is a Grain dataset of decoded batches, `source.pass_dataset(pass_index, key,
batch_size)`, where `key` is the pipeline's key as uint32 words on the host
(`datarax.core.prng.key_words(key)`), or `None` for file order. It computes the pass's order over record ids, reads each record's payload at its
offset in an index built once from the shard files' frame headers, checks each frame's CRCs as
tf.data's TFRecord reader does (a damaged frame raises `datarax.sources.DamagedRecordError` naming its file and
record), and decodes last. The shuffle buffer holds record ids, not records. It implements
Grain's `set_slice`: each of k worker processes computes the same order and reads and decodes
only its own batches (every k-th), so the order does not depend on k and the workers read the
data once between them. A run read by worker processes sends them the offset index as one
read-only copy in shared memory (16 bytes a record), made once for the source and removed with
it, so a worker holds none of its own.

Each record is named by its `tfds_id`, the shard file and the offset in it, so its text is
looked up again by `source.provenance(batch.indices)`.

::: datarax.sources.tfds_streaming_source

::: datarax.sources.DamagedRecordError
