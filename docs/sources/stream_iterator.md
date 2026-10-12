# stream_iterator

How a stream is read. `iter_batches(batch_size, *, num_passes=1, key=None, with_provenance=False,
read_size=None, drop_last=False)` on any stream built on `StreamingSourceBase` returns a
`StreamBatches`: one iterator over the stream's passes, cutting the records each pass yields into
host `Batch`es named by the stream (its ids, or arrival ordinals that never repeat within the
iterator), each row's pass in `epochs`. A pass's last records are completed from the next pass's
head unless `drop_last` skips them; `num_passes=None` never ends.

```python
from datarax.sources import HFStreamingConfig, HFStreamingSource

stream = HFStreamingSource(HFStreamingConfig(name="stanfordnlp/sst2", split="train"))
batches = stream.iter_batches(32, num_passes=2)
first = next(batches)
state = batches.get_state()  # {"pass", "records", "arrived", "passes_left"}

resumed = stream.iter_batches(32, num_passes=2)
resumed.set_state(state)  # serves exactly what `batches` would have served next
```

The iterator is where the read stands; the stream holds no position, so its graph definition
never changes as it is read. A pass resumed midway is read again from its start to the saved
count. An error raised while a pass is read reaches the `next` that met it, and the iterator then
refuses to go on; a new iterator starts again. `close()` closes a pass's reader stopped midway.

::: datarax.sources.stream_iterator
