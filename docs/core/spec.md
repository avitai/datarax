# Element Specs

Helpers for building element specs and for checking data against them.

Two descriptions of the same data are kept apart:

| Helper | Describes | A host `float64` array with x64 off |
|--------|-----------|-------------------------------------|
| `array_to_spec`, `array_to_spec_strip_leading` | The data exactly as given, from array metadata, with no copy | `float64` |
| `device_spec` | The same data once converted to JAX arrays | `float32` |

A source declares the records its batches carry. `MemorySource` converts its
stored data to JAX arrays in `get_batch_at`, so it declares `device_spec` of its
storage. A streaming source that emits host arrays declares them as they are.

## Checking batches

```python
import jax
import jax.numpy as jnp
import numpy as np

from datarax.core.spec import SpecMismatchError, validate_batch

element_spec = {"image": jax.ShapeDtypeStruct((28, 28), jnp.float32)}
batch = {"image": np.zeros((32, 28, 28), dtype=np.float64)}

try:
    validate_batch(batch, element_spec, batch_size=32)
except SpecMismatchError as error:
    print(error.problems)  # ("['image']: dtype float64 != expected float32",)
```

`validate_batch` checks tree structure, per-element shapes, dtypes and one shared
record count, and reports every problem with its field path. With `batch_size`,
a short final batch passes. It reads only shapes and dtypes, so it never copies
or casts data, and it can run inside `jax.jit` while tracing without adding
anything to the compiled graph.

`Pipeline` runs this check on every batch a streaming source emits, before the
batch reaches the DAG. When a pass starts it also calls `validate_device_dtypes`
on the declared spec, so a declared `float64` field is refused while
`jax_enable_x64` is off instead of being narrowed silently. The pipeline reads
`element_spec()` once per source and x64 setting, because a streaming source may
open its backend to answer it, so a source's declaration must not change after
construction.

## See Also

- [Data Source Protocol](data_source.md)
- [Data Sources Guide](../user_guide/data_sources.md)

---

::: datarax.core.spec
