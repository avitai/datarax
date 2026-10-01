# Index Words

A record's index is a 64-bit integer, held as two uint32 words `(hi, lo)`: the
layout of `Batch.indices`, of `record_indices_at`, and of the positions and
records of the shuffled order. JAX runs with 64-bit types off by default, and
`fold_in` takes 32-bit data, so the words are what reaches the device; the
arithmetic on them (addition, comparison, products built from 16-bit limbs)
runs the same in NumPy and in JAX, with x64 off.

```python
import numpy as np

from datarax.core.index_words import from_words, to_words

words = to_words([5, 2**32 + 5])  # uint32 (2, 2): [[0, 5], [1, 5]]
from_words(words)  # uint64 [5, 4294967301]
```

A source holds at most `2**64 - 1` records (`MAX_RECORDS`): the all-ones
index, `PADDING_INDEX`, marks a row that is not a record. A gather traced on
the device addresses rows with the low word, so an in-memory source holds at
most `2**32` rows; `StreamingDiskSource` reads on the host and takes every
64-bit index.

## See Also

- [prng](prng.md) - Per-record keys fold both words of the index
- [element_batch](element_batch.md) - `Batch.indices` and `PADDING_INDEX`
- [data_source](data_source.md) - `record_indices_at` and `get_records`

---

::: datarax.core.index_words
