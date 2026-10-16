# Block Order

The global order sends consecutive positions to unrelated records, so a source
read from storage larger than the page cache reads a device page for every
record. A block order keeps each batch inside a few runs of consecutive records,
which a memory map or a file reader serves with sequential reads, and still
visits every record once per epoch.

`BlockOrder(block_size=b, blocks_per_buffer=n)` is CorgiPile's buffer order
(Xu et al., arXiv 2206.05830, Algorithm 1): the records split into blocks of
`b` consecutive records, the full blocks are permuted, the permuted blocks are
grouped `n` to a buffer, and each buffer's positions are permuted. `n = 1` is
the plain block order. A batch therefore reads from the `n` blocks of the
buffer it falls in, and a read from storage keeps about `b * n` records' bytes
in the page cache.

```python
import jax
import numpy as np

from datarax.core.block_order import BlockOrder, order_positions
from datarax.core.index_words import from_words, to_words

order = BlockOrder(block_size=256, blocks_per_buffer=64)
positions = to_words(np.arange(1024))
records = from_words(order_positions(positions, 1_000_000, jax.random.key(0), order))
len(np.unique(records // 256))  # at most 64 blocks: the first buffer's
```

Like the global order, it is computed per position from the keyed cipher: no
order is stored, a batch costs O(batch), and the same key and position always
name the same record, however positions are batched. `order_positions` is the one function
applying an order: sequential without a key, the global order with a key and no
block order, the block order with both. `block_size` and `blocks_per_buffer`
are static (an order is hashable and keys a compiled program); positions and
key may be traced, under `jit`, `vmap` and `scan`.

When `b` does not divide the length, the short final block is never permuted
with the full blocks: it stays last, shuffled with the last buffer's other
records, so its at most `b - 1` records fall in an epoch's last buffer.

## See Also

- [index_words](index_words.md) - Positions and records as two uint32 words
- [data_source](data_source.md) - `record_indices_at`, which names a batch's records

---

::: datarax.core.block_order
