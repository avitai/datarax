# PRNG

Where a datarax component's randomness comes from.

A component built from a seeded configuration receives an `nnx.Rngs` over the
`DEFAULT_RNG_STREAMS` (`augment`, `dropout`, `params`, `shuffling`, `default`),
built with `substrax.rng.rngs_from_seed`: each stream's key is derived from the
seed and the stream's name, so a stream's key does not depend on the other
streams present.

```python
from substrax.rng import rngs_from_seed

from datarax.core.prng import DEFAULT_RNG_STREAMS

rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)
key = rngs.augment()  # a raw key for an external library
```

A stochastic operator draws one key per record with `per_record_keys`, so a
record's randomness depends only on the operator's base key, the epoch and the
record's stable index, never on batch size, batch position or shuffle order.

## See Also

- [Operator](operator.md) - Where per-record keys are consumed
- [Config](config.md) - Seeding a component from configuration
- [NNX Best Practices](../user_guide/nnx_best_practices.md) - PRNG patterns

---

::: datarax.core.prng
