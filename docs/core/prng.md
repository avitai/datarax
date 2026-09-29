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

A stochastic operator draws one key per record with `per_record_keys`, which
folds the record's epoch, its draw within the epoch and its 64-bit index (two
uint32 words) into the operator's base key. A record's randomness depends only
on those, never on batch size, batch position, shuffle order or the number of
processes; indices past `2^32` keep distinct keys.

## Dtypes

Operators follow JAX's and Flax NNX's dtype rules; nothing is datarax-specific.
An operator without parameters returns each field in the field's dtype, as
`nnx.Dropout` does: the values it draws, in JAX's default dtype, are applied in
the data's dtype. An operator with learnable parameters (`LoudnessOperator`, a
composite with learnable weights, `CrepeF0Operator`) is a Flax layer: its
config's `param_dtype` (float32) is the parameters' dtype, and its computation
runs in `dtype`, or the promotion of its input and parameters when `dtype` is
`None`, so bfloat16 input computes in float32 and float64 input in float64.

Float64 data works with `jax_enable_x64` on; x64 is a program-wide setting, and
a draw's value, like any JAX draw, depends on it. TPUs run float32 and bfloat16.

## See Also

- [Operator](operator.md) - Where per-record keys are consumed
- [Config](config.md) - Seeding a component from configuration
- [NNX Best Practices](../user_guide/nnx_best_practices.md) - PRNG patterns

---

::: datarax.core.prng
