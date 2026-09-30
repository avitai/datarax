# Missing, Masked and Padded Values

Three different things, with three representations:

| | meaning | the value | marked by |
|---|---|---|---|
| missing | the record has no value for the field | does not exist; its slot holds zeros | `present` inside the field: the field is a `Maybe(value, present)` in `data` |
| masked | the value exists and an operator hides it from the model | kept in `data`: it is the target | `state[state_keys.MASKED][field]`, written by the operator |
| padding | a row that is not a record | zeros | `state[state_keys.WEIGHT] = 0` and `PADDING_INDEX` ([batch_ops](batch_ops.md)) |

Every field exists in every record at its static shape, so batches with different fields missing
share one structure and one compiled program. `present` is a bool per record (leading axis `B` in a
batch, `()` for one record), placed with the batch like every other leaf: one byte per record per
field, and reading it inside a step transfers nothing.

## `Maybe` has no arithmetic

A fill value cannot be used by accident. `field * 2`, `field + 1.0`, `field == 0`,
`jnp.mean(field)` and `np.asarray(field)` raise `TypeError`; the value is read with
`field.value_or(fill)`, or with `field.present` in hand. Missing slots hold zeros, not NaN: a NaN
hidden behind `jnp.where` still makes the gradient NaN while the loss stays finite.

```python
import jax
import jax.numpy as jnp
import numpy as np
from datarax.core import Maybe, batch_ops

present = np.array([True, False, True, False])
depth = Maybe(np.where(present[:, None], np.ones((4, 3), np.float32), 0.0), present)
batch = batch_ops.from_arrays({"image": np.zeros((4, 8, 8, 3), np.float32), "depth": depth})

try:
    batch["depth"] * 2
except TypeError as error:
    print("refused:", error)

filled = jax.jit(lambda b: b["depth"].value_or(-1.0))(batch)
print(filled[:, 0])  # [ 1. -1.  1. -1.]
record = batch_ops.element(batch, 1)
print(bool(record.data["depth"].present))  # False
```

A `Maybe` moves through every batch operation (`element`, `take`, `split`, `concatenate`,
`stack`, `mask`, `compact`) and through placement (`batch_ops.shardings`) with its rows, at any
nesting depth in `data`. A spec describes it as a `Maybe` of two `jax.ShapeDtypeStruct`s.

## Filling and masking: what an operator records in `state`

An operator that fills a missing value returns the field with `present` set and records
`state[IMPUTED][field]`, so a loss can tell observed values from imputed ones. An operator that
hides a present value keeps it in `data` as the target and records `state[MASKED][field]`.

```python
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from datarax.core import Element, Maybe, OperatorModule, batch_ops
from datarax.core.config import OperatorConfig
from datarax.core.state_keys import IMPUTED, MASKED


class FillFromImage(OperatorModule):
    """Fills a missing caption embedding from the image; marks what it filled."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.project = nnx.Linear(12, 4, rngs=rngs)

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        caption = element.data["caption"]
        fill = ~caption.present
        value = jnp.where(fill, self.project(element.data["image"].reshape(-1)), caption.value)
        return element.replace(
            data={**element.data, "caption": Maybe(value, caption.present | fill)},
            state={**element.state, IMPUTED: {"caption": fill}},
        )


class MaskImage(OperatorModule):
    """Hides half the image pixels from the model; the image stays the target."""

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        assert key is not None
        hidden = jax.random.bernoulli(key, 0.5, element.data["image"].shape)
        return element.update_state({MASKED: {"image": hidden}})


present = np.array([True, False, True, False])
caption = Maybe(np.where(present[:, None], np.ones((4, 4), np.float32), 0.0), present)
batch = batch_ops.from_arrays({"image": np.ones((4, 2, 2, 3), np.float32), "caption": caption})

filled = FillFromImage(rngs=nnx.Rngs(0))(batch)
print(filled["caption"].present)  # [ True  True  True  True]
print(filled.states[IMPUTED]["caption"])  # [False  True False  True]

masked = MaskImage(
    OperatorConfig(stochastic=True, stream_name="mask"), rngs=nnx.Rngs(mask=0)
)(filled)
assert np.array_equal(masked["image"], filled["image"])  # the target is kept
print(masked.states[MASKED]["image"].shape)  # (4, 2, 2, 3)
```

A built-in operator that treats a field as an array (`MapOperator` over the field, the parallel
merges and ensemble reductions of `CompositeOperatorModule`, `BatchMixOperator`, the image and
audio operators, `CrossModalOperator` inputs) refuses a `Maybe` with a `TypeError` naming the
field: fill it first, or write an operator that reads `present`. Operators that move records
whole (per-record maps, sequential and branching composites, probabilistic wrappers) carry it
unchanged.

## See Also

- [Element and Batch](element_batch.md) - The record and batch types a `Maybe` lives in
- [State keys](state_keys.md) - `MASKED`, `IMPUTED`, `WEIGHT` and the mixing entries
- [Batch operations](batch_ops.md) - Padding rows: `mask`, `compact`, `record_count`

---

::: datarax.core.maybe
