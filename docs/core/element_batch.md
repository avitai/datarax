# Element and Batch Types

Core types for individual elements and batched data.

## What datarax records about a record

Beside the record's `data` and the user's `state`, datarax keeps four typed attributes. Each has
the record axis first on a `Batch`, so every row operation moves it with its rows.

| attribute | on an `Element` | on a `Batch` | written by |
|---|---|---|---|
| weight | `weight`, float32 `()`, 1 by default | `weights`, float32 `(B,)`: 1 for a record, 0 for a padding row | `batch_ops.mask`, `batch_ops.compact`, an operator, `batch.replace(weights=...)` |
| masked | `masked`, bool marks in the data's nesting, `{}` by default | `masked`, the same with `B` first | an operator that hides a value and keeps it as the target |
| imputed | `imputed`, bool marks in the data's nesting, `{}` by default | `imputed`, the same with `B` first | an operator that fills a missing value |
| mix | none | `mix`: `Mix(partners, lams)`, int32 and float `(B,)`, or `None` | a mixing operator (`BatchMixOperator`) |

A mark sits at a field's shape or a prefix of it ([missing, masked and padded values](maybe.md)).
`state` and `batch_state` hold only the user's names.

A per-record wrapper that applies its child to some records (`ProbabilisticOperator`,
`SelectorOperator`, the conditional and branching composites) carries the weight and marks its
child writes through every branch: where the child does not apply, the record keeps its own
weight, and a mark it did not have is False. A user `state` entry has no such neutral value, so
a child that adds one only where it applies is refused, naming the wrapper, the child and the
entry; give the entry a value before the wrapper.

A batch is mixed once: a second mixing operator is refused with `MixedTwiceError`, since a loss
reads one partner and one coefficient per row. A mix's partners are row numbers of the batch the
mixing operator returned; `take`, `slice_rows`, `split`, `concatenate` and `compact` move them
with their rows without renumbering them.

## See Also

- [Core Overview](index.md) - All core protocols
- [Batching](../batching/index.md) - Batch creation
- [Element Operator](../operators/element_operator.md) - Element transforms
- [Types & Protocols](../root/index.md) - Type definitions
- [Missing values](maybe.md) - `Maybe`, a field a record may lack, and the `masked` and
  `imputed` marks

---

::: datarax.core.element_batch
