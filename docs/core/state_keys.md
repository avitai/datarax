# State Keys

The names of the per-record `state` entries datarax itself writes.

| key | per record | written by |
|---|---|---|
| `WEIGHT` | float weight in losses and metrics; 0 for a padding row | `batch_ops.mask`, `batch_ops.compact` |
| `MIX_PARTNER` | int32, the row a record was mixed with | `BatchMixOperator` |
| `MIX_LAMBDA` | float, one per batch (`batch_state`): the fraction kept | `BatchMixOperator` |
| `MASKED` | `state[MASKED][field]`, bool: a present value hidden from the model, kept in `data` as the target | the masking operator |
| `IMPUTED` | `state[IMPUTED][field]`, bool: a missing value filled, with the field's `Maybe.present` set | the filling operator |

`MASKED` and `IMPUTED` hold one entry per data field, so a loss reads
`batch.states[IMPUTED]["caption"]` beside `batch["caption"].present`
([missing, masked and padded values](maybe.md) has a running example).

## See Also

- [Batch operations](batch_ops.md) - `mask` and `compact` write the weight
- [Missing, masked and padded values](maybe.md) - `Maybe`, `MASKED` and `IMPUTED`

---

::: datarax.core.state_keys
