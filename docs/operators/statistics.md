# Operator Statistics

An operator can apply statistics to every record it transforms: a mean and a standard deviation
to normalize by, a peak to scale against, a threshold fitted to the data. Statistics are either
**stored** on the operator, or **computed from each batch** as it arrives.

## Stored statistics

Stored statistics are given when the operator is built; `get_statistics` reads them back and
`set_statistics` replaces their values.

```python
operator = NormalizeOperator(NormalizeConfig(), statistics={"mean": 0.5, "std": 0.2})
operator.set_statistics({"mean": 0.4, "std": 0.3})  # same entries, shapes and dtypes
```

The store is a plain `nnx.Variable` of arrays, created at its real shape in the constructor, so
the statistics are module state rather than configuration, and the operator's state layout never
changes after it is built. Three consequences follow. They round-trip through a checkpoint with
the rest of the operator's state; two operators with equal configurations share one compiled trace
whatever their statistics hold, and replacing their values compiles nothing new; and
`set_statistics` refuses values of another structure, shape or dtype, as it refuses an operator
built without statistics, leaving the store unchanged.

## Statistics fitted to each batch

An operator that derives its statistics from the data overrides `compute_statistics`. `__call__`
calls it **once per batch, before the records are mapped**, and gives every record in that batch
the same result:

```python
class BatchNormalize(OperatorModule):
    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        image = batch.data["image"]
        return {"mean": jnp.mean(image), "std": jnp.std(image) + 1e-6}

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        normalized = (data["image"] - stats["mean"]) / stats["std"]
        return element.replace(data={**data, "image": normalized})
```

`batch.data` carries the batch on axis 0, so a reduction over it describes the whole batch. The
statistics are part of the traced computation, so gradients flow through them.

The default `compute_statistics` returns the stored statistics, so an operator with fixed
statistics needs no override.

A caller may also pass statistics explicitly by calling `apply_batch` with them; for a
deterministic operator the keys are `None`:

```python
operator.apply_batch(batch, None, {"mean": 0.0, "std": 1.0})
```

## Wrappers and their children

A wrapper applies no statistics of its own; how its children get theirs follows how it applies
them.

A **sequential** composite calls each child on the whole batch in turn, so each child computes its
statistics on the input it actually receives: a later child's describe the earlier child's output.

```python
composite = CompositeOperatorModule(
    CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL),
    operators=[BatchNormalize(config), Brighten(config)],
)
```

A wrapper that decides or merges **per record** (a selector, a probabilistic wrapper, a parallel,
weighted, ensemble, conditional or branching composite) applies its children inside one
vectorized call, so a child cannot compute statistics while it runs. The wrapper computes one
entry per child, once per batch, on its own input, which is each child's input, and gives every
child its own. A wrapper whose children all compute none passes none. A chain applied per record
(a sequential composite inside such a wrapper, or a conditional sequential composite) refuses a
later child that fits statistics per batch, since those would describe the chain's input rather
than the child's.

A weighted-parallel composite strips `weight_key` from the data before any child runs, and it is
stripped before the children's statistics are computed too, so a child's statistics describe the
fields it is actually given.

## See Also

- [Operators Overview](index.md) - All operator types
- [Composite Operator](composite_operator.md) - Operator composition
- [Element Operator](element_operator.md) - Element-level transformations
- [Operators Tutorial](../examples/core/operators-tutorial.md)
