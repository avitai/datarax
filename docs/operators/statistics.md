# Operator Statistics

An operator can apply statistics to every record it transforms: a mean and a standard deviation
to normalize by, a peak to scale against, a threshold fitted to the data. Statistics are either
**stored** on the operator, or **computed from each batch** as it arrives.

## Stored statistics

`set_statistics` stores the values an operator applies; `get_statistics` reads them back and
`reset_statistics` clears them.

```python
operator = NormalizeOperator(NormalizeConfig())
operator.set_statistics({"mean": 0.5, "std": 0.2})
```

The store is a plain `nnx.Variable`, so the statistics are module state rather than
configuration. Two consequences follow. They round-trip through a checkpoint with the rest of the
operator's state, and two operators with equal configurations share one compiled trace whatever
their statistics hold — a fitted number never becomes part of what a transform compares.

## Statistics fitted to each batch

An operator that derives its statistics from the data overrides `compute_statistics`. The batch
path calls it **once per batch, before the batch is vectorized**, and gives every record in that
batch the same result:

```python
class BatchNormalize(OperatorModule):
    def compute_statistics(self, batch_data):
        image = batch_data["image"]
        return {"mean": jnp.mean(image), "std": jnp.std(image) + 1e-6}

    def apply(self, data, state, metadata, key=None, stats=None):
        normalized = (data["image"] - stats["mean"]) / stats["std"]
        return {**data, "image": normalized}, state, metadata
```

`batch_data` carries the batch on axis 0, so a reduction over it describes the whole batch. The
statistics are part of the traced computation, so gradients flow through them.

The default `compute_statistics` returns whatever `set_statistics` stored, which is why an
operator with fixed statistics needs no override and produces exactly what it did before.

A caller may also pass statistics explicitly, and an explicit argument wins:

```python
operator.apply_batch(batch, stats={"mean": 0.0, "std": 1.0})
```

## Wrappers and their children

A composite, a selector and a probabilistic wrapper apply no statistics of their own. Each
computes one entry per child, once per batch, and gives every child the statistics that child
computed on the wrapper's input. A wrapper whose children all compute none passes none.

```python
composite = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[BatchNormalize(config), Brighten(config)],
    ),
)
```

Here `BatchNormalize` receives the statistics it computed on the composition's input, and
`Brighten` receives its own.

!!! note "Children see the composition's input"

    A wrapper applies its children inside one vectorized call, so a child cannot compute
    statistics of its own while it runs — by then the batch is gone. A sequential composition's
    later children therefore see the statistics of the **composition's** input, not of the
    previous child's output. Where a stage needs statistics of exactly what reaches it, give it
    its own `Pipeline` stage.

A weighted-parallel composite strips `weight_key` from the data before any child runs, and it is
stripped before the children's statistics are computed too, so a child's statistics describe the
fields it is actually given.

## See Also

- [Operators Overview](index.md) - All operator types
- [Composite Operator](composite_operator.md) - Operator composition
- [Element Operator](element_operator.md) - Element-level transformations
- [Operators Tutorial](../examples/core/operators-tutorial.md)
