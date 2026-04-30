# DAG Construction Guide

Datarax uses a Directed Acyclic Graph (DAG) model to represent data pipelines. This guide explains how to construct and execute DAGs.

## Introduction

A DAG in Datarax consists of **Nodes** representing operations (data sources, transformations, batching) and **Edges** representing data flow.

## Building a DAG

You build a pipeline by instantiating the `Pipeline` class with a
source, a list of stages, a batch size, and an `nnx.Rngs` instance.
For DAG-shaped pipelines (branching or merging), use
`Pipeline.from_dag(...)`.

### 1. Linear Pipeline

```python
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = Pipeline(
    source=source,
    stages=[op1, op2],
    batch_size=32,
    rngs=nnx.Rngs(0),
)
```

### 2. Execute

The pipeline is iterable and supports `pipeline.scan(...)` for
whole-epoch JIT compilation.

```python
for batch in pipeline:
    process(batch)
```

### 3. Branching DAG

Use `Pipeline.from_dag` when stages need to branch or merge. Each
node declares its predecessors via the `edges` mapping and the sink
selects the output node.

```python
pipeline = Pipeline.from_dag(
    source=source,
    nodes={"augment": aug, "normalize": norm, "merge": merge},
    edges={"augment": [], "normalize": [], "merge": ["augment", "normalize"]},
    sink="merge",
    batch_size=32,
    rngs=nnx.Rngs(0),
)
```

For runnable recipes (parallel, merge variants — stack/average/concat —
and conditional `Branch`), see the
[Branching DAG Cookbook](../examples/advanced/dag/branching-dag-cookbook.md).

## Stage Types

Any `nnx.Module` whose `__call__(batch) -> batch` transforms the
batch can be used as a stage. Datarax also provides:

- **OperatorModule subclasses** (e.g. `BrightnessOperator`,
  `NoiseOperator`): receive an `Element`, return an updated
  `Element`. Pipeline detects these and uses an optimized fast path.
- **Plain `nnx.Module`**: receives the dict batch directly. Use this
  for user-defined transforms.

## API Reference

For full details on available nodes and execution options, see the [DAG API Reference](../dag/index.md).
