# DAG Construction Guide

Datarax uses a Directed Acyclic Graph (DAG) model to represent data pipelines. This guide explains how to construct and execute DAGs.

## Introduction

A DAG in Datarax consists of **Nodes** representing operations (data sources, transformations, batching) and **Edges** representing data flow.

## Building a DAG

You can build a DAG using the `DAGExecutor` class.

### 1. Initialize Executor

```python
from datarax.dag import DAGExecutor

executor = DAGExecutor()
```

### 2. Add Nodes

Add nodes to the executor. The order matters for sequential dependencies.

```python
from datarax.sources import MemorySource
from datarax.dag.nodes import DataSourceNode, BatchNode

# Add Source
source = MemorySource(data)
executor.add(DataSourceNode(source))

# Add Batching
executor.add(BatchNode(batch_size=32))
```

### 3. Execute

Iterate over the executor to run the pipeline.

```python
for batch in executor:
    process(batch)
```

## Node Types

- **DataSourceNode**: Entry point for data.
- **OperatorNode**: Applies transformations (`FunctionTransformer`, `OperatorModule`).
- **BatchNode**: Batches elements.
- **ShuffleNode**: Shuffles data.

## API Reference

For full details on available nodes and execution options, see the [DAG API Reference](../dag/index.md).
