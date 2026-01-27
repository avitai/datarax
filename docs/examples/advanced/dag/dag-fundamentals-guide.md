# DAG Pipeline Fundamentals Guide

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~45 min |
| **Prerequisites** | Pipeline Tutorial, Operators Tutorial |
| **Format** | Python + Jupyter |

## Overview

Master the Directed Acyclic Graph (DAG) pipeline architecture in Datarax.
This guide covers explicit node construction, control flow, caching strategies,
and building production-ready data pipelines with maximum performance.

## Learning Goals

By the end of this guide, you will be able to:

1. Construct explicit DAG pipelines with node types
2. Use operator-based composition (`>>` and `|` operators)
3. Implement control flow patterns (Sequential, Parallel, Branch, Merge)
4. Add caching for expensive transformations
5. Build rebatch strategies for dynamic batch sizing
6. Understand DAG execution and optimization

## Coming from PyTorch?

| PyTorch | Datarax DAG |
|---------|-------------|
| `DataLoader(dataset)` | `from_source(source)` or `DataSourceNode(source)` |
| `transforms.Compose` | `Sequential([...])` or `node1 >> node2` |
| Multiple dataloaders | `Parallel([...])` or `node1 \| node2` |
| `collate_fn` | `BatchNode` with custom logic |
| N/A (manual caching) | `Cache(node, cache_size=100)` |

## Coming from TensorFlow?

| TensorFlow tf.data | Datarax DAG |
|--------------------|-------------|
| `tf.data.Dataset` | `DataSourceNode` |
| `dataset.map()` | `OperatorNode(operator)` |
| `dataset.batch()` | `BatchNode(batch_size)` |
| `dataset.cache()` | `Cache(node)` |
| `dataset.prefetch()` | `PrefetchNode(buffer_size)` |
| `dataset.shuffle()` | `ShuffleNode(source)` |

## Coming from Google Grain?

| Grain | Datarax DAG |
|-------|-------------|
| `grain.DataLoader` | `DAGExecutor` or `from_source()` |
| `grain.MapTransform` | `OperatorNode(ElementOperator(...))` |
| `grain.Batch` | `BatchNode(batch_size)` |
| N/A | `Cache`, `Sequential`, `Parallel`, `Branch` |

## Files

- **Python Script**: [`examples/advanced/dag/01_dag_fundamentals_guide.py`](https://github.com/avitai/datarax/blob/main/examples/advanced/dag/01_dag_fundamentals_guide.py)
- **Jupyter Notebook**: [`examples/advanced/dag/01_dag_fundamentals_guide.ipynb`](https://github.com/avitai/datarax/blob/main/examples/advanced/dag/01_dag_fundamentals_guide.ipynb)

## Quick Start

### Run the Python Script

```bash
python examples/advanced/dag/01_dag_fundamentals_guide.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/advanced/dag/01_dag_fundamentals_guide.ipynb
```

## Architecture Overview

```mermaid
graph TB
    subgraph Sources["Data Sources"]
        DS[DataSourceNode]
    end

    subgraph Processing["Processing Nodes"]
        BN[BatchNode]
        ON[OperatorNode]
        SN[ShuffleNode]
        PN[PrefetchNode]
    end

    subgraph ControlFlow["Control Flow"]
        SEQ[Sequential<br/>node1 >> node2]
        PAR[Parallel<br/>node1 \| node2]
        BR[Branch]
        MG[Merge]
    end

    subgraph Optimization["Optimization"]
        CA[Cache]
        RB[RebatchNode]
    end

    DS --> BN
    BN --> ON
    ON --> SEQ
    SEQ --> PAR
    PAR --> CA

    style Sources fill:#e1f5fe
    style Processing fill:#f3e5f5
    style ControlFlow fill:#e8f5e9
    style Optimization fill:#fff3e0
```

## Key Concepts

### Part 1: DAG Node Hierarchy

Datarax DAG nodes form a hierarchy with specific responsibilities:

| Node | Purpose | Example |
|------|---------|---------|
| `DataSourceNode` | Pipeline entry point | `DataSourceNode(source)` |
| `BatchNode` | Create batches | `BatchNode(batch_size=32)` |
| `OperatorNode` | Apply transforms | `OperatorNode(operator)` |
| `ShuffleNode` | Randomize order | Via `MemorySourceConfig(shuffle=True)` |
| `PrefetchNode` | Background loading | `PrefetchNode(buffer_size=2)` |
| `Cache` | Store results | `Cache(node, cache_size=100)` |
| `Sequential` | Chain operations | `node1 >> node2` |
| `Parallel` | Multiple branches | `node1 \| node2` |

### Part 2: Simple Pipeline with from_source()

```python
from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = from_source(source, batch_size=32)

for batch in pipeline:
    print(f"Batch: {batch['image'].shape}")
```

**Terminal Output:**
```
Batch: (32, 32, 32, 3)
```

### Part 3: Explicit DAG Construction

```python
from datarax.dag.nodes import DataSourceNode, BatchNode, OperatorNode

# Create nodes explicitly
data_node = DataSourceNode(source, name="ImageSource")
batch_node = BatchNode(batch_size=32, name="Batcher")

# Compose using >> operator
pipeline = data_node >> batch_node
```

**Terminal Output:**
```
Explicit DAG construction:
  Pipeline: Sequential(ImageSource >> Batcher)
  Type: Sequential
```

### Part 4: Composition Operators

DAG nodes support operator-based composition:

| Operator | Creates | Usage |
|----------|---------|-------|
| `>>` | Sequential | `a >> b >> c` (chain) |
| `\|` | Parallel | `a \| b \| c` (parallel branches) |

```python
# Sequential: output of one feeds into next
seq = op1 >> op2 >> op3

# Parallel: same input to all, list of outputs
par = branch_a | branch_b
```

### Part 5: Caching Expensive Operations

```python
from datarax.dag.nodes import Cache

# Wrap expensive node with caching
cached_op = Cache(
    OperatorNode(expensive_op, name="ExpensiveOp"),
    cache_size=100
)
```

**Benefits:**
- LRU eviction when cache is full
- Only caches deterministic operations (no RNG key)
- Significant speedup for repeated inputs

### Part 6: Building Production Pipelines

```python
def build_production_pipeline(data, batch_size=32, shuffle=True):
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle, seed=42),
        data=data,
        rngs=nnx.Rngs(0),
    )

    pipeline = (
        from_source(source, batch_size=batch_size)
        .add(OperatorNode(normalize_op, name="Normalize"))
        .add(OperatorNode(augment_op, name="Augment"))
    )

    return pipeline
```

## Results

Running the guide produces:

```
============================================================
DAG Pipeline Fundamentals Guide
============================================================

1. Simple Pipeline (from_source):
   Batch shape: (16, 32, 32, 3)

2. Sequential Composition (>> operator):
   Pipeline: Sequential(A >> B)

3. Parallel Composition (| operator):
   Pipeline: Parallel([Left, Right])

4. Production Pipeline:
   Processed 4 batches

============================================================
Guide completed successfully!
============================================================
```

## Execution Model

| Aspect | Description |
|--------|-------------|
| **Pull-based** | Data pulled through graph on iteration |
| **Lazy evaluation** | Nodes execute only when outputs needed |
| **State tracking** | NNX modules maintain state across iterations |

### Optimization Points

| Optimization | Pattern |
|--------------|---------|
| Caching | Use `Cache` for deterministic expensive ops |
| Batching | Batch early in pipeline for vmap efficiency |
| Operator fusion | Combine multiple light ops into single function |
| Memory reuse | Use `drop_remainder=True` for fixed batch shapes |

## Next Steps

- [Composition Strategies](../../core/composition-strategies-tutorial.md) - Operator composition
- [Sharding Guide](../distributed/sharding-guide.md) - Distributed pipelines
- [Performance Guide](../performance/optimization-guide.md) - Optimization tips

## API Reference

- [`DAGExecutor`](../../../dag/dag_executor.md) - Pipeline executor
- [`Node`](../../../dag/base.md) - Base node class
- [`DataSourceNode`](../../../dag/data_source.md) - Source node
