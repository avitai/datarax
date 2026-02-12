# DAG

Directed Acyclic Graph (DAG) based pipeline execution system. The DAG module provides the execution engine that orchestrates data flow through your pipeline, handling batching, caching, and control flow.

## Components

| Component | Purpose | Key Features |
|-----------|---------|--------------|
| **DAGExecutor** | Pipeline execution | Caching, JIT, checkpointing |
| **Nodes** | Graph building blocks | Source, Batch, Operator, Control |
| **Control Flow** | Branching & merging | Sequential, Parallel, Branch, Merge |

`★ Insight ─────────────────────────────────────`

- DAGExecutor enforces **batch-first** processing by default
- Use `>>` operator for fluent pipeline construction
- Nodes are composable - combine freely with control flow
- JIT compilation available for performance-critical pipelines

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.dag import from_source, DAGExecutor
from datarax.dag.nodes import OperatorNode

# Build pipeline with fluent API
pipeline = (
    from_source(my_source, batch_size=32)
    >> OperatorNode(normalize)
    >> OperatorNode(augment)
)

# Iterate over batches
for batch in pipeline:
    loss = train_step(batch)
```

## Core Modules

- [dag_executor](dag_executor.md) - Main execution engine with caching and JIT
- [dag_config](dag_config.md) - DAG configuration and setup options

## Node Types

Nodes are the building blocks of DAG pipelines:

### Data Nodes

- [data_source](data_source.md) - Wrap data sources as nodes
- [loaders](loaders.md) - DataLoader with batching and sampling

### Processing Nodes

- [field_operators](field_operators.md) - Field-level transformations
- [rebatch](rebatch.md) - Reshape batches (split, combine)
- [caching](caching.md) - Cache intermediate results

### Control Flow

- [control_flow](control_flow.md) - Sequential, Parallel, Branch, Merge
- [base](base.md) - Base node classes and interfaces

## Pipeline Patterns

```python
# Linear pipeline
pipeline = source >> batch >> transform

# Parallel branches
executor.parallel([transform_a, transform_b])
executor.merge("concat")

# Conditional branching
executor.branch(
    condition=is_training,
    true_path=augment,
    false_path=identity,
)
```

## Real-World Examples

- [Learned ISP Pipeline](../examples/advanced/differentiable/learned-isp-guide.md) - 5-stage differentiable ISP using `>>` operator with gradient flow through every stage
- [DDSP Audio Synthesis](../examples/advanced/differentiable/ddsp-audio-synthesis.md) - Parallel + MergeBatchNode DAG for harmonic and noise synthesis
- [DADA Learned Augmentation](../examples/advanced/differentiable/dada-learned-augmentation.md) - Datarax pipelines composed with Gumbel-Softmax for differentiable augmentation search

## See Also

- [DAG Executor Guide](dag_executor.md) - Complete executor documentation
- [Control Flow Guide](control_flow.md) - Branching and merging
- [DAG Construction](../user_guide/dag_construction.md) - User guide
- [Pipeline Tutorial](../examples/core/pipeline-tutorial.md)
