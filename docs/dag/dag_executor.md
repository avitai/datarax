# DAG Executor

The `DAGExecutor` is the core execution engine for Datarax pipelines. It executes complex data processing workflows as directed acyclic graphs (DAGs), providing automatic optimization, caching, and JIT compilation support.

## Key Features

| Feature | Description |
|---------|-------------|
| **DAG execution** | Topologically sorted execution of graph nodes |
| **Automatic caching** | Cache intermediate results for repeated operations |
| **JIT compilation** | Optional `jax.jit` compilation for performance |
| **Checkpointing** | Save and restore complete pipeline state |
| **Lazy RNG** | Random number generation only when needed |
| **Batch-first processing** | Enforces proper batching before transformations |

`★ Insight ─────────────────────────────────────`

- DAGExecutor is the main entry point for building Datarax pipelines
- Use the `>>` operator for fluent pipeline construction
- Batch-first enforcement prevents common "forgot to batch" errors
- The executor automatically detects stochastic operations and creates RNG only when needed

`─────────────────────────────────────────────────`

## Quick Start

### Linear Pipeline

```python
from datarax.dag import DAGExecutor, from_source
from datarax.dag.nodes import BatchNode, OperatorNode

# Method 1: Using from_source helper
pipeline = from_source(my_source, batch_size=32)

# Method 2: Using DAGExecutor directly
executor = DAGExecutor()
executor.add(my_source)
executor.batch(32)
executor.add(OperatorNode(normalize_op))
```

### Fluent API with >>

```python
from datarax.dag import from_source

# Build pipeline with >> operator
pipeline = (
    from_source(my_source, batch_size=64)
    >> OperatorNode(normalize_op)
    >> OperatorNode(augment_op)
)

# Iterate over batches
for batch in pipeline:
    loss = train_step(batch)
```

## Complex DAG Patterns

### Parallel Branches

Apply multiple transformations to the same input:

```python
executor = DAGExecutor()
executor.add(source)
executor.batch(32)
executor.parallel([
    OperatorNode(transform_a),
    OperatorNode(transform_b),
])
executor.merge("concat")  # or "sum", "mean", "stack"
```

### Conditional Branching

Route data through different paths:

```python
def is_large_image(batch):
    return batch["image"].shape[-2] > 256

executor.branch(
    condition=is_large_image,
    true_path=OperatorNode(resize_down),
    false_path=OperatorNode(resize_up),
)
```

## Pipeline Iteration

```python
# Create pipeline
pipeline = from_source(source, batch_size=32) >> transform

# Iterate (automatically creates iterator)
for batch in pipeline:
    process(batch)

# Manual control
pipeline.reset()
for epoch in range(10):
    for batch in pipeline:
        train_step(batch)
    pipeline.reset()
```

## JIT Compilation

Enable JIT compilation for faster execution:

```python
pipeline = DAGExecutor(
    jit_compile=True,  # JIT compile the execution graph
)
pipeline.add(source).batch(32).add(transform)

# First iteration compiles, subsequent ones are fast
for batch in pipeline:
    process(batch)
```

## Caching

Control caching behavior:

```python
pipeline = DAGExecutor(
    enable_caching=True,  # Cache intermediate results (default)
)

# Add explicit cache points
pipeline.cache(cache_size=100)

# Clear all caches
pipeline.clear_cache()
```

## Checkpointing

Save and restore pipeline state:

```python
# Get current state
state = pipeline.get_state()

# Restore from state
pipeline.set_state(state)

# With Orbax checkpoint handler
from datarax.checkpoint import OrbaxCheckpointHandler

handler = OrbaxCheckpointHandler()
handler.save("/checkpoints", pipeline, step=1000)

# Later...
handler.restore("/checkpoints", pipeline)
```

## Visualization

Inspect the pipeline structure:

```python
# Text visualization
print(pipeline.visualize())
# Output:
# DAGExecutor(
#   Sequential(
#     DataSourceNode(HFEagerSource)
#     BatchNode(batch_size=32)
#     OperatorNode(NormalizeOp)
#   )
# )

# String representation
print(pipeline)
# DAGExecutor(name=DAGExecutor, iterations=0, epochs=0, cached=True, jit=False)
```

## Batch-First Enforcement

By default, DAGExecutor enforces batch-first processing:

```python
# This will raise an error:
executor = DAGExecutor(enforce_batch=True)  # default
executor.add(source)
executor.add(transform)  # Error: Add BatchNode first!

# Correct:
executor.add(source)
executor.batch(32)  # Add batch node first
executor.add(transform)  # Now OK

# Disable enforcement if needed:
executor = DAGExecutor(enforce_batch=False)
```

## Helper Functions

### from_source

Create a pipeline from a data source:

```python
from datarax.dag import from_source

pipeline = from_source(
    source,
    batch_size=32,
    enforce_batch=True,
    jit_compile=False,
)
```

### pipeline

Create from a sequence of nodes:

```python
from datarax.dag import pipeline
from datarax.dag.nodes import DataLoader

# First node must be a DataLoader
pipe = pipeline(
    DataLoader(source, batch_size=32),
    OperatorNode(transform),
)
```

## See Also

- [DAG Construction Guide](../user_guide/dag_construction.md) - Building pipelines
- [Control Flow Nodes](control_flow.md) - Branch, Merge, Parallel
- [Pipeline Tutorial](../examples/core/pipeline-tutorial.md)
- [Checkpointing Guide](../user_guide/checkpointing_guide.md)

---

## API Reference

::: datarax.dag.dag_executor
