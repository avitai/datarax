# Control Flow Nodes

Datarax provides control flow nodes for building complex DAG structures with branching, parallel execution, and merging. These nodes enable sophisticated data processing pipelines that go beyond simple linear chains.

## Available Nodes

| Node | Description |
|------|-------------|
| `Sequential` | Chain nodes: output of one → input of next |
| `Parallel` | Execute multiple nodes on same input |
| `Branch` | Route data based on condition |
| `Merge` | Combine outputs from parallel branches |
| `Identity` | Pass data through unchanged (placeholder) |

`★ Insight ─────────────────────────────────────`

- Control flow nodes are **composable** - combine them freely
- Use `>>` operator for sequential and `|` for parallel composition
- For JAX JIT compatibility, Branch uses `jax.lax.cond` internally
- NNX modules passed to Branch get proper state tracking via `nnx.cond`

`─────────────────────────────────────────────────`

## Sequential Node

Chain nodes where each output feeds into the next:

```python
from datarax.dag.nodes import Sequential, Identity

# Using constructor
seq = Sequential([node_a, node_b, node_c])

# Using >> operator (preferred)
seq = node_a >> node_b >> node_c

# Execute
result = seq(data, key=rng_key)
```

## Parallel Node

Execute multiple nodes on the same input:

```python
from datarax.dag.nodes import Parallel

# Using constructor
parallel = Parallel([node_a, node_b])

# Using | operator
parallel = node_a | node_b

# Execute - returns list of outputs
results = parallel(data, key=rng_key)
# results = [output_a, output_b]
```

## Branch Node

Route data through different paths based on a condition:

```python
from datarax.dag.nodes import Branch

def is_image(data):
    """Return True for images, False otherwise."""
    return "image" in data and data["image"].ndim == 4

branch = Branch(
    condition=is_image,
    true_path=image_processor,
    false_path=text_processor,
)

# Data is routed based on condition
result = branch(data, key=rng_key)
```

### JAX Compatibility

Branch uses `jax.lax.cond` for JIT-compatible conditional execution:

```python
# Works inside jax.jit
@jax.jit
def process(data):
    return branch(data)

# For NNX modules, uses nnx.cond for state tracking
class MyBranch(nnx.Module):
    def __init__(self):
        self.branch = Branch(
            condition=lambda x: x["type"] == 0,
            true_path=NNXModule(),
            false_path=AnotherNNXModule(),
        )
```

## Merge Node

Combine outputs from parallel branches:

```python
from datarax.dag.nodes import Merge

# Different merge strategies
merge_concat = Merge(strategy="concat", axis=-1)
merge_stack = Merge(strategy="stack", axis=0)
merge_sum = Merge(strategy="sum")
merge_mean = Merge(strategy="mean")

# Typically used after Parallel
pipeline = (
    Parallel([node_a, node_b])
    >> Merge(strategy="concat")
)
```

### Merge Strategies

| Strategy | Description |
|----------|-------------|
| `"concat"` | Concatenate along specified axis |
| `"stack"` | Stack into new dimension |
| `"sum"` | Element-wise sum |
| `"mean"` | Element-wise average |

```python
# Example outputs for [array_a, array_b] where each is shape (32, 10)
# concat (axis=-1): shape (32, 20)
# stack (axis=0): shape (2, 32, 10)
# sum: shape (32, 10)
# mean: shape (32, 10)
```

## Identity Node

Pass data through unchanged (useful as placeholder):

```python
from datarax.dag.nodes import Identity

identity = Identity()
result = identity(data)  # result == data
```

## Combining Control Flow

Build complex patterns by combining nodes:

```python
from datarax.dag.nodes import Sequential, Parallel, Branch, Merge

# Multi-branch ensemble
ensemble = (
    Parallel([
        model_a,
        model_b,
        model_c,
    ])
    >> Merge(strategy="mean")
)

# Conditional with parallel inside
complex_pipeline = Sequential([
    preprocessing,
    Branch(
        condition=is_complex,
        true_path=Parallel([heavy_op_a, heavy_op_b]) >> Merge("concat"),
        false_path=simple_op,
    ),
    postprocessing,
])
```

## Convenience Functions

```python
from datarax.dag.nodes.control_flow import parallel, branch

# Create parallel node
par = parallel(node_a, node_b, node_c)

# Create branch node
br = branch(
    condition=my_condition,
    true_path=true_node,
    false_path=false_node,
)
```

## Integration with DAGExecutor

Control flow nodes integrate seamlessly with DAGExecutor:

```python
from datarax.dag import from_source

pipeline = from_source(source, batch_size=32)
pipeline.parallel([transform_a, transform_b])
pipeline.merge("concat")
pipeline.branch(
    condition=my_condition,
    true_path=path_a,
    false_path=path_b,
)

for batch in pipeline:
    process(batch)
```

## See Also

- [DAG Executor](dag_executor.md) - Pipeline execution
- [Composite Operator](../operators/composite_operator.md) - Operator-level composition
- [DAG Construction Guide](../user_guide/dag_construction.md)
- [Pipeline Tutorial](../examples/core/pipeline-tutorial.md)

---

## API Reference

::: datarax.dag.nodes.control_flow
