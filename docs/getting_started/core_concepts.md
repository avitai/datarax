# Core Concepts

This guide explains the **Three-Tier Architecture** of Datarax, designed to provide a clear separation of concerns between state management, parametric operations, and structural data organization.

## Architecture Guidelines

The Datarax architecture is built on three hierarchical tiers:

1.  **Tier 1: DataraxModule** (Base Foundation)
2.  **Tier 2A: OperatorModule** (Parametric & Learnable)
3.  **Tier 2B: StructuralModule** (Non-Parametric & Structural)

This conceptual separation ensures that components are highly composable, type-safe, and easy to reason about.

```mermaid
graph TD
    A[DataraxModule] --> B[OperatorModule]
    A[DataraxModule] --> C[StructuralModule]
    B --> D[Operators]
    C --> E[Data Sources]
    C --> F[Batchers]
    C --> G[Samplers]
    C --> H[Sharders]
```

---

## Tier 1: DataraxModule (The Foundation)

**`DataraxModule`** is the base class for ALL components in the library. It inherits from `flax.nnx.Module`, providing the fundamental capabilities required for robust state management in JAX.

### Key Capabilities
-   **State Management**: Automatically tracks state (parameters, RNG keys, metrics) using Flax NNX.
-   **Checkpointing**: Integration with Orbax for saving and restoring full pipeline state.
-   **Iteration Tracking**: Keeps track of the number of iterations/calls.

All Datarax components, regardless of their specific role, share this common DNA. Capabilities that
only one kind of module needs live on that kind: operators hold the statistics they apply
(`set_statistics`), and samplers memoize each sampled result by request size.

---

## Tier 2A: OperatorModule (Parametric Transformations)

**`OperatorModule`** represents the "compute" layer of your data pipeline. These modules perform **differentiable, parametric transformations** on data.

### Characteristics
-   **Parametric**: Can have learnable parameters (e.g., weights in a normalization layer).
-   **Differentiable**: Fully compatible with JAX's automatic differentiation (`jax.grad`).
-   **Input/Output**: Expects and returns a `Batch` of data.
-   **Modes**:

    -   **Deterministic**: output = f(input) (e.g., Resize, Crop, Normalize)
    -   **Stochastic**: output = f(input, rng) (e.g., RandomFlip, ColorJitter, Mixup)

### Operator Definition

An **`OperatorModule`** is a unified abstraction for all data transformations. Whether a transformation is deterministic (like resizing an image) or stochastic (like adding random noise), it is implemented as an Operator.

This abstraction simplifies the mental model: everything that *changes* data values is an Operator.

### Usage Example

```python
from datarax.core.operator import OperatorModule, OperatorConfig
import flax.nnx as nnx
import jax

# Defining a custom Deterministic Operator
class NormalizeOperator(OperatorModule):
    def __init__(self, config, mean, std):
        super().__init__(config)
        self.mean = mean
        self.std = std

    def apply(self, data, state, metadata, key=None, stats=None):
        # Simplified implementation
        return data, state, metadata

# Defining a custom Stochastic Operator
class RandomFlipOperator(OperatorModule):
    def apply(self, data, state, metadata, key=None, stats=None):
        # key is this record's PRNG key; draw whatever randomness you apply from it
        return data, state, metadata

# Instantiation
norm_op = NormalizeOperator(
    OperatorConfig(stochastic=False),
    mean=0.5, std=0.5
)

augment_op = RandomFlipOperator(
    OperatorConfig(stochastic=True, stream_name="augment"),
    rngs=nnx.Rngs(augment=42)
)
```


### Per-Record Determinism

Stochastic operators key their randomness on the **epoch** and each record's
**stable index**, not on batch position or how many batches have been consumed.
Each operator draws one base key at construction and derives a per-record key as
`fold_in(fold_in(base_key, epoch), record_index)`. Within an epoch a record is
augmented **identically** regardless of batch size, shuffle order, how records are
split across workers, or resume point, and every epoch draws fresh augmentation —
while gradients still flow through the transformation. The `Pipeline` asks its
source for the indices of the records it serves (`record_indices_at`); a streaming
source's records are named by their position in the stream.

---

## Tier 2B: StructuralModule (Data Organization)

**`StructuralModule`** represents the "organization" layer. These modules change the **structure** or **arrangement** of data but do not modify the data values themselves in a learnable way.

### Characteristics
-   **Non-Parametric**: Configuration is static and known at compile-time (e.g., batch size).
-   **Metadata-Aware**: Handles data organization, batching, and distribution.
-   **Immutable Config**: Uses frozen configuration classes to ensure structural stability.
-   **Input/Output**: Flexible (can be individual elements, batches, or indices).

### Key Implementations
1.  **Data Sources**: specialized `StructuralModule` that yields initial data Element.
2.  **Batchers**: Group individual elements into a `Batch`.
3.  **Samplers**: Generate sequences of indices for data retrieval.
4.  **Sharders**: Split batches across multiple devices (GPUs/TPUs).

### Usage Example

```python
from datarax.core.structural import StructuralConfig, StructuralModule
from datarax.core.batcher import BatcherModule
import flax.nnx as nnx

# Defining a custom Batcher
class SimpleBatcher(BatcherModule):
    def process(self, elements, *args, batch_size, drop_remainder=False, **kwargs):
        # Simplified batching logic for demonstration
        batch = []
        for element in elements:
            batch.append(element)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if not drop_remainder and batch:
            yield batch

# Instantiation (batch_size is NOT in init)
batcher = SimpleBatcher(
    StructuralConfig(stochastic=False)
)

# Usage
# batches = list(batcher(data_stream, batch_size=32))
```

---

## Summary of Differences

| Feature | OperatorModule | StructuralModule |
| :--- | :--- | :--- |
| **Primary Role** | Data Transformation | Data Organization |
| **Learnable?** | Yes | No |
| **Differentiable?** | Yes | No |
| **Configuration** | Immutable config; learnable parameters live in `nnx.Param` | Immutable constants |
| **Examples** | Normalization, Augmentation | Batching, Sampling, Sharding |

## The DAG Execution Model

Datarax pipelines are constructed as a Directed Acyclic Graph (DAG) of these modules, wrapped in `Node` containers.

-   **Stage**: Any `nnx.Module` placed in `Pipeline(stages=[...])`. `OperatorModule` subclasses get an optimized fast path; plain `nnx.Module`s receive the dict batch directly.
-   **Source**: A `DataSourceModule` passed to `Pipeline(source=...)`. No wrapper class needed.
-   **Batching**: Configured via the `batch_size` argument on `Pipeline(...)`. No wrapper node.

Data flows through these nodes, with `DataraxModule` ensuring that state is correctly propagated and managed at every step.
