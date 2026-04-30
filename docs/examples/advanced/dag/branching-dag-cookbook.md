# Branching DAG Cookbook: Branch, Merge, Parallel

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~1 min |
| **Prerequisites** | DAG Fundamentals Guide |
| **Format** | Python + Jupyter |

## Overview

Datarax pipelines compose via two surfaces:

- `Pipeline(stages=[...])` for linear chains
- `Pipeline.from_dag(nodes=, edges=, sink=)` for branching topologies

The legacy `Branch` / `Merge` / `Parallel` node classes were removed
because each pattern is a small `nnx.Module` plus the right `edges`
declaration. This cookbook ships one runnable recipe per pattern,
plus the verified output shape so you can sanity-check your own
variant.

## Coming from the legacy API?

| Legacy node | New idiom |
|-------------|-----------|
| `Branch(condition=, true_path=, false_path=)` | An `nnx.Module` whose `__call__` uses `jax.lax.cond` (or `nnx.cond` for state-bearing branches) |
| `Parallel([branch_a, branch_b])` | Two nodes with `edges[name] = []` — both consume the source |
| `Merge(strategy="stack")` | An `nnx.Module` whose `__call__` accepts the predecessors as positional args and returns the merged batch |

## Files

- **Python Script**: [`examples/advanced/dag/02_branching_dag_cookbook.py`](https://github.com/avitai/datarax/blob/main/examples/advanced/dag/02_branching_dag_cookbook.py)
- **Jupyter Notebook**: [`examples/advanced/dag/02_branching_dag_cookbook.ipynb`](https://github.com/avitai/datarax/blob/main/examples/advanced/dag/02_branching_dag_cookbook.ipynb)

## Quick Start

```bash
python examples/advanced/dag/02_branching_dag_cookbook.py
```

```bash
jupyter lab examples/advanced/dag/02_branching_dag_cookbook.ipynb
```

## Key Recipes

### 1. Parallel + Stack-merge

Two branches consume the source independently; the merge node
stacks both outputs along a new axis.

```python
class _Stack(nnx.Module):
    def __call__(self, normalized, inverted):
        return {
            "image": jnp.stack(
                [normalized["image"], inverted["image"]],
                axis=1,
            ),
            "label": normalized["label"],
        }


pipeline = Pipeline.from_dag(
    source=source,
    nodes={"normalize": _Normalize(), "invert": _Invert(), "stack": _Stack()},
    edges={
        "normalize": [],
        "invert": [],
        "stack": ["normalize", "invert"],  # order matches __call__ args
    },
    sink="stack",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
# Output: image (8, 2, 32, 32, 3)
```

Each merge enumerates its output fields explicitly. The image is
genuinely merged across branches; the label is invariant (neither
``_Normalize`` nor ``_Invert`` mutates it) so it's copied from one
side. Avoid ``return {**normalized, "image": ...}`` — that idiom
silently encodes the assumption that *all* non-image fields are
identical between branches, which fails as soon as a future stage
mutates one of them.

### 2. Average-merge

Same topology as recipe 1, different merge logic. Switching from
"stack" to "average" only swaps the merge `nnx.Module` — the
topology declaration is unchanged.

```python
class _Average(nnx.Module):
    def __call__(self, brightened, inverted):
        return {
            "image": (brightened["image"] + inverted["image"]) / 2,
            "label": brightened["label"],
        }
```

### 3. Branch (conditional routing)

```python
class _BrightenIfDark(nnx.Module):
    def __call__(self, batch):
        is_dark = jnp.mean(batch["image"]) < 0.3
        new_image = jax.lax.cond(
            is_dark,
            lambda img: jnp.clip(img + 0.1, 0.0, 1.0),
            lambda img: img,
            batch["image"],
        )
        return {**batch, "image": new_image}


pipeline = Pipeline(
    source=source,
    stages=[_BrightenIfDark()],
    batch_size=8,
    rngs=nnx.Rngs(0),
)
```

For state-bearing branches (one or both branches need to update an
`nnx.Module`'s parameters), substitute `nnx.cond` for `jax.lax.cond`
so the state is correctly tracked across the conditional.

### 4. Triple-Branch Parallel + channel-concat

Parallelism is just topology — N branches that all consume the
source, plus a merge that takes them as N positional arguments.

```python
class _Concat3(nnx.Module):
    def __call__(self, normalized, brightened, inverted):
        return {
            "image": jnp.concatenate(
                [normalized["image"], brightened["image"], inverted["image"]],
                axis=-1,
            ),
            "label": normalized["label"],
        }
```

## Verified Outputs

| Recipe | Output shape | Notes |
|--------|--------------|-------|
| Parallel + Stack | `(8, 2, 32, 32, 3)` | axis=1 holds the two branches |
| Parallel + Average | `(8, 32, 32, 3)` | element-wise mean |
| Branch (`jax.lax.cond`) | `(8, 32, 32, 3)` | mean unchanged on bright batches |
| Triple-Branch + Concat | `(8, 32, 32, 9)` | three RGB streams along channels |

## Inspecting the Topology

The resolved execution order is exposed via `pipeline.stages`. For a
DAG it is the topological order the executor will use:

```python
for stage in triple_branch.stages:
    print(type(stage).__name__)
# _Brighten
# _Invert
# _Normalize
# _Concat3
```

## Best Practices

| Practice | Why |
|----------|-----|
| Name nodes after their **role**, not their type | `"merge"` reads better than `"stack_module_1"` in tracebacks |
| Order `edges[sink] = [pred_a, pred_b]` to match the merge's `__call__` parameters | The framework passes args positionally in this order |
| Use `nnx.cond` (not `jax.lax.cond`) when a branch updates module state | `jax.lax.cond` doesn't track NNX state; `nnx.cond` does |
| Prefer linear `stages=[...]` when no branching is needed | One less concept; identical performance |

## Next Steps

- [DAG Fundamentals Guide](dag-fundamentals-guide.md) — full walkthrough of `Pipeline` and `Pipeline.from_dag`.
- [Pipeline Tutorial](../../core/pipeline-tutorial.md) — linear pipelines for beginners.
- [DAG Construction Guide](../../../user_guide/dag_construction.md) — user-guide reference for both modes.

## API Reference

- [`Pipeline.from_dag`](../../../user_guide/dag_construction.md#3-branching-dag) — branching constructor signature.
- [Flax NNX `cond`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/transformations.html) — state-tracking conditional.
- [`jax.lax.cond`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) — stateless conditional primitive.
