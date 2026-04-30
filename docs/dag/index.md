# DAG

Datarax pipelines support directed-acyclic-graph composition through
the [`Pipeline`](../user_guide/dag_construction.md) class. A linear
list of stages is the common case; `Pipeline.from_dag` adds branching
and merging when stages need to consume the source independently or
combine outputs from multiple predecessors.

## Composition modes

| Mode | Constructor | When to use |
|------|-------------|-------------|
| Linear | `Pipeline(source=..., stages=[...], ...)` | Each stage's output feeds the next stage |
| DAG | `Pipeline.from_dag(source=..., nodes={...}, edges={...}, sink=...)` | Branching, merging, or named-node inspection |

`★ Insight ─────────────────────────────────────`

- `Pipeline` is itself an `nnx.Module` — gradients flow through stages naturally
- `Pipeline.step` is `@nnx.jit`-decorated; `Pipeline.scan` compiles whole epochs to one XLA graph
- Cycles in the DAG raise `ValueError` at construction time

`─────────────────────────────────────────────────`

## Linear pipeline

```python
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))
pipeline = Pipeline(
    source=source,
    stages=[normalize, augment],
    batch_size=32,
    rngs=nnx.Rngs(0),
)

for batch in pipeline:
    loss = train_step(batch)
```

## Branching DAG

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

`merge` consumes both `augment` and `normalize` outputs as positional
arguments. The `edges` mapping declares each node's predecessors;
empty list means "consumes the source directly."

## Cookbook: Branch, Merge, Parallel

The legacy ``Branch`` / ``Parallel`` / ``Merge`` node classes are
gone, but each pattern is expressible as a small ``nnx.Module`` plus
the right ``edges``. The recipes below are runnable; each ends with
a verified output line.

For the full runnable file (four recipes including a triple-branch
parallel), see the
[Branching DAG Cookbook example](../examples/advanced/dag/branching-dag-cookbook.md).

### Parallel: two stages consuming the source independently

Two nodes whose ``edges[name] = []`` both feed off the source
directly. The topological executor schedules them in either order
(they are independent) and any downstream node that lists both as
predecessors waits for both to finish.

```python
from flax import nnx
import jax.numpy as jnp

from datarax.pipeline import Pipeline


class _Normalize(nnx.Module):
    def __call__(self, batch):
        return {**batch, "image": batch["image"] / 255.0}


class _Invert(nnx.Module):
    def __call__(self, batch):
        return {**batch, "image": 1.0 - batch["image"]}


class _Stack(nnx.Module):
    """Merge: stack the two parallel branches along a new axis.

    The merge enumerates each output field explicitly. The image is
    genuinely merged (stacked); the label is identical on both sides
    so we copy it from either branch. Spreading one branch with
    ``**normalized`` would work for this case but would silently
    drop changes if a future stage mutated the label — be explicit.
    """

    def __call__(self, normalized, inverted):
        return {
            "image": jnp.stack([normalized["image"], inverted["image"]], axis=1),
            "label": normalized["label"],
        }


pipeline = Pipeline.from_dag(
    source=source,
    nodes={"normalize": _Normalize(), "invert": _Invert(), "stack": _Stack()},
    edges={
        "normalize": [],          # parallel branch 1 — consumes source
        "invert": [],             # parallel branch 2 — consumes source
        "stack": ["normalize", "invert"],
    },
    sink="stack",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
batch = next(iter(pipeline))
# image shape = (8, 2, H, W, C)  ← two branches stacked along axis=1
```

### Merge: a downstream node reduces multiple predecessors

Any callable signature works — the args arrive in the order
``edges[sink] = [pred_1, pred_2, ...]`` declares.

```python
class _Average(nnx.Module):
    """Element-wise mean of two branches."""

    def __call__(self, brightened, inverted):
        return {
            "image": (brightened["image"] + inverted["image"]) / 2,
            "label": brightened["label"],
        }


pipeline = Pipeline.from_dag(
    source=source,
    nodes={"brighten": _Brighten(), "invert": _Invert(), "merge": _Average()},
    edges={"brighten": [], "invert": [], "merge": ["brighten", "invert"]},
    sink="merge",
    batch_size=8,
    rngs=nnx.Rngs(0),
)
```

### Branch: conditional routing via ``jax.lax.cond``

Place the conditional inside an ``nnx.Module`` so the predicate is
JIT-traceable. ``jax.lax.cond`` works for stateless branches;
``nnx.cond`` is the equivalent that preserves ``nnx.Module`` state
when one of the branches is a learned module.

```python
import jax
import jax.numpy as jnp
from flax import nnx


class _BrightenIfDark(nnx.Module):
    """Apply brighten if the batch's mean image is below 0.3, else normalize."""

    def __call__(self, batch):
        is_dark = jnp.mean(batch["image"]) < 0.3
        new_image = jax.lax.cond(
            is_dark,
            lambda img: jnp.clip(img + 0.1, 0.0, 1.0),
            lambda img: img / 255.0,
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

### Why no dedicated node class?

Expressing each pattern as a regular ``nnx.Module`` means:

- **Merge** can do anything (stack, mean, concat, weighted sum, even
  a learned aggregator) — no fixed strategy enum to extend.
- **Branch** can use any JIT-traceable predicate, including
  per-batch statistics from arbitrary upstream stages.
- **Parallel** needs no class at all — it is a topological property
  of the DAG, not a wrapper.

Each stage is also automatically checkpointable (it is an
``nnx.Module``), gradient-flowing (its parameters are leaves of
``nnx.state(pipeline)``), and JIT-friendly (folded into
``Pipeline.step`` and ``Pipeline.scan``).

## Whole-epoch JIT with `pipeline.scan`

```python
final_carry, outputs = pipeline.scan(
    step_fn=train_step,
    modules=(model, optimizer),
    length=steps_per_epoch,
    init_carry=initial_carry,
)
```

The whole epoch compiles as a single XLA graph; per-step Python
overhead is eliminated and gradients flow through the data path.

## Stage types

Any `nnx.Module` whose `__call__(batch) -> batch` transforms the
batch can be used as a stage:

- **`OperatorModule` subclasses** (e.g. `BrightnessOperator`,
  `NoiseOperator`): receive an `Element`, return an updated
  `Element`. Pipeline detects these and uses an optimized fast path.
- **Plain `nnx.Module`**: receives the dict batch directly. Use this
  for user-defined transforms.

## Real-World examples

- [Learned ISP Pipeline](../examples/advanced/differentiable/learned-isp-guide.md) — 5-stage differentiable ISP using `stages=[...]` with gradient flow through every stage.
- [DDSP Audio Synthesis](../examples/advanced/differentiable/ddsp-audio-synthesis.md) — Branching DAG for harmonic and noise synthesis.
- [DADA Learned Augmentation](../examples/advanced/differentiable/dada-learned-augmentation.md) — Pipelines composed with Gumbel-Softmax for differentiable augmentation search.

## See also

- [DAG Construction Guide](../user_guide/dag_construction.md) — User guide for both modes
- [DAG Fundamentals Example](../examples/advanced/dag/dag-fundamentals-guide.md) — Walk-through with a runnable script
- [Pipeline Tutorial](../examples/core/pipeline-tutorial.md) — Linear pipelines for beginners
