# Integrating datarax with your model

datarax is a data-loading and processing library. It does not constrain
your model architecture, loss form, optimizer choice, or training loop
shape. Pick the integration tier that fits your needs:

| Tier | API | When to use |
|---|---|---|
| **A — iterator** | `for batch in pipeline: ...` | Prototyping, eval, any loop where iterator overhead is not the bottleneck |
| **B — single step** | `batch = pipeline.step()` | When you own your own JIT / scan / grad orchestration |
| **C — scan epoch** | `pipeline.scan(step_fn, modules=(...), length=...)` | Production training; entire epoch compiles to one XLA graph |

Tier C delivers an order-of-magnitude speedup over the iterator path on
typical training workloads because the entire epoch (data fetch +
augmentation + forward + loss + grad + optimizer update) becomes one
XLA call.

## What you must implement

1. **Model** — a standard `flax.nnx.Module`. Constructor takes `rngs`.
   Has a `__call__` of any signature that consumes a batch.

2. **Loss function** — a plain Python callable, typically
   `(model, batch) -> scalar`. No datarax-specific base class.

3. **Optimizer** — your choice: `nnx.Optimizer(model, optax.adam(...))`,
   raw optax, or custom. datarax does not know optimizers exist.

4. **Step function** — for Tier C, the body of one training step:

   ```python
   def step_fn(model, optimizer, batch):
       loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
       optimizer.update(model, grads)
       return loss
   ```

5. **Pipeline** — pick a construction shape:

   - Linear: `Pipeline(source=..., stages=[...], batch_size=N, rngs=...)`
   - Declarative DAG: `Pipeline.from_dag(source=..., nodes=..., edges=..., sink=..., batch_size=N, rngs=...)`
   - Custom: subclass `Pipeline` and override `__call__`.

## Examples in this directory

- `01_ml_classification.py` — image classification with a small CNN.
  Demonstrates Tier A and Tier C side by side.
- `02_sciml_pinn.py` — toy physics-informed neural network with a
  data-fit loss + PDE residual loss combined in one `step_fn`.
- `03_calibration_eval.py` — pure inference (no optimizer, no
  gradients) showing that Tier C works equally well for evaluation
  and calibration sweeps.

## Differentiability

datarax pipelines are JAX-traceable end-to-end. Within a single
`pipeline.scan` call the trace covers data fetch, every stage, the
user's loss function, and any `nnx.value_and_grad` evaluated inside
the step body — gradients flow through the entire epoch as one XLA
graph.

Shuffling and differentiability are **orthogonal**. Shuffled sources
produce integer indices to select records; integer arrays don't carry
gradients in JAX (this is a fundamental JAX semantic, not a datarax
choice). Gradients flow through the **values** at the shuffled
indices, not through the index choice itself. This is the standard
JAX/Flax data-loading pattern and matches what every JAX training
loop already assumes.

If a user case calls for differentiating through the sampling decision
itself (rare in data loading; common in discrete-latent-variable
models), the standard JAX patterns apply at the user level: REINFORCE
/ score-function estimators, Gumbel-softmax relaxation, or
straight-through estimators. datarax pipelines compose with all of
these because they treat sampling as a black box producing a batch.

## Compatibility notes

- **Train / inference mode**: toggle the model's mode (e.g. via the
  standard NNX mode flags) outside `pipeline.scan`. Pipeline does not
  touch model mode.
- **Stochastic models**: pass an `nnx.Rngs` separate from the
  pipeline's. The pipeline's `rngs` is for source / stage stochasticity.
- **Distributed (multi-device) training**: compose `jax.shard_map` /
  `nnx.pmap` on top. Orthogonal to the pipeline.
- **Checkpointing**: model, optimizer, and pipeline are all
  `nnx.Module`s. Use Orbax to checkpoint them as a tuple. Pipeline's
  iteration position (`_position`) is an `nnx.Variable` and serializes
  with the rest of its state.
