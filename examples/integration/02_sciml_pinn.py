# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# Physics-Informed Neural Network — Pipeline.scan with custom loss

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~2 min |
| **Prerequisites** | Pipeline Quickstart, JAX/Flax NNX basics, basic ODE knowledge |
| **Format** | Python + Jupyter |

## Overview

Trains a small MLP `u_theta(x)` to approximate the solution of
`du/dx = -u`, demonstrating that `Pipeline.scan` accommodates arbitrary
loss compositions — including those that take gradients of model outputs
with respect to inputs.

## Setup

```bash
uv pip install datarax
```

Activate the project virtualenv:

```bash
source activate.sh
```

## Learning Goals

By the end of this example, you will be able to:

1. Compose data-fit and PDE-residual losses inside a Pipeline scan body.
2. Take `jax.grad` of the model with respect to its input.
3. Use `Pipeline.scan` for a non-classification training loop.
"""

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# %% [markdown]
"""
## 1. Model — small MLP from R -> R
"""


# %%
class _MLP(nnx.Module):
    def __init__(self, *, hidden: int = 32, rngs: nnx.Rngs) -> None:
        self.l1 = nnx.Linear(1, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.tanh(self.l1(x))
        x = nnx.tanh(self.l2(x))
        return self.l3(x).squeeze(-1)


# %% [markdown]
"""
## 2. Combined loss: data fit + PDE residual
"""


# %%
def pinn_loss(model: _MLP, batch: dict) -> jax.Array:
    """Data MSE plus (du/dx + u)^2 residual.

    For ``du/dx = -u``, the residual ``du/dx + u`` should be zero
    everywhere; squaring it gives the physics-informed loss.
    """
    x = batch["x"]  # (B,)
    u_true = batch["u"]  # (B,)

    def model_scalar(xi: jax.Array) -> jax.Array:
        return model(xi.reshape(1, 1))[0]

    u_pred = jax.vmap(model_scalar)(x)
    data_mse = jnp.mean((u_pred - u_true) ** 2)

    du_dx = jax.vmap(jax.grad(model_scalar))(x)
    residual = du_dx + u_pred
    pde_mse = jnp.mean(residual**2)

    return data_mse + 0.1 * pde_mse


# %% [markdown]
"""
## 3. Synthetic data — sample x in [0, 2], compute u = exp(-x)
"""


# %%
def build_data(num_points: int = 256) -> dict:
    """Run build_data."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 2.0, size=(num_points,)).astype(np.float32)
    u = np.exp(-x).astype(np.float32)
    return {"x": jnp.asarray(x), "u": jnp.asarray(u)}


# %% [markdown]
"""
## 4. Driver — train via Pipeline.scan with the combined loss
"""


# %%
def main() -> None:
    """Run main."""
    print("=" * 70)
    print("SciML PINN: u_theta(x) approximates u'(x) = -u(x), u(0) = 1")
    print("=" * 70)

    data = build_data(num_points=256)
    batch_size = 32
    num_epochs = 5
    steps_per_epoch = 256 // batch_size

    pipeline = Pipeline(
        source=MemorySource(MemorySourceConfig(shuffle=False), data),
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
    )

    model = _MLP(hidden=32, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(3e-3), wrt=nnx.Param)

    def step_fn(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(pinn_loss)(model, batch)
        optimizer.update(model, grads)
        return loss

    for epoch in range(num_epochs):
        pipeline._position.value = jnp.int32(0)
        losses = pipeline.scan(step_fn, modules=(model, optimizer), length=steps_per_epoch)
        print(f"  epoch {epoch + 1}: combined loss = {float(jnp.mean(losses)):.5f}")

    # Evaluate on a fresh grid
    x_test = jnp.linspace(0.0, 2.0, 11).reshape(-1, 1)
    u_pred = jax.vmap(lambda xi: model(xi.reshape(1, 1))[0])(x_test.squeeze(-1))
    u_true = jnp.exp(-x_test.squeeze(-1))

    print("\n  x      u_true    u_pred    abs_err")
    for xi, ut, up in zip(x_test.squeeze(-1), u_true, u_pred):
        err = float(jnp.abs(ut - up))
        print(f"  {float(xi):.2f}   {float(ut):.4f}   {float(up):.4f}   {err:.4f}")


if __name__ == "__main__":
    main()
