# Best Practices for Flax NNX Module Design

This guide outlines best practices for designing and implementing Flax NNX-based modules in Datarax. Following these practices ensures modules are consistent, maintainable, and compatible with JAX transformations.

## Core Principles

Flax NNX is built on fundamental principles:

1. **Python Reference Semantics**: NNX modules are regular Python objects with reference sharing and mutability
2. **Eager Initialization**: All parameters are created at module instantiation time
3. **Explicit State Management**: State is managed through Variables with clear ownership
4. **Simplified Transforms**: Lifted transforms that work directly with NNX objects

## Module Structure

### Standard Module Pattern

```python
from flax import nnx
import jax.numpy as jnp

class MyModule(nnx.Module):
    """Module description.

    More detailed explanation of what the module does.
    """

    def __init__(
        self,
        features_in: int,
        features_out: int,
        *,  # Keyword-only arguments after this
        rngs: nnx.Rngs,
    ):
        """Initialize the module.

        Args:
            features_in: Input feature dimension
            features_out: Output feature dimension
            rngs: RNG container for parameter initialization
        """
        # Parameters are created eagerly (no lazy initialization)
        self.kernel = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (features_in, features_out))
        )
        self.bias = nnx.Param(jnp.zeros((features_out,)))

        # Store static configuration (not wrapped in Variable)
        self.features_in = features_in
        self.features_out = features_out

    def __call__(self, x: jax.Array) -> jax.Array:
        """Process input data.

        Args:
            x: Input array of shape (batch, features_in)

        Returns:
            Output array of shape (batch, features_out)
        """
        # Variables support arithmetic operators directly
        return x @ self.kernel + self.bias
```

### Key Points

- **Keyword-only RNGs**: Use `*, rngs: nnx.Rngs` to enforce keyword-only syntax
- **Eager initialization**: Parameters are created immediately in `__init__`
- **Direct state ownership**: Modules hold their parameters as instance attributes
- **Static vs dynamic**: Store hyperparameters (shapes, counts) as static attributes; only use Variables for what needs to be traced by JAX

## Variable Types

Use the appropriate NNX variable type for each state:

| Variable Type | Purpose | Example |
|--------------|---------|---------|
| `nnx.Param` | Trainable parameters (weights, biases) | `self.kernel = nnx.Param(weights)` |
| `nnx.BatchStat` | Batch statistics (running means) | `self.running_mean = nnx.BatchStat(jnp.zeros(dim))` |
| `nnx.Variable` | General state variables | `self.count = nnx.Variable(jnp.array(0))` |
| Custom | Domain-specific state | `class Count(nnx.Variable): pass` |

```python
class StatefulModule(nnx.Module):
    def __init__(self, dim: int, *, rngs: nnx.Rngs):
        # Trainable parameter
        self.weight = nnx.Param(nnx.initializers.lecun_normal()(rngs.params(), (dim, dim)))

        # Batch statistics
        self.running_mean = nnx.BatchStat(jnp.zeros(dim))

        # General state
        self.step_count = nnx.Variable(jnp.array(0))
```

### Accessing Variable Values

NNX Variables implement numeric operators, so you can use them directly in expressions:

```python
def __call__(self, x):
    # Both forms work - Variables support arithmetic operators
    y = x @ self.weight  # Direct use (preferred for readability)
    y = x @ self.weight[...]  # Explicit slice access (for Array variables)

    return y
```

> **Note**: The `.value` attribute is deprecated as of Flax 0.12.0. Use `variable[...]` for Array variables or `variable.get_value()` for other types.

### Updating State

In-place mutation of Variable values is supported and is the standard pattern:

```python
def __call__(self, x):
    # In-place update using slice notation (recommended for Flax 0.12.0+)
    self.step_count[...] = self.step_count[...] + 1

    # For arrays, use JAX update patterns
    self.running_mean[...] = 0.9 * self.running_mean[...] + 0.1 * x.mean()

    return self.process(x)
```

## RNG Handling

### Constructor Pattern

Always use keyword-only `rngs` parameter:

```python
def __init__(
    self,
    dim: int,
    *,  # Keyword-only parameters after this
    rngs: nnx.Rngs,
):
    # Access specific RNG streams
    self.kernel = nnx.Param(
        nnx.initializers.lecun_normal()(rngs.params(), (dim, dim))
    )
```

### Named RNG Streams

Use named streams for different purposes:

```python
# Create RNGs with named streams
rngs = nnx.Rngs(params=0, dropout=1, augment=2)

# Or with a single default seed
rngs = nnx.Rngs(0)  # Creates 'default' stream
```

### Using RNGs in Forward Pass

For modules that need randomness during forward pass (like Dropout):

```python
class StochasticModule(nnx.Module):
    def __init__(self, rate: float = 0.1, *, rngs: nnx.Rngs):
        self.rate = rate
        self.rngs = rngs

    def __call__(self, x):
        # Get a key from the RNG stream
        key = self.rngs.dropout()
        mask = jax.random.bernoulli(key, 1 - self.rate, x.shape)
        return jnp.where(mask, x / (1 - self.rate), 0)
```

## Training and Evaluation Modes

Use the built-in `train()` and `eval()` methods:

```python
model = MLP([784, 256, 10], rngs=nnx.Rngs(0))

# Training mode: Dropout enabled, BatchNorm uses batch statistics
model.train()
y_train = model(x_batch)

# Evaluation mode: Dropout disabled, BatchNorm uses running statistics
model.eval()
y_eval = model(x_batch)
```

## Module Composition

### Nested Modules

Compose modules by assigning them as attributes:

```python
class MLP(nnx.Module):
    def __init__(self, features: list[int], *, rngs: nnx.Rngs):
        # Use nnx.List for module collections (required in Flax 0.12.0+)
        self.layers = nnx.List([
            nnx.Linear(features[i], features[i + 1], rngs=rngs)
            for i in range(len(features) - 1)
        ])
        self.dropout = nnx.Dropout(0.1, rngs=rngs)

    def __call__(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = nnx.relu(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)
```

> **Important**: As of Flax 0.12.0, plain Python lists containing modules must be wrapped with `nnx.List()`. This ensures proper parameter tracking and JAX transformation compatibility.

### Module Collections

Use `nnx.Dict` and `nnx.List` for dynamic module collections:

```python
class MultiHeadModel(nnx.Module):
    def __init__(self, num_heads: int, dim: int, *, rngs: nnx.Rngs):
        # Use nnx.Dict for named collections
        self.heads = nnx.Dict({
            f'head_{i}': nnx.Linear(dim, dim, rngs=rngs)
            for i in range(num_heads)
        })

        # Use nnx.List for indexed collections
        self.layers = nnx.List([
            nnx.Linear(dim, dim, rngs=rngs)
            for _ in range(3)
        ])
```

> **Why use nnx.Dict/List?** Plain Python `dict`/`list` won't properly track parameters, state, or integrate with NNX transformations.

## JAX Transformations

### Using nnx.jit

For automatic state management:

```python
@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)
        return jnp.mean((y_pred - y) ** 2)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss
```

### Using the Functional API

For pure JAX transforms, use split/merge:

```python
graphdef, state = nnx.split(model)

@jax.jit
def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    y = model(x)
    _, state = nnx.split(model)
    return y, state

y, state = forward(graphdef, state, x)
nnx.update(model, state)  # Propagate state back
```

### StateAxes for Fine-Grained Control

Control how different state types are transformed:

```python
# Vectorize parameters, broadcast counts
state_axes = nnx.StateAxes({nnx.Param: 0, Count: None})
y = nnx.vmap(forward_fn, in_axes=(state_axes, 0))(model, x_batch)
```

## Performance Best Practices

### Vectorization

Use vectorized operations, not Python loops:

```python
# Good - vectorized
def transform_batch(self, batch):
    return batch * self.scale

# Avoid - loop over batch
def transform_batch(self, batch):
    results = []
    for item in batch:
        results.append(item * self.scale)
    return jnp.stack(results)
```

### Avoid Python Control Flow on Traced Values

```python
# Bad - JAX cannot trace this
def process(self, x):
    if x.sum() > 0:  # Depends on traced value
        return x * 2
    return x

# Good - Use JAX control flow
def process(self, x):
    return jax.lax.cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x,
        x
    )
```

## Module Introspection

Use functional forms for iteration (instance methods are deprecated):

```python
# Iterate over direct children
for name, child in nnx.iter_children(model):
    print(f"Child: {name} -> {type(child)}")

# Iterate over all modules in tree
for path, module in nnx.iter_modules(model):
    print(f"Module at {path}: {type(module)}")

# Iterate over entire graph including variables
for path, value in nnx.iter_graph(model):
    if isinstance(value, nnx.Variable):
        print(f"Variable at {path}: {value[...].shape}")
```

## Visualization

```python
# Rich visualization (in Jupyter/Colab)
nnx.display(model)

# Tabular summary
print(nnx.tabulate(model)(jnp.ones((1, 784))))
```

## See Also

- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx_basics.html)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Datarax Troubleshooting Guide](troubleshooting_guide.md)
