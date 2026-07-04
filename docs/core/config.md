# Configuration System

Datarax uses a **typed configuration system** based on Python dataclasses. All module configurations inherit from a base class hierarchy that provides validation, immutability guarantees, and consistent behavior.

## Configuration Hierarchy

```
DataraxModuleConfig (base)
├── OperatorConfig (for parametric/learnable modules)
│   ├── MapOperatorConfig
│   ├── ElementOperatorConfig
│   └── BatchMixOperatorConfig
└── StructuralConfig (for structural processors)
    ├── HFEagerConfig
    └── TFDSEagerConfig
```

!!! note "Key points"

    - **OperatorConfig**: For parametric/learnable modules
    - **StructuralConfig**: For structural processors
    - All configs are frozen dataclasses — module state is mutable, configs never are
    - All configs validate on construction (`__post_init__`)
    - For shuffle configurations, use `seed=42` parameter

## Base Configuration

All configs inherit common settings:

```python
from datarax.core.config import DataraxModuleConfig

# Base attributes available to all configs:
config = DataraxModuleConfig(
    cacheable=False,           # Enable caching
    batch_stats_fn=None,       # Dynamic statistics function
    precomputed_stats=None,    # Static statistics
)
```

!!! note "Mutual Exclusivity"
    `batch_stats_fn` and `precomputed_stats` cannot both be set.

## Operator Configuration

For operators with mutable state and potential randomness:

```python
from datarax.core.config import OperatorConfig

# Deterministic operator
config = OperatorConfig(
    stochastic=False,
)

# Stochastic operator (requires stream_name)
config = OperatorConfig(
    stochastic=True,
    stream_name="augment",  # Required!
)
```

### Validation Rules

| stochastic | stream_name | Result |
|------------|-------------|--------|
| `False` | `None` | ✅ Valid deterministic |
| `True` | `"name"` | ✅ Valid stochastic |
| `True` | `None` | ❌ Error: stream_name required |
| `False` | `"name"` | ❌ Error: stream_name forbidden |

### Batch Strategy

`OperatorConfig` also exposes `batch_strategy`, controlling how the operator maps over the batch axis:

```python
from datarax.core.config import OperatorConfig

# Parallel (default): vmap over the batch axis
config = OperatorConfig(batch_strategy="vmap")

# Sequential, low-memory: scan over the batch axis
config = OperatorConfig(batch_strategy="scan")
```

| batch_strategy | Behavior |
|----------------|----------|
| `"vmap"` | Parallel map over the batch axis (default) |
| `"scan"` | Sequential map over the batch axis (lower memory) |

## Specific Operator Configs

### ElementOperatorConfig

For element-level transformations:

```python
from datarax.core.config import ElementOperatorConfig

# Deterministic
config = ElementOperatorConfig(stochastic=False)

# Stochastic
config = ElementOperatorConfig(
    stochastic=True,
    stream_name="element_aug",
)
```

### MapOperatorConfig

For per-array-leaf transformations:

```python
from datarax.core.config import MapOperatorConfig

# Full-tree mode (transform entire element)
config = MapOperatorConfig(subtree=None)

# Subtree mode (transform specific fields)
config = MapOperatorConfig(
    subtree={"image": None, "mask": None},
)
```

!!! note
    `MapOperator` currently enforces `stochastic=False`.

### BatchMixOperatorConfig

For batch-level mixing (MixUp/CutMix):

```python
from datarax.core.config import BatchMixOperatorConfig

# MixUp
config = BatchMixOperatorConfig(
    mode="mixup",
    alpha=0.4,
    label_field="label",
)

# CutMix
config = BatchMixOperatorConfig(
    mode="cutmix",
    alpha=1.0,
    data_field="image",
)
```

!!! note
    `BatchMixOperatorConfig` is always stochastic - `stochastic=True` is forced.

## Structural Configuration

For modules with frozen configuration (compile-time constants):

```python
from datarax.core.config import FrozenInstanceError, StructuralConfig

config = StructuralConfig(
    stochastic=False,
    stream_name=None,
)

# Every config is frozen after construction — mutation raises.
try:
    config.stochastic = True
except FrozenInstanceError:
    print("Configs are immutable; build a new one instead")
```

### Why Frozen?

Every config in Datarax freezes after construction — this is not unique to
`StructuralConfig`. What distinguishes `StructuralConfig` is *what* it configures
(structural processors), not that it alone is frozen. Immutable configs are used
because:

1. They represent compile-time constants for JIT
2. Changing config after construction breaks invariants
3. Immutability prevents subtle bugs

Module state remains mutable; only the config is frozen.

## Creating Custom Configs

Extend the base classes for custom modules:

```python
from dataclasses import dataclass
from datarax.core.config import OperatorConfig

@dataclass(frozen=True)
class MyCustomOperatorConfig(OperatorConfig):
    """Configuration for MyCustomOperator."""

    # Custom fields
    strength: float = 1.0
    mode: str = "default"

    def __post_init__(self):
        # Validate custom fields
        if self.strength <= 0:
            raise ValueError("strength must be positive")
        if self.mode not in ("default", "advanced"):
            raise ValueError(f"Unknown mode: {self.mode}")

        # Call parent validation
        super().__post_init__()
```

## Using with Modules

```python
from datarax.operators import ElementOperator
from datarax.core.config import ElementOperatorConfig

# Create config
config = ElementOperatorConfig(
    stochastic=True,
    stream_name="noise",
)

# Pass to module
operator = ElementOperator(
    config,
    fn=my_transform_fn,
    rngs=nnx.Rngs(42),
)
```

## See Also

- [Element Operator](../operators/element_operator.md) - Using ElementOperatorConfig
- [Composite Operator](../operators/composite_operator.md) - CompositeOperatorConfig
- [HF Source](../sources/hf_source.md) - HFEagerConfig
- [TFDS Source](../sources/tfds_source.md) - TFDSEagerConfig

---

## API Reference

::: datarax.core.config
