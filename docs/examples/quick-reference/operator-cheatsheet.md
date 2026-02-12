# Operator Cheat Sheet

Copy-paste-ready patterns for all Datarax operator types.

## MapOperator

Applies a function to specific fields in the data dictionary.

```python
from datarax.operators import MapOperator, MapOperatorConfig

# Transform a single field
normalize = MapOperator(
    MapOperatorConfig(subtree={"image": None}),
    fn=lambda x: x / 255.0,
    rngs=nnx.Rngs(0),
)

# Transform multiple fields
scale = MapOperator(
    MapOperatorConfig(subtree={"image": None, "mask": None}),
    fn=lambda x: x.astype(jnp.float32),
    rngs=nnx.Rngs(0),
)

# Full-tree mode (applies to entire data dict)
identity = MapOperator(
    MapOperatorConfig(subtree=None),
    fn=lambda x: x,
    rngs=nnx.Rngs(0),
)
```

## ElementOperator

Applies a function to the entire Element (data + state + metadata).

```python
from datarax.operators import ElementOperator, ElementOperatorConfig

# Deterministic element transform
def add_length(element):
    text = element.data["text"]
    length = jnp.array(text.shape[0])
    return element.update_data({"text": text, "length": length})

length_op = ElementOperator(
    ElementOperatorConfig(),
    fn=add_length,
    rngs=nnx.Rngs(0),
)

# Stochastic element transform
def random_crop(element, *, rngs):
    key = rngs.augment()
    # ... crop logic using key
    return element.update_data({"image": cropped})

crop_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="augment"),
    fn=random_crop,
    rngs=nnx.Rngs(0),
)
```

## Custom OperatorModule Subclass

For operators with learnable parameters or complex state.

```python
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Element
import flax.nnx as nnx

class MyOperator(OperatorModule):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.scale = nnx.Param(jnp.ones(()))  # Learnable parameter

    def apply(self, element: Element, *, rngs=None) -> Element:
        scaled = element.data["image"] * self.scale.value
        return element.update_data({"image": scaled})

op = MyOperator(OperatorConfig(), rngs=nnx.Rngs(0))
```

## Image Operators

Built-in operators for common image augmentations.

```python
from datarax.operators.modality.image import (
    BrightnessOperator, BrightnessOperatorConfig,
    ContrastOperator, ContrastOperatorConfig,
    RotationOperator, RotationOperatorConfig,
    NoiseOperator, NoiseOperatorConfig,
)

# Brightness adjustment
brightness = BrightnessOperator(
    BrightnessOperatorConfig(
        field_key="image", brightness_range=(-0.2, 0.2),
        stochastic=True, stream_name="brightness",
    ),
    rngs=nnx.Rngs(0),
)

# Contrast adjustment
contrast = ContrastOperator(
    ContrastOperatorConfig(
        field_key="image", contrast_range=(0.8, 1.2),
        stochastic=True, stream_name="contrast",
    ),
    rngs=nnx.Rngs(0),
)

# Random rotation
rotation = RotationOperator(
    RotationOperatorConfig(
        field_key="image", angle_range=(-15, 15),
        stochastic=True, stream_name="rotation",
    ),
    rngs=nnx.Rngs(0),
)

# Gaussian noise
noise = NoiseOperator(
    NoiseOperatorConfig(
        field_key="image", mode="gaussian", noise_std=0.05,
        stochastic=True, stream_name="noise",
    ),
    rngs=nnx.Rngs(0),
)
```

## Stochastic vs Deterministic

| Mode | Config | RNG | Use Case |
|------|--------|-----|----------|
| Deterministic | `stochastic=False` | Not needed | Normalization, type casting |
| Stochastic | `stochastic=True, stream_name="aug"` | Required | Augmentation, dropout |

```python
# Deterministic: no randomness
det_config = OperatorConfig(stochastic=False)

# Stochastic: requires stream_name
stoch_config = OperatorConfig(stochastic=True, stream_name="augment")
```

## batch_strategy: vmap vs scan

Control how operators process batch elements.

| Strategy | Memory | Speed | Use When |
|----------|--------|-------|----------|
| `"vmap"` (default) | O(B) | Fast (parallel) | Small operators, training |
| `"scan"` | O(1) | Slower (sequential) | Memory-heavy operators (CREPE, large CNNs) |

```python
# Default: vmap (parallel, higher memory)
config = OperatorConfig(batch_strategy="vmap")

# Low memory: scan (sequential, O(1) memory)
config = OperatorConfig(batch_strategy="scan")
```

## Composition

Chain operators together using `CompositeOperatorModule`.

```python
from datarax.operators import (
    CompositeOperatorModule, CompositeOperatorConfig, CompositionStrategy,
)

# Sequential composition (op1 -> op2 -> op3)
composite = CompositeOperatorModule(
    CompositeOperatorConfig(
        strategy=CompositionStrategy.SEQUENTIAL,
        operators=[brightness, contrast, noise],
    ),
    rngs=nnx.Rngs(0),
)

# Apply with probability
from datarax.operators import ProbabilisticOperator, ProbabilisticOperatorConfig

maybe_noise = ProbabilisticOperator(
    ProbabilisticOperatorConfig(operator=noise, probability=0.5),
    rngs=nnx.Rngs(0),
)
```

## Using Operators in Pipelines

```python
from datarax.dag import from_source, OperatorNode

pipeline = from_source(source, batch_size=32)
pipeline.add(OperatorNode(brightness))
pipeline.add(OperatorNode(contrast))

for batch in pipeline:
    augmented_images = batch["image"]
```
