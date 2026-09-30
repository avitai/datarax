# Operator Cheat Sheet

Copy-paste-ready patterns for all Datarax operator types.

## MapOperator

Applies a function to specific fields in the data dictionary.

```python
from datarax.operators import MapOperator, MapOperatorConfig

# Transform a single field (fn signature is always fn(x, key))
normalize = MapOperator(
    MapOperatorConfig(subtree={"image": None}),
    fn=lambda x, key: x / 255.0,
    rngs=nnx.Rngs(0),
)

# Transform multiple fields
scale = MapOperator(
    MapOperatorConfig(subtree={"image": None, "mask": None}),
    fn=lambda x, key: x.astype(jnp.float32),
    rngs=nnx.Rngs(0),
)

# Full-tree mode (applies to entire data dict)
identity = MapOperator(
    MapOperatorConfig(subtree=None),
    fn=lambda x, key: x,
    rngs=nnx.Rngs(0),
)
```

## ElementOperator

Applies a function to the entire Element (data + state + metadata).

```python
from datarax.operators import ElementOperator, ElementOperatorConfig

# Deterministic element transform (fn signature is always fn(element, key))
def add_length(element, key):
    text = element.data["text"]
    length = jnp.array(text.shape[0])
    return element.update_data({"text": text, "length": length})

length_op = ElementOperator(
    ElementOperatorConfig(),
    fn=add_length,
    rngs=nnx.Rngs(0),
)

# Stochastic element transform (the per-record key arrives as the second arg)
def add_noise(element, key):
    image = element.data["image"]
    noise = 0.05 * jax.random.normal(key, image.shape, dtype=image.dtype)
    return element.update_data({"image": image + noise})

noise_op = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="augment"),
    fn=add_noise,
    rngs=nnx.Rngs(augment=0),
)
```

## Custom OperatorModule Subclass

For operators with learnable parameters or complex state.

```python
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig
import jax.numpy as jnp
import flax.nnx as nnx
from datarax.core.element_batch import Element
import jax
from typing import Any

class MyOperator(OperatorModule):
    def __init__(self, config, *, rngs):
        super().__init__(config, rngs=rngs)
        self.scale = nnx.Param(jnp.ones(()))  # Learnable parameter

    # apply() is a pure function of one record: it takes and returns an Element.
    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        scaled = data["image"] * self.scale.value
        return element.replace(data={**data, "image": scaled})

op = MyOperator(OperatorConfig(), rngs=nnx.Rngs(0))
```

## Image Operators

Built-in operators for common image augmentations.

```python
from datarax.operators import ProbabilisticOperator, ProbabilisticOperatorConfig
from datarax.operators.modality.image import (
    BrightnessOperator, BrightnessOperatorConfig,
    ContrastOperator, ContrastOperatorConfig,
    FlipOperator, FlipOperatorConfig,
    RandomCropOperator, RandomCropOperatorConfig,
    RotationOperator, RotationOperatorConfig,
    NoiseOperator, NoiseOperatorConfig,
)

# Random crop: pad 4 pixels, crop 32x32 at each record's own offset (torchvision's
# RandomCrop(32, padding=4)); eval mode takes the centre crop
crop = RandomCropOperator(
    RandomCropOperatorConfig(
        field_key="image", size=(32, 32), padding=4,
        stochastic=True, stream_name="crop",
    ),
    rngs=nnx.Rngs(crop=0),
)

# Random horizontal flip with probability 0.5 (torchvision's RandomHorizontalFlip()):
# FlipOperator is deterministic; ProbabilisticOperator decides per record
flip = ProbabilisticOperator(
    ProbabilisticOperatorConfig(probability=0.5),
    operator=FlipOperator(FlipOperatorConfig(field_key="image", axis="horizontal")),
    rngs=nnx.Rngs(augment=0),
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
        ),
    operators=[brightness, contrast, noise],
)

# Apply with probability
from datarax.operators import ProbabilisticOperator, ProbabilisticOperatorConfig

maybe_noise = ProbabilisticOperator(
    ProbabilisticOperatorConfig(probability=0.5),
    operator=noise, rngs=nnx.Rngs(0),
)
```

## Using Operators in Pipelines

```python
from datarax.pipeline import Pipeline

pipeline = Pipeline(source=source, stages=[brightness, contrast], batch_size=32, rngs=nnx.Rngs(0))
for batch in pipeline:
    augmented_images = batch["image"]
```
