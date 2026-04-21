"""Datarax adapter -- reference implementation for the benchmark framework.

Wraps Datarax's public API (build_source_pipeline, MemorySource, DAGExecutor)
using the PipelineAdapter lifecycle: setup -> warmup -> iterate -> teardown.

Supports all 25 scenarios with transform chaining via
OperatorNode and topology dispatch for DAG-based scenarios. External
backends (TFDS, HuggingFace) are created via config.extra["backend"].

Design ref: Section 7.2 of the benchmark report.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks.adapters import register
from benchmarks.adapters._utils import cast_to_float32, normalize_uint8
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig
from datarax import build_source_pipeline, OperatorNode
from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.operators.element_operator import ElementOperator
from datarax.performance.synchronization import block_until_ready_tree
from datarax.sources import MemorySource, MemorySourceConfig


# ---------------------------------------------------------------------------
# Transform function library
# ---------------------------------------------------------------------------
# Each function follows the ElementOperator signature:
#   fn(element: Element, key: jax.Array) -> Element
#
# normalize_uint8 / cast_to_float32 are shared with Grain adapter (DRY).
# They use jax.tree.map (JIT-safe), NOT apply_to_dict (not JIT-safe).


def _normalize(element: Element, key: jax.Array) -> Element:
    """Normalize uint8 images to [0, 1] float32."""
    del key
    return element.replace(data=jax.tree.map(normalize_uint8, element.data))


def _cast_to_float32(element: Element, key: jax.Array) -> Element:
    """Cast all arrays to float32."""
    del key
    return element.replace(data=jax.tree.map(cast_to_float32, element.data))


def _scale(element: Element, key: jax.Array) -> Element:
    """Scale all values by 2.0."""
    del key
    new_data = jax.tree.map(lambda x: x * 2.0, element.data)
    return element.replace(data=new_data)


def _clip(element: Element, key: jax.Array) -> Element:
    """Clip values to [0, 1]."""
    del key
    new_data = jax.tree.map(lambda x: jnp.clip(x, 0.0, 1.0), element.data)
    return element.replace(data=new_data)


def _add(element: Element, key: jax.Array) -> Element:
    """Add 0.1 to all values."""
    del key
    new_data = jax.tree.map(lambda x: x + 0.1, element.data)
    return element.replace(data=new_data)


def _multiply(element: Element, key: jax.Array) -> Element:
    """Multiply all values by 0.9."""
    del key
    new_data = jax.tree.map(lambda x: x * 0.9, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# Stochastic transform functions (use key parameter for per-element RNG)
# ---------------------------------------------------------------------------


def _gaussian_noise(element: Element, key: jax.Array) -> Element:
    """Add Gaussian noise (std=0.05) to all arrays. Stochastic."""

    def add_noise(x):
        noise = jax.random.normal(key, shape=x.shape) * 0.05
        return x + noise

    return element.replace(data=jax.tree.map(add_noise, element.data))


def _random_brightness(element: Element, key: jax.Array) -> Element:
    """Adjust brightness by random delta in [-0.2, 0.2]. Stochastic."""
    delta = jax.random.uniform(key, minval=-0.2, maxval=0.2)
    new_data = jax.tree.map(lambda x: x + delta, element.data)
    return element.replace(data=new_data)


def _random_scale(element: Element, key: jax.Array) -> Element:
    """Scale by random factor in [0.8, 1.2]. Stochastic."""
    factor = jax.random.uniform(key, minval=0.8, maxval=1.2)
    new_data = jax.tree.map(lambda x: x * factor, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# Compute-heavy stochastic transforms (GPU-accelerated)
# ---------------------------------------------------------------------------


def _random_resized_crop(element: Element, key: jax.Array) -> Element:
    """Random crop (75% area) + bilinear resize to original shape. Stochastic.

    Uses a fixed crop ratio (JIT-compatible: dynamic_slice needs static sizes)
    with random position, then bilinear interpolation resize back to original
    dimensions. ~20 FLOPs/pixel from bilinear interpolation.
    """

    def crop_and_resize(x: jax.Array) -> jax.Array:
        h, w, c = x.shape[0], x.shape[1], x.shape[2]
        # Fixed 75% crop (static sizes for JIT compatibility)
        crop_h, crop_w = (h * 3) // 4, (w * 3) // 4

        # Random crop position (dynamic start indices are JIT-safe)
        max_top = h - crop_h
        max_left = w - crop_w
        top = jax.random.randint(key, (), 0, jnp.maximum(max_top, 1))
        left = jax.random.randint(key, (), 0, jnp.maximum(max_left, 1))

        cropped = jax.lax.dynamic_slice(x, (top, left, 0), (crop_h, crop_w, c))

        # Bilinear resize back to original shape (compute-heavy)
        return jax.image.resize(cropped, (h, w, c), method="bilinear")

    new_data = jax.tree.map(crop_and_resize, element.data)
    return element.replace(data=new_data)


def _random_horizontal_flip(element: Element, key: jax.Array) -> Element:
    """Randomly flip images horizontally with 50% probability. Stochastic."""
    should_flip = jax.random.bernoulli(key, p=0.5)

    def maybe_flip(x: jax.Array) -> jax.Array:
        return jax.lax.cond(should_flip, lambda a: jnp.flip(a, axis=1), lambda a: a, x)

    new_data = jax.tree.map(maybe_flip, element.data)
    return element.replace(data=new_data)


def _color_jitter(element: Element, key: jax.Array) -> Element:
    """Random brightness, contrast, saturation adjustment. Stochastic, compute-heavy.

    ~15 FLOPs/pixel from per-channel operations.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    def jitter(x: jax.Array) -> jax.Array:
        # Brightness: add random delta [-0.4, 0.4]
        brightness = jax.random.uniform(k1, minval=-0.4, maxval=0.4)
        x = x + brightness

        # Contrast: scale toward mean by random factor [0.6, 1.4]
        contrast = jax.random.uniform(k2, minval=0.6, maxval=1.4)
        mean = jnp.mean(x, axis=(0, 1), keepdims=True)
        x = (x - mean) * contrast + mean

        # Saturation: interpolate toward grayscale by random factor [0.6, 1.4]
        saturation = jax.random.uniform(k3, minval=0.6, maxval=1.4)
        gray = jnp.mean(x, axis=-1, keepdims=True)
        x = (x - gray) * saturation + gray

        return jnp.clip(x, 0.0, 1.0)

    new_data = jax.tree.map(jitter, element.data)
    return element.replace(data=new_data)


def _gaussian_blur(element: Element, key: jax.Array) -> Element:
    """Apply Gaussian blur with random sigma. Stochastic, compute-heavy.

    Uses jax.lax.conv_general_dilated for 2D depthwise convolution
    with a 5x5 Gaussian kernel. ~50 FLOPs/pixel.
    """
    sigma = jax.random.uniform(key, minval=0.1, maxval=2.0)

    def blur(x: jax.Array) -> jax.Array:
        h, w, c = x.shape

        # Build 2D Gaussian kernel (5x5)
        coords = jnp.arange(5, dtype=jnp.float32) - 2.0
        kernel_1d = jnp.exp(-0.5 * (coords / sigma) ** 2)
        kernel_2d = jnp.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / jnp.sum(kernel_2d)

        # Reshape for depthwise conv: (H, W, C) -> (1, H, W, C)
        x_4d = x[jnp.newaxis]  # (1, H, W, C)

        # Kernel shape for depthwise: (kH, kW, 1, C)
        kernel_4d = kernel_2d[:, :, jnp.newaxis, jnp.newaxis]
        kernel_4d = jnp.broadcast_to(kernel_4d, (5, 5, 1, c))

        # Depthwise convolution with SAME padding
        blurred = jax.lax.conv_general_dilated(
            x_4d,
            kernel_4d,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=c,
        )
        return blurred[0]  # Remove batch dim

    new_data = jax.tree.map(blur, element.data)
    return element.replace(data=new_data)


def _random_solarize(element: Element, key: jax.Array) -> Element:
    """Randomly solarize (invert pixels above threshold). Stochastic.

    Threshold randomly sampled in [0.5, 1.0].
    """
    threshold = jax.random.uniform(key, minval=0.5, maxval=1.0)

    def solarize(x: jax.Array) -> jax.Array:
        return jnp.where(x > threshold, 1.0 - x, x)

    new_data = jax.tree.map(solarize, element.data)
    return element.replace(data=new_data)


def _random_grayscale(element: Element, key: jax.Array) -> Element:
    """Randomly convert to grayscale with 20% probability. Stochastic."""
    should_gray = jax.random.bernoulli(key, p=0.2)

    def maybe_gray(x: jax.Array) -> jax.Array:
        gray = jnp.mean(x, axis=-1, keepdims=True)
        gray_rgb = jnp.broadcast_to(gray, x.shape)
        return jax.lax.cond(should_gray, lambda _: gray_rgb, lambda _: x, None)

    new_data = jax.tree.map(maybe_gray, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# NLP / Tabular transforms for heavy scenarios (HNLP-1, HTAB-1, HMM-1)
# ---------------------------------------------------------------------------


def _log_transform(element: Element, key: jax.Array) -> Element:
    """Element-wise log1p for dense features (HTAB-1)."""
    del key
    new_data = jax.tree.map(lambda x: jnp.log1p(jnp.abs(x)), element.data)
    return element.replace(data=new_data)


def _hash_embedding_index(element: Element, key: jax.Array) -> Element:
    """Hash integers into embedding bucket indices (HTAB-1)."""
    del key
    new_data = jax.tree.map(lambda x: jnp.abs(x) % 10000, element.data)
    return element.replace(data=new_data)


def _create_attention_mask(element: Element, key: jax.Array) -> Element:
    """Create binary attention mask: 1 for real tokens, 0 for padding (HNLP-1)."""
    del key
    new_data = jax.tree.map(lambda x: (x != 0).astype(jnp.float32), element.data)
    return element.replace(data=new_data)


def _create_causal_mask(element: Element, key: jax.Array) -> Element:
    """Create lower-triangular causal attention mask (HNLP-1).

    O(seq_len^2) computation -- benefits from JIT compilation.
    """

    del key

    def make_mask(x: jax.Array) -> jax.Array:
        seq_len = x.shape[-1] if x.ndim > 1 else x.shape[0]
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))

    new_data = jax.tree.map(make_mask, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# Transform registries
# ---------------------------------------------------------------------------

# Deterministic: transform name -> function mapping
_TRANSFORM_FNS: dict[str, Any] = {
    "Normalize": _normalize,
    "CastToFloat32": _cast_to_float32,
    "Scale": _scale,
    "Clip": _clip,
    "Add": _add,
    "Multiply": _multiply,
    "LogTransform": _log_transform,
    "HashEmbeddingIndex": _hash_embedding_index,
    "CreateAttentionMask": _create_attention_mask,
    "CausalMaskGeneration": _create_causal_mask,
}

# Stochastic: transform name -> function mapping
_STOCHASTIC_TRANSFORM_FNS: dict[str, Any] = {
    "GaussianNoise": _gaussian_noise,
    "RandomBrightness": _random_brightness,
    "RandomScale": _random_scale,
    "RandomResizedCrop": _random_resized_crop,
    "RandomHorizontalFlip": _random_horizontal_flip,
    "ColorJitter": _color_jitter,
    "GaussianBlur": _gaussian_blur,
    "RandomSolarize": _random_solarize,
    "RandomGrayscale": _random_grayscale,
}

# Combined for setup() transform loop
_ALL_TRANSFORM_FNS: dict[str, Any] = {**_TRANSFORM_FNS, **_STOCHASTIC_TRANSFORM_FNS}


@register
class DataraxAdapter(PipelineAdapter):
    """PipelineAdapter implementation for Datarax.

    Supports all 25 benchmark scenarios. For standard sequential pipelines,
    chains OperatorNode wrappers for each transform in config.transforms.
    """

    def __init__(self) -> None:
        """Initialize the Datarax adapter."""
        super().__init__()
        self._pipeline: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "Datarax"

    @property
    def version(self) -> str:
        """Return the Datarax version string."""
        import datarax

        return datarax.__version__

    def is_available(self) -> bool:
        """Return True if Datarax is installed."""
        return importlib.util.find_spec("datarax") is not None

    def _create_operator(
        self,
        transform_name: str,
        rngs: nnx.Rngs,
    ) -> ElementOperator:
        """Map a transform name to a Datarax ElementOperator."""
        fn = _TRANSFORM_FNS.get(transform_name)
        if fn is not None:
            config = ElementOperatorConfig(stochastic=False)
            return ElementOperator(config, fn=fn, rngs=rngs)

        fn = _STOCHASTIC_TRANSFORM_FNS.get(transform_name)
        if fn is not None:
            config = ElementOperatorConfig(stochastic=True, stream_name="augment")
            return ElementOperator(config, fn=fn, rngs=rngs)

        raise ValueError(
            f"Unknown transform '{transform_name}'. Available: {sorted(_ALL_TRANSFORM_FNS)}"
        )

    def _create_source(
        self,
        config: ScenarioConfig,
        data: Any,
        rngs: nnx.Rngs,
    ) -> Any:
        """Create the appropriate data source based on config."""
        backend = config.extra.get("backend") if config.extra else None

        if backend == "tfds_eager":
            from datarax.sources import TFDSEagerConfig, TFDSEagerSource

            src_config = TFDSEagerConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                seed=config.seed,
                as_supervised=True,
            )
            return TFDSEagerSource(src_config, rngs=rngs)

        if backend == "tfds_streaming":
            from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource

            src_config = TFDSStreamingConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                as_supervised=True,
            )
            return TFDSStreamingSource(src_config, rngs=rngs)

        if backend == "hf_eager":
            from datarax.sources import HFEagerConfig, HFEagerSource

            src_config = HFEagerConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                seed=config.seed,
            )
            return HFEagerSource(src_config, rngs=rngs)

        if backend == "hf_streaming":
            from datarax.sources import HFStreamingConfig, HFStreamingSource

            src_config = HFStreamingConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
            )
            return HFStreamingSource(src_config, rngs=rngs)

        # Default: MemorySource
        source_config = MemorySourceConfig(shuffle=False)
        return MemorySource(config=source_config, data=data, rngs=rngs)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the Datarax pipeline for the given scenario configuration."""
        self._config = config
        rngs = nnx.Rngs(config.seed, augment=config.seed + 1)

        source = self._create_source(config, data, rngs)
        prefetch_size = int(config.extra.get("prefetch_size", 2)) if config.extra else 2
        pipeline = build_source_pipeline(
            source,
            batch_size=config.batch_size,
            prefetch_size=prefetch_size,
        )

        for transform_name in config.transforms:
            if transform_name in _ALL_TRANSFORM_FNS:
                op = self._create_operator(transform_name, rngs)
                pipeline = pipeline >> OperatorNode(op)

        self._pipeline = pipeline

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._pipeline

    def _materialize_batch(self, batch: Any) -> list[Any]:
        data = batch.get_data()  # Works with both Batch and BatchView
        block_until_ready_tree(data)
        return jax.tree.leaves(data)

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._pipeline = None
        super().teardown()

    def available_transforms(self) -> set[str]:
        """All transforms Datarax can execute (deterministic + stochastic)."""
        return set(_ALL_TRANSFORM_FNS)
