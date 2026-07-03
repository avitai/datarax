"""Datarax adapter -- reference implementation for the benchmark framework.

Wraps Datarax's public API (Pipeline, MemorySource)
using the PipelineAdapter lifecycle: setup -> warmup -> iterate -> teardown.

Supports all 25 scenarios with transform chaining via
Pipeline composition for sequential and DAG scenarios. External
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
from benchmarks.adapters.base import Capability, PipelineAdapter, ScenarioConfig
from datarax import Pipeline
from datarax.core.config import BatchMixOperatorConfig, ElementOperatorConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.operators.element_operator import ElementOperator
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig
from datarax.performance.synchronization import block_until_ready_tree
from datarax.pipeline.nodes import CachingIterator, RebatchNode
from datarax.sources import (
    MemorySource,
    MemorySourceConfig,
    MixDataSourcesConfig,
    MixDataSourcesNode,
)


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
        c = x.shape[2]

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


def _random_crop(element: Element, key: jax.Array) -> Element:
    """Random fixed-size crop (87.5% area) then pad back to original shape. Stochastic.

    Crop size is static (JIT-safe ``dynamic_slice``); the padded-back result keeps
    the original shape so batches stay uniform. Used by PR-2, NNX-1, XFMR-1.
    """

    def crop(x: jax.Array) -> jax.Array:
        h, w, c = x.shape[0], x.shape[1], x.shape[2]
        crop_h, crop_w = (h * 7) // 8, (w * 7) // 8
        top = jax.random.randint(key, (), 0, jnp.maximum(h - crop_h, 1))
        left = jax.random.randint(key, (), 0, jnp.maximum(w - crop_w, 1))
        cropped = jax.lax.dynamic_slice(x, (top, left, 0), (crop_h, crop_w, c))
        # Pad back to the original shape so the batch dimension stays uniform.
        return jnp.pad(cropped, ((0, h - crop_h), (0, w - crop_w), (0, 0)))

    return element.replace(data=jax.tree.map(crop, element.data))


def _affine_rotation(element: Element, key: jax.Array) -> Element:
    """Random 90-degree rotation (k in 0..3). Stochastic, JIT-safe via lax.switch.

    A shape-preserving affine transform used by XFMR-1 to exercise the JIT+vmap path.
    """
    k = jax.random.randint(key, (), 0, 4)

    def rotate(x: jax.Array) -> jax.Array:
        branches = [lambda a=a: jnp.rot90(x, k=a, axes=(0, 1)) for a in range(4)]
        # rot90 by 1 or 3 transposes H/W; guard to square crops keeps shape uniform.
        return jax.lax.switch(k, branches)

    return element.replace(data=jax.tree.map(rotate, element.data))


def _multi_scale_resize(element: Element, key: jax.Array) -> Element:
    """Downsample to half resolution then bilinear-resize back. Deterministic, compute-heavy.

    Simulates the multi-resolution pipeline (CV-4): the round trip through a smaller
    scale keeps the output shape while incurring real resize FLOPs.
    """
    del key

    def resize_roundtrip(x: jax.Array) -> jax.Array:
        # x.shape entries are static ints; keep the resize target static (Python max),
        # not a traced jnp.maximum, or jax.image.resize rejects it under jit.
        h, w, c = x.shape[0], x.shape[1], x.shape[2]
        small = jax.image.resize(x, (max(h // 2, 1), max(w // 2, 1), c), "bilinear")
        return jax.image.resize(small, (h, w, c), "bilinear")

    return element.replace(data=jax.tree.map(resize_roundtrip, element.data))


def _dynamic_pad(element: Element, key: jax.Array) -> Element:
    """Pad each 1-D sequence up to the next multiple of 128. Deterministic.

    Models NLP-2's dynamic padding with a static pad width (JIT-safe): the padded
    length is a static function of the input length, so shapes stay traceable.
    """
    del key

    def pad_seq(x: jax.Array) -> jax.Array:
        seq_len = x.shape[-1]
        target = ((seq_len + 127) // 128) * 128
        if target == seq_len:
            return x
        pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target - seq_len)]
        return jnp.pad(x, pad_width)

    return element.replace(data=jax.tree.map(pad_seq, element.data))


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


def _expensive_transform(element: Element, key: jax.Array) -> Element:
    """Compute-heavy deterministic transform (IO-4 cache benchmark).

    A chain of transcendental ops, expensive relative to the cheap transform, so
    caching its output produces a measurable epoch-2 speedup.
    """
    del key

    def expensive(x: jax.Array) -> jax.Array:
        for _ in range(8):
            x = jnp.sin(x) + jnp.cos(x)
        return x

    return element.replace(data=jax.tree.map(expensive, element.data))


def _cheap_transform(element: Element, key: jax.Array) -> Element:
    """Light deterministic transform run after the cache (IO-4)."""
    del key
    return element.replace(data=jax.tree.map(lambda x: x + 0.1, element.data))


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
    "MultiScaleResize": _multi_scale_resize,
    "DynamicPad": _dynamic_pad,
    "ExpensiveTransform": _expensive_transform,
    "CheapTransform": _cheap_transform,
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
    "RandomCrop": _random_crop,
    "AffineRotation": _affine_rotation,
    # RandomFlip is the generic alias scenarios use for horizontal flipping.
    "RandomFlip": _random_horizontal_flip,
}

# Combined for setup() transform loop
_ALL_TRANSFORM_FNS: dict[str, Any] = {**_TRANSFORM_FNS, **_STOCHASTIC_TRANSFORM_FNS}


class _LearnableAugmentOperator(nnx.Module):
    """Differentiable, mode-aware augment stage for the PC-5 benchmark.

    Applies a per-channel learnable affine (``scale``, ``bias`` are ``nnx.Param``,
    so ``jax.grad`` flows end-to-end through the pipeline) followed by dropout.
    ``nnx.Dropout`` honours ``nnx.Module.train()``/``eval()`` — the jitter is active
    while training and becomes identity in eval — so the whole stage is both
    learnable and train/eval aware. Consumes the raw batch dict the Pipeline
    threads between stages.
    """

    def __init__(self, num_channels: int, *, rngs: nnx.Rngs) -> None:
        """Initialize learnable scale/bias params and a train/eval-aware dropout."""
        self.scale = nnx.Param(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=jnp.float32))
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self._num_channels = num_channels

    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Apply the learnable affine (where channels match) then dropout."""

        def apply(x: jax.Array) -> jax.Array:
            if x.ndim >= 1 and x.shape[-1] == self._num_channels:
                x = x * self.scale.value + self.bias.value
            return self.dropout(x)

        return jax.tree.map(apply, batch)


class _MergeModule(nnx.Module):
    """DAG merge node: averages two branch outputs element-wise (PC-2).

    Receives its two predecessors' batch dicts as positional args (the
    ``Pipeline.from_dag`` multi-input path) and returns their mean.
    """

    def __call__(self, a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """Average the two branch outputs leaf-wise."""
        return jax.tree.map(lambda x, y: (x + y) * 0.5, a, b)


@register
class DataraxAdapter(PipelineAdapter):
    """PipelineAdapter implementation for Datarax.

    Supports all 25 benchmark scenarios. For standard sequential pipelines,
    composes stages from each transform in config.transforms.
    """

    def __init__(self) -> None:
        """Initialize the Datarax adapter."""
        super().__init__()
        self._pipeline: Any = None
        self._cached_iter: CachingIterator[Any] | None = None
        self._buffer_depth: int = 2

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

    def _create_mixed_source(self, data: Any, rngs: nnx.Rngs) -> Any:
        """Build a MixDataSourcesNode that mixes two halves of the data 50/50 (IO-3).

        Splits each field down the leading (record) axis into two MemorySources and
        mixes them with equal weight, exercising the weighted multi-source path.
        """
        first = {key: value[: len(value) // 2] for key, value in data.items()}
        second = {key: value[len(value) // 2 :] for key, value in data.items()}
        source_config = MemorySourceConfig(shuffle=False)
        sub_sources: list[DataSourceModule] = [
            MemorySource(config=source_config, data=first, rngs=rngs),
            MemorySource(config=source_config, data=second, rngs=rngs),
        ]
        mix_config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        return MixDataSourcesNode(mix_config, sub_sources, rngs=rngs)

    def _create_source(
        self,
        config: ScenarioConfig,
        data: Any,
        rngs: nnx.Rngs,
    ) -> Any:
        """Create the appropriate data source based on config."""
        if Capability.MIXED_SOURCE in set(config.required_capabilities):
            return self._create_mixed_source(data, rngs)

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

    def _append_capability_operators(
        self, stages: list[Any], config: ScenarioConfig, rngs: nnx.Rngs
    ) -> None:
        """Append the structural operators a scenario's capabilities require.

        Element transforms already populate ``stages`` in order; structural
        operators (batch mixing, ...) run after them, parameterised from
        ``config.extra``. Extend this dispatch as more capabilities are wired.
        """
        caps = set(config.required_capabilities)
        if Capability.BATCH_MIXING in caps:
            mix_config = BatchMixOperatorConfig(
                mode=config.extra.get("mix_mode", "mixup"),
                alpha=config.extra.get("mix_alpha", 0.4),
            )
            stages.append(BatchMixOperator(mix_config, rngs=rngs))

        if Capability.PROBABILISTIC in caps:
            probability = float(config.extra.get("probability", 0.5))
            # ProbabilisticAugment: apply a stochastic augment with probability p.
            augment = ElementOperator(
                ElementOperatorConfig(stochastic=True, stream_name="augment"),
                fn=_random_brightness,
                rngs=rngs,
            )
            stages.append(
                ProbabilisticOperator(
                    ProbabilisticOperatorConfig(operator=augment, probability=probability),
                    rngs=rngs,
                )
            )
            # ConditionalSelect: choose between candidate augments per record.
            choices: list[OperatorModule] = [
                ElementOperator(
                    ElementOperatorConfig(stochastic=True, stream_name="augment"),
                    fn=fn,
                    rngs=rngs,
                )
                for fn in (_random_brightness, _gaussian_noise)
            ]
            stages.append(
                SelectorOperator(
                    SelectorOperatorConfig(operators=choices, stream_name="augment"),
                    rngs=rngs,
                )
            )

        if Capability.LEARNABLE_TRANSFORM in caps:
            num_channels = config.element_shape[-1] if config.element_shape else 1
            stages.append(_LearnableAugmentOperator(num_channels, rngs=rngs))

        if Capability.REBATCHING in caps:
            # Differentiable in-DAG rebatch from batch_size down to target_batch_size.
            target = int(config.extra.get("target_batch_size", config.batch_size))
            group_size = max(1, config.batch_size // target)
            stages.append(RebatchNode(group_size))

    def _build_branching_pipeline(self, source: Any, config: ScenarioConfig, rngs: nnx.Rngs) -> Any:
        """Build a two-branch parallel DAG that merges by averaging (PC-2).

        ``branch_a`` and ``branch_b`` both consume the source batch; ``merge``
        averages their outputs. Exercises ``Pipeline.from_dag`` topology.
        """
        branch_a = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="augment"),
            fn=_random_brightness,
            rngs=rngs,
        )
        branch_b = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="augment"),
            fn=_gaussian_noise,
            rngs=rngs,
        )
        return Pipeline.from_dag(
            source=source,
            nodes={"branch_a": branch_a, "branch_b": branch_b, "merge": _MergeModule()},
            edges={"branch_a": [], "branch_b": [], "merge": ["branch_a", "branch_b"]},
            sink="merge",
            batch_size=config.batch_size,
            rngs=nnx.Rngs(config.seed),
        )

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the Datarax pipeline for the given scenario configuration."""
        self._config = config
        self._cached_iter = None
        self._buffer_depth = int(config.extra.get("prefetch_size", 2)) if config.extra else 2
        rngs = nnx.Rngs(config.seed, augment=config.seed + 1, batch_mix=config.seed + 2)

        source = self._create_source(config, data, rngs)

        if Capability.DAG_BRANCHING in set(config.required_capabilities):
            self._pipeline = self._build_branching_pipeline(source, config, rngs)
            return

        missing = [name for name in config.transforms if name not in _ALL_TRANSFORM_FNS]
        if missing:
            raise ValueError(
                f"{self.name} does not implement transforms {missing}; "
                f"the scenario should be reported as unsupported instead."
            )
        stages: list[Any] = [self._create_operator(name, rngs) for name in config.transforms]
        self._append_capability_operators(stages, config, rngs)

        self._pipeline = Pipeline(
            source=source,
            stages=stages,
            batch_size=config.batch_size,
            rngs=nnx.Rngs(config.seed),
        )

        if Capability.CACHING in set(config.required_capabilities):
            # Iteration-boundary cache: the expensive pipeline runs once, later
            # passes replay cached batches (see CachingIterator).
            self._cached_iter = CachingIterator(iter(self._pipeline))

    def _iterate_batches(self) -> Iterator[Any]:
        # The Pipeline yields device-resident JAX batches, so no separate host->device
        # prefetch stage is needed (unlike host-loader frameworks). ``_buffer_depth``
        # records the requested prefetch policy for parity/metadata; the pipeline keeps
        # data on device inherently.
        if self._cached_iter is not None:
            yield from iter(self._cached_iter)
        else:
            yield from self._pipeline

    def _materialize_batch(self, batch: Any) -> list[Any]:
        # Pipeline yields plain dicts; legacy Batch/BatchView yields a wrapper.
        data = batch.get_data() if hasattr(batch, "get_data") else batch
        block_until_ready_tree(data)
        return jax.tree.leaves(data)

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._pipeline = None
        self._cached_iter = None
        super().teardown()

    def available_transforms(self) -> set[str]:
        """All transforms Datarax can execute (deterministic + stochastic)."""
        return set(_ALL_TRANSFORM_FNS)

    def available_capabilities(self) -> set[str]:
        """Structural pipeline features Datarax can execute (see Capability)."""
        return {
            Capability.BATCH_MIXING,
            Capability.PROBABILISTIC,
            Capability.LEARNABLE_TRANSFORM,
            Capability.DAG_BRANCHING,
            Capability.MIXED_SOURCE,
            Capability.REBATCHING,
            Capability.CACHING,
        }
