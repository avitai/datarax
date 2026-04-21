"""tf.data adapter for the benchmark framework.

Wraps TensorFlow's tf.data.Dataset with the PipelineAdapter lifecycle.
Tier 1 -- uses AUTOTUNE for idiomatic best practice.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


# ---------------------------------------------------------------------------
# Transform mapping: ScenarioConfig.transforms -> tf.data.Dataset.map()
# ---------------------------------------------------------------------------


def _tf_normalize(x: Any) -> Any:
    """Normalize uint8 tensors to float32."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {
            k: tf.cast(v, tf.float32) / 255.0 if v.dtype == tf.uint8 else v for k, v in x.items()
        }
    return tf.cast(x, tf.float32) / 255.0 if x.dtype == tf.uint8 else x


def _tf_cast_to_float32(x: Any) -> Any:
    """Cast tensors to float32."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {k: tf.cast(v, tf.float32) for k, v in x.items()}
    return tf.cast(x, tf.float32)


def _tf_gaussian_noise(x: Any) -> Any:
    """Add Gaussian noise with native TensorFlow ops."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {
            k: v + tf.random.normal(tf.shape(v), stddev=0.05, dtype=v.dtype) for k, v in x.items()
        }
    return x + tf.random.normal(tf.shape(x), stddev=0.05, dtype=x.dtype)


def _tf_random_brightness(x: Any) -> Any:
    """Apply random brightness."""
    import tensorflow as tf

    delta = tf.random.uniform([], minval=-0.2, maxval=0.2)  # type: ignore[reportArgumentType]
    if isinstance(x, dict):
        return {k: v + delta for k, v in x.items()}
    return x + delta


def _tf_random_scale(x: Any) -> Any:
    """Apply random multiplicative scale."""
    import tensorflow as tf

    factor = tf.random.uniform([], minval=0.8, maxval=1.2)  # type: ignore[reportArgumentType]
    if isinstance(x, dict):
        return {k: v * factor for k, v in x.items()}
    return x * factor


def _tf_random_resized_crop(x: Any) -> Any:
    """Random crop + resize using tf.image."""
    import tensorflow as tf

    def crop_single(img: Any) -> Any:
        bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(img),
            bounding_boxes=tf.zeros([1, 0, 4]),
            min_object_covered=0.0,
            area_range=(0.08, 1.0),
            aspect_ratio_range=(3.0 / 4.0, 4.0 / 3.0),
            use_image_if_no_bounding_boxes=True,
        )
        offset_y, offset_x, _ = tf.unstack(bbox[0])  # type: ignore[reportIndexIssue]
        target_h, target_w, _ = tf.unstack(bbox[1])  # type: ignore[reportIndexIssue]
        cropped = tf.slice(img, [offset_y, offset_x, 0], [target_h, target_w, -1])
        return tf.image.resize(cropped, [224, 224])

    if isinstance(x, dict):
        return {k: crop_single(v) if len(v.shape) == 3 else v for k, v in x.items()}
    return crop_single(x)


def _tf_random_horizontal_flip(x: Any) -> Any:
    """Randomly flip image tensors left-to-right."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {
            k: tf.image.random_flip_left_right(v) if len(v.shape) == 3 else v for k, v in x.items()
        }
    return tf.image.random_flip_left_right(x)


def _tf_color_jitter(x: Any) -> Any:
    """Brightness + contrast + saturation jitter using tf.image."""
    import tensorflow as tf

    def jitter(img: Any) -> Any:
        img = tf.image.random_brightness(img, max_delta=0.4)
        img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
        return tf.image.random_saturation(img, lower=0.6, upper=1.4)

    if isinstance(x, dict):
        return {k: jitter(v) if len(v.shape) == 3 else v for k, v in x.items()}
    return jitter(x)


def _tf_gaussian_blur(x: Any) -> Any:
    """Gaussian blur via depthwise convolution."""
    import tensorflow as tf

    def blur(img: Any) -> Any:
        kernel_1d = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32)
        kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
        kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
        kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]  # type: ignore[reportIndexIssue]
        channels = tf.shape(img)[-1]  # type: ignore[reportIndexIssue]
        kernel = tf.tile(kernel_2d, [1, 1, channels, 1])
        img_4d = img[tf.newaxis]
        blurred = tf.nn.depthwise_conv2d(
            tf.cast(img_4d, tf.float32),
            kernel,
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        return tf.cast(blurred[0], img.dtype)

    if isinstance(x, dict):
        return {k: blur(v) if len(v.shape) == 3 else v for k, v in x.items()}
    return blur(x)


def _tf_random_solarize(x: Any) -> Any:
    """Invert pixels above threshold with 50% probability."""
    import tensorflow as tf

    def solarize(img: Any) -> Any:
        should_solarize = tf.random.uniform([]) < 0.5
        return tf.cond(
            should_solarize,
            lambda: tf.where(img >= 128, 255 - img, img),
            lambda: img,
        )

    if isinstance(x, dict):
        return {k: solarize(v) if len(v.shape) == 3 else v for k, v in x.items()}
    return solarize(x)


def _tf_random_grayscale(x: Any) -> Any:
    """Convert to grayscale with 20% probability."""
    import tensorflow as tf

    def grayscale(img: Any) -> Any:
        should_gray = tf.random.uniform([]) < 0.2
        gray = tf.image.rgb_to_grayscale(img)
        gray_3ch = tf.tile(gray, [1, 1, 3])
        return tf.cond(should_gray, lambda: gray_3ch, lambda: img)

    if isinstance(x, dict):
        return {
            k: grayscale(v) if len(v.shape) == 3 and v.shape[-1] == 3 else v for k, v in x.items()
        }
    return grayscale(x)


def _tf_log_transform(x: Any) -> Any:
    """Element-wise log1p for dense features."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {k: tf.math.log1p(tf.abs(tf.cast(v, tf.float32))) for k, v in x.items()}
    return tf.math.log1p(tf.abs(tf.cast(x, tf.float32)))


def _tf_create_attention_mask(x: Any) -> Any:
    """Binary attention mask: 1 for real tokens, 0 for padding."""
    import tensorflow as tf

    if isinstance(x, dict):
        return {k: tf.cast(tf.not_equal(v, 0), tf.float32) for k, v in x.items()}
    return tf.cast(tf.not_equal(x, 0), tf.float32)


def _causal_mask_for_tensor(x: Any) -> Any:
    """Build a lower-triangular causal mask for one tensor."""
    import tensorflow as tf

    seq_len = tf.shape(x)[-1]  # type: ignore[reportIndexIssue]
    return tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)


def _tf_create_causal_mask(x: Any) -> Any:
    """Lower-triangular causal attention mask."""
    if isinstance(x, dict):
        return {k: _causal_mask_for_tensor(v) for k, v in x.items()}
    return _causal_mask_for_tensor(x)


_TF_TRANSFORMS = {
    "Normalize": _tf_normalize,
    "CastToFloat32": _tf_cast_to_float32,
    "GaussianNoise": _tf_gaussian_noise,
    "RandomBrightness": _tf_random_brightness,
    "RandomScale": _tf_random_scale,
    "RandomResizedCrop": _tf_random_resized_crop,
    "RandomHorizontalFlip": _tf_random_horizontal_flip,
    "ColorJitter": _tf_color_jitter,
    "GaussianBlur": _tf_gaussian_blur,
    "RandomSolarize": _tf_random_solarize,
    "RandomGrayscale": _tf_random_grayscale,
    "LogTransform": _tf_log_transform,
    "CreateAttentionMask": _tf_create_attention_mask,
    "CausalMaskGeneration": _tf_create_causal_mask,
}


def _get_tf_transform(name: str) -> Any:
    """Map a transform name to a TF-compatible map function."""
    return _TF_TRANSFORMS.get(name)


@register
class TfDataAdapter(PipelineAdapter):
    """PipelineAdapter for tf.data."""

    def __init__(self) -> None:
        """Initialize the tf.data adapter."""
        super().__init__()
        self._dataset: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "tf.data"

    @property
    def version(self) -> str:
        """Return the TensorFlow version string."""
        import tensorflow as tf

        return tf.__version__

    def is_available(self) -> bool:
        """Return True if TensorFlow is installed."""
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False

    def available_transforms(self) -> set[str]:
        """All transforms tf.data can execute via _get_tf_transform."""
        return {
            "Normalize",
            "CastToFloat32",
            "GaussianNoise",
            "RandomBrightness",
            "RandomScale",
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "ColorJitter",
            "GaussianBlur",
            "RandomSolarize",
            "RandomGrayscale",
            "LogTransform",
            "CreateAttentionMask",
            "CausalMaskGeneration",
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the tf.data pipeline for the given scenario configuration."""
        import tensorflow as tf

        ds = tf.data.Dataset.from_tensor_slices(data)

        for transform_name in config.transforms:
            fn = _get_tf_transform(transform_name)
            if fn is not None:
                ds = ds.map(fn, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(config.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self._dataset = ds
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._dataset

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        if isinstance(batch, dict):
            return [np.asarray(v) for v in batch.values()]
        return [np.asarray(batch)]

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._dataset = None
        super().teardown()
