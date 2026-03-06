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


def _get_tf_transform(name: str) -> Any:
    """Map a transform name to a TF-compatible map function."""
    import tensorflow as tf

    def _normalize(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: tf.cast(v, tf.float32) / 255.0 if v.dtype == tf.uint8 else v
                for k, v in x.items()
            }
        return tf.cast(x, tf.float32) / 255.0 if x.dtype == tf.uint8 else x

    def _cast_to_float32(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: tf.cast(v, tf.float32) for k, v in x.items()}
        return tf.cast(x, tf.float32)

    def _gaussian_noise(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: v + tf.random.normal(tf.shape(v), stddev=0.05, dtype=v.dtype)
                for k, v in x.items()
            }
        return x + tf.random.normal(tf.shape(x), stddev=0.05, dtype=x.dtype)

    def _random_brightness(x: Any) -> Any:
        delta = tf.random.uniform([], minval=-0.2, maxval=0.2)  # type: ignore[reportArgumentType]
        if isinstance(x, dict):
            return {k: v + delta for k, v in x.items()}
        return x + delta

    def _random_scale(x: Any) -> Any:
        factor = tf.random.uniform([], minval=0.8, maxval=1.2)  # type: ignore[reportArgumentType]
        if isinstance(x, dict):
            return {k: v * factor for k, v in x.items()}
        return x * factor

    # --- Heavy transforms using native tf.image ops (best practice) ---

    def _random_resized_crop(x: Any) -> Any:
        """Random crop + resize using tf.image (native TF, graph-optimized)."""

        def _crop_single(img: Any) -> Any:
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
            return {k: _crop_single(v) if len(v.shape) == 3 else v for k, v in x.items()}
        return _crop_single(x)

    def _random_horizontal_flip(x: Any) -> Any:
        if isinstance(x, dict):
            return {
                k: tf.image.random_flip_left_right(v) if len(v.shape) == 3 else v
                for k, v in x.items()
            }
        return tf.image.random_flip_left_right(x)

    def _color_jitter(x: Any) -> Any:
        """Brightness + contrast + saturation jitter using tf.image."""

        def _jitter(img: Any) -> Any:
            img = tf.image.random_brightness(img, max_delta=0.4)
            img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
            img = tf.image.random_saturation(img, lower=0.6, upper=1.4)
            return img

        if isinstance(x, dict):
            return {k: _jitter(v) if len(v.shape) == 3 else v for k, v in x.items()}
        return _jitter(x)

    def _gaussian_blur_tf(x: Any) -> Any:
        """Gaussian blur via depthwise convolution (native TF)."""

        def _blur(img: Any) -> Any:
            # 5x5 Gaussian kernel
            kernel_1d = tf.constant([1, 4, 6, 4, 1], dtype=tf.float32)
            kernel_1d = kernel_1d / tf.reduce_sum(kernel_1d)
            kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)
            kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]  # type: ignore[reportIndexIssue]
            channels = tf.shape(img)[-1]  # type: ignore[reportIndexIssue]
            kernel = tf.tile(kernel_2d, [1, 1, channels, 1])
            img_4d = img[tf.newaxis]
            blurred = tf.nn.depthwise_conv2d(
                tf.cast(img_4d, tf.float32), kernel, strides=[1, 1, 1, 1], padding="SAME"
            )
            return tf.cast(blurred[0], img.dtype)

        if isinstance(x, dict):
            return {k: _blur(v) if len(v.shape) == 3 else v for k, v in x.items()}
        return _blur(x)

    def _random_solarize_tf(x: Any) -> Any:
        """Invert pixels above threshold with 50% probability."""

        def _solarize(img: Any) -> Any:
            should_solarize = tf.random.uniform([]) < 0.5
            return tf.cond(
                should_solarize,
                lambda: tf.where(img >= 128, 255 - img, img),
                lambda: img,
            )

        if isinstance(x, dict):
            return {k: _solarize(v) if len(v.shape) == 3 else v for k, v in x.items()}
        return _solarize(x)

    def _random_grayscale_tf(x: Any) -> Any:
        """Convert to grayscale with 20% probability."""

        def _grayscale(img: Any) -> Any:
            should_gray = tf.random.uniform([]) < 0.2
            gray = tf.image.rgb_to_grayscale(img)
            gray_3ch = tf.tile(gray, [1, 1, 3])
            return tf.cond(should_gray, lambda: gray_3ch, lambda: img)

        if isinstance(x, dict):
            return {
                k: _grayscale(v) if len(v.shape) == 3 and v.shape[-1] == 3 else v
                for k, v in x.items()
            }
        return _grayscale(x)

    def _log_transform_tf(x: Any) -> Any:
        """Element-wise log1p for dense features."""
        if isinstance(x, dict):
            return {k: tf.math.log1p(tf.abs(tf.cast(v, tf.float32))) for k, v in x.items()}
        return tf.math.log1p(tf.abs(tf.cast(x, tf.float32)))

    def _create_attention_mask_tf(x: Any) -> Any:
        """Binary attention mask: 1 for real tokens, 0 for padding."""
        if isinstance(x, dict):
            return {k: tf.cast(tf.not_equal(v, 0), tf.float32) for k, v in x.items()}
        return tf.cast(tf.not_equal(x, 0), tf.float32)

    def _create_causal_mask_tf(x: Any) -> Any:
        """Lower-triangular causal attention mask."""
        if isinstance(x, dict):
            return {
                k: tf.linalg.band_part(
                    tf.ones((tf.shape(v)[-1], tf.shape(v)[-1]), dtype=tf.float32),  # type: ignore[reportIndexIssue]
                    -1,
                    0,
                )
                for k, v in x.items()
            }
        seq_len = tf.shape(x)[-1]  # type: ignore[reportIndexIssue]
        return tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)

    _TRANSFORMS = {
        "Normalize": _normalize,
        "CastToFloat32": _cast_to_float32,
        "GaussianNoise": _gaussian_noise,
        "RandomBrightness": _random_brightness,
        "RandomScale": _random_scale,
        "RandomResizedCrop": _random_resized_crop,
        "RandomHorizontalFlip": _random_horizontal_flip,
        "ColorJitter": _color_jitter,
        "GaussianBlur": _gaussian_blur_tf,
        "RandomSolarize": _random_solarize_tf,
        "RandomGrayscale": _random_grayscale_tf,
        "LogTransform": _log_transform_tf,
        "CreateAttentionMask": _create_attention_mask_tf,
        "CausalMaskGeneration": _create_causal_mask_tf,
    }

    return _TRANSFORMS.get(name)


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
