"""tf.data adapter for the benchmark framework.

Wraps TensorFlow's tf.data.Dataset with the BenchmarkAdapter lifecycle.
Tier 1 -- uses AUTOTUNE for idiomatic best practice.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig


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
        delta = tf.random.uniform([], minval=-0.2, maxval=0.2)
        if isinstance(x, dict):
            return {k: v + delta for k, v in x.items()}
        return x + delta

    def _random_scale(x: Any) -> Any:
        factor = tf.random.uniform([], minval=0.8, maxval=1.2)
        if isinstance(x, dict):
            return {k: v * factor for k, v in x.items()}
        return x * factor

    _TRANSFORMS = {
        "Normalize": _normalize,
        "CastToFloat32": _cast_to_float32,
        "GaussianNoise": _gaussian_noise,
        "RandomBrightness": _random_brightness,
        "RandomScale": _random_scale,
    }

    return _TRANSFORMS.get(name)


@register
class TfDataAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for tf.data."""

    def __init__(self) -> None:
        self._dataset: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "tf.data"

    @property
    def version(self) -> str:
        import tensorflow as tf

        return tf.__version__

    def is_available(self) -> bool:
        try:
            import tensorflow  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "CV-1",  # Normalize + CastToFloat32
            "NLP-1",  # No transforms
            "TAB-1",  # Normalize on float32
            "DIST-1",  # Normalize on float32
            "PR-1",  # Normalize on float32
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
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
        self._dataset = None
        self._config = None
