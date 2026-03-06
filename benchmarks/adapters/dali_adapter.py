"""NVIDIA DALI adapter for the benchmark framework.

Wraps NVIDIA DALI's GPU-accelerated pipeline with the PipelineAdapter
lifecycle. Tier 2 -- requires CUDA.

Design ref: Section 14.2 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


class _ExternalSourceIterator:
    """Feeds in-memory numpy data to DALI's external_source operator."""

    def __init__(self, data: np.ndarray, batch_size: int) -> None:
        self._data = data
        self._batch_size = batch_size
        self._idx = 0

    def __iter__(self) -> _ExternalSourceIterator:
        self._idx = 0
        return self

    def __next__(self) -> np.ndarray:
        if self._idx >= len(self._data):
            raise StopIteration
        end = min(self._idx + self._batch_size, len(self._data))
        batch = self._data[self._idx : end]
        self._idx = end
        return batch


@register
class DaliAdapter(PipelineAdapter):
    """PipelineAdapter for NVIDIA DALI."""

    def __init__(self) -> None:
        """Initialize the DALI adapter."""
        super().__init__()
        self._pipe: Any = None
        self._iterator: Any = None

    @property
    def name(self) -> str:
        """Return the adapter display name."""
        return "NVIDIA DALI"

    @property
    def version(self) -> str:
        """Return the DALI version string."""
        import nvidia.dali as dali

        return getattr(dali, "__version__", "unknown")

    def is_available(self) -> bool:
        """Return True if DALI is installed and CUDA is available."""
        try:
            import nvidia.dali  # noqa: F401

            # DALI requires CUDA
            import nvidia.dali.fn  # noqa: F401

            return True
        except (ImportError, RuntimeError):
            return False

    def supported_scenarios(self) -> set[str]:
        """Return the set of supported benchmark scenario IDs."""
        return {
            "CV-1",  # Normalize + CastToFloat32 on uint8 images
            "HCV-1",  # ImageNet-scale with heavy GPU transforms (native DALI)
            "NLP-1",  # No transforms (pure iteration)
            "TAB-1",  # Normalize on float32 (pass-through)
            "DIST-1",  # Normalize on float32
            "HPC-1",  # SSL 8-op chain (native DALI GPU)
            "AUG-1",  # Stochastic chain
            "AUG-2",  # Deterministic vs stochastic
            "AUG-3",  # Stochastic depth scaling
        }

    @staticmethod
    def _numpy_to_dali_dtype(np_dtype: np.dtype) -> Any:
        """Map numpy dtype to DALI DALIDataType."""
        from nvidia.dali import types

        _DTYPE_MAP = {
            np.dtype("uint8"): types.UINT8,
            np.dtype("int32"): types.INT32,
            np.dtype("int64"): types.INT64,
            np.dtype("float32"): types.FLOAT,
            np.dtype("float64"): types.FLOAT64,
        }
        return _DTYPE_MAP.get(np_dtype, types.FLOAT)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Set up the DALI pipeline for the given scenario configuration."""
        from nvidia.dali import fn, pipeline_def, types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

        primary_key = next(iter(data))
        source_data = data[primary_key]
        dali_dtype = self._numpy_to_dali_dtype(source_data.dtype)
        source_iter = _ExternalSourceIterator(source_data, config.batch_size)

        @pipeline_def(
            batch_size=config.batch_size, num_threads=max(config.num_workers, 1), device_id=0
        )
        def pipe():
            images = fn.external_source(source=source_iter, dtype=dali_dtype)
            for t_name in config.transforms:
                if t_name == "Normalize":
                    images = fn.normalize(images)
                elif t_name == "CastToFloat32":
                    images = fn.cast(images, dtype=types.FLOAT)
                elif t_name == "GaussianNoise":
                    noise = fn.random.normal(images, stddev=0.05)
                    images = images + noise
                elif t_name == "RandomBrightness":
                    images = fn.brightness(images, brightness=fn.random.uniform(range=(-0.2, 0.2)))
                elif t_name == "RandomScale":
                    images = images * fn.random.uniform(range=(0.8, 1.2))
                # --- Heavy transforms (native DALI GPU ops for HCV-1/HPC-1) ---
                elif t_name == "RandomResizedCrop":
                    images = fn.random_resized_crop(
                        images,
                        size=(224, 224),
                        random_area=(0.08, 1.0),
                        random_aspect_ratio=(3.0 / 4.0, 4.0 / 3.0),
                    )
                elif t_name == "RandomHorizontalFlip":
                    images = fn.flip(images, horizontal=fn.random.coin_flip())
                elif t_name == "ColorJitter":
                    images = fn.brightness_contrast(
                        images,
                        brightness=fn.random.uniform(range=(0.6, 1.4)),
                        contrast=fn.random.uniform(range=(0.6, 1.4)),
                    )
                    images = fn.saturation(images, saturation=fn.random.uniform(range=(0.6, 1.4)))
                elif t_name == "GaussianBlur":
                    images = fn.gaussian_blur(images, sigma=(0.1, 2.0), window_size=5)
                elif t_name == "RandomSolarize":
                    # DALI: invert pixels above threshold with 50% probability
                    should_solarize = fn.random.coin_flip(probability=0.5)
                    solarized = types.Constant(255) - images
                    mask = images >= types.Constant(128)
                    inverted = mask * solarized + (1 - mask) * images
                    images = should_solarize * inverted + (1 - should_solarize) * images
                elif t_name == "RandomGrayscale":
                    should_gray = fn.random.coin_flip(probability=0.2)
                    gray = fn.color_space_conversion(
                        images,
                        image_type=types.RGB,
                        output_type=types.GRAY,
                    )
                    gray_3ch = fn.cat(gray, gray, gray, axis=2)
                    images = should_gray * gray_3ch + (1 - should_gray) * images
            return images

        self._pipe = pipe()
        self._pipe.build()

        self._iterator = DALIGenericIterator(
            self._pipe,
            ["data"],
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        )
        self._config = config

    def warmup(self, num_batches: int = 3) -> None:
        """DALI requires an iterator reset after warmup."""
        super().warmup(num_batches)
        self._iterator.reset()

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._iterator

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        # DALI returns list of dicts; extract the primary tensor
        return [batch[0]["data"].cpu().numpy()]

    def teardown(self) -> None:
        """Release resources and reset adapter state."""
        self._iterator = None
        self._pipe = None
        super().teardown()
