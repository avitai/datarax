"""IO-1: Source Backend Comparison.

Compares data loading throughput across different source backends:
memory (MemorySource), TFDS eager/streaming, and HuggingFace eager/streaming.

The memory_source variant is Tier 1 and always available. The remaining
variants require optional dependencies (tensorflow_datasets, datasets)
and are Tier 2. External backends load MNIST (60K images, 28×28×1) via
their native APIs — the adapter creates the appropriate source object.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "IO-1"
TIER1_VARIANT: str | None = "memory_source"

# Memory source uses synthetic data
_MEM_DATASET_SIZE = 5_000
_MEM_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64
_TRANSFORMS = ["Normalize"]


# External backends load MNIST (small, fast to download)
_MNIST_DATASET_SIZE = 60_000
_MNIST_ELEMENT_SHAPE = (28, 28, 1)
_MNIST_DATASET = "mnist"
_MNIST_SPLIT = "train"


def _make_memory_source_data() -> dict:
    """Generate in-memory uint8 image data for the memory_source variant."""
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    return {"image": gen.images(_MEM_DATASET_SIZE, *_MEM_ELEMENT_SHAPE, dtype="uint8")}


def _make_empty_data() -> dict:
    """Return empty dict — external backends load their own data."""
    return {}


VARIANTS: dict[str, ScenarioVariant] = {
    "memory_source": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_MEM_DATASET_SIZE,
            element_shape=_MEM_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={"variant_name": "memory_source"},
        ),
        data_generator=_make_memory_source_data,
    ),
    "tfds_eager": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_MNIST_DATASET_SIZE,
            element_shape=_MNIST_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "tfds_eager",
                "backend": "tfds_eager",
                "dataset_name": _MNIST_DATASET,
                "split": _MNIST_SPLIT,
            },
        ),
        data_generator=_make_empty_data,
    ),
    "tfds_streaming": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_MNIST_DATASET_SIZE,
            element_shape=_MNIST_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "tfds_streaming",
                "backend": "tfds_streaming",
                "dataset_name": _MNIST_DATASET,
                "split": _MNIST_SPLIT,
            },
        ),
        data_generator=_make_empty_data,
    ),
    "hf_eager": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_MNIST_DATASET_SIZE,
            element_shape=_MNIST_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "hf_eager",
                "backend": "hf_eager",
                "dataset_name": _MNIST_DATASET,
                "split": _MNIST_SPLIT,
            },
        ),
        data_generator=_make_empty_data,
    ),
    "hf_streaming": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_MNIST_DATASET_SIZE,
            element_shape=_MNIST_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "hf_streaming",
                "backend": "hf_streaming",
                "dataset_name": _MNIST_DATASET,
                "split": _MNIST_SPLIT,
            },
        ),
        data_generator=_make_empty_data,
    ),
}


get_variant = make_get_variant(VARIANTS)
