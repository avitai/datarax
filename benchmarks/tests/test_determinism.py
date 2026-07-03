"""Determinism and cross-adapter fairness contracts.

The comparative benchmarks assume every adapter measures the same workload:
identical raw bytes in, and (for transform-free configs) identical batches
out. These tests pin that assumption — a shuffling or resampling adapter
would silently benchmark a different data order than its competitors.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
import pytest

from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter


_DATASET_SIZE = 256
_ELEMENT_SHAPE = (8, 8, 3)
_BATCH_SIZE = 32


def _plain_config() -> ScenarioConfig:
    """Transform-free CV config: output batches must mirror the input bytes."""
    return ScenarioConfig(
        scenario_id="CV-1",
        dataset_size=_DATASET_SIZE,
        element_shape=_ELEMENT_SHAPE,
        batch_size=_BATCH_SIZE,
        transforms=[],
        extra={"variant_name": "determinism"},
    )


def _image_data() -> dict[str, np.ndarray]:
    """Deterministic uint8 image data shared by every adapter under test."""
    rng = np.random.default_rng(42)
    return {"image": rng.integers(0, 256, (_DATASET_SIZE, *_ELEMENT_SHAPE), dtype=np.uint8)}


def _first_batch(adapter: PipelineAdapter, config: ScenarioConfig, data: dict) -> np.ndarray:
    """Set up the adapter, pull one batch, and return its first array field."""
    adapter.setup(config, data)
    try:
        batch = next(iter(adapter._iterate_batches()))
        arrays = adapter._materialize_batch(batch)
        return np.asarray(arrays[0])
    finally:
        adapter.teardown()


class TestSameSeedReproducibility:
    """Same config + same data -> byte-identical batches across setups."""

    def test_datarax_first_batch_reproducible(self):
        """Two independent datarax setups yield identical first batches."""
        config = _plain_config()
        data = _image_data()
        first = _first_batch(DataraxAdapter(), config, data)
        second = _first_batch(DataraxAdapter(), config, data)
        np.testing.assert_array_equal(first, second)

    def test_datarax_epoch_is_reproducible(self):
        """A full epoch replays identically after a fresh setup."""
        config = _plain_config()
        data = _image_data()

        def epoch(adapter: DataraxAdapter) -> list[bytes]:
            adapter.setup(config, data)
            try:
                batches = []
                for index, batch in enumerate(adapter._iterate_batches()):
                    if index >= _DATASET_SIZE // _BATCH_SIZE:
                        break
                    batches.append(np.asarray(adapter._materialize_batch(batch)[0]).tobytes())
                return batches
            finally:
                adapter.teardown()

        np.testing.assert_array_equal(epoch(DataraxAdapter()), epoch(DataraxAdapter()))


# Adapters that wrap raw numpy in-memory data and are installed locally.
# (module path, class name, framework import required)
_COMPARABLE_ADAPTERS = [
    ("benchmarks.adapters.datarax_adapter", "DataraxAdapter", None),
    ("benchmarks.adapters.grain_adapter", "GrainAdapter", "grain"),
    ("benchmarks.adapters.pytorch_dl_adapter", "PyTorchDataLoaderAdapter", "torch"),
    ("benchmarks.adapters.spdl_adapter", "SpdlAdapter", "spdl"),
]


def _load_adapter(module_path: str, class_name: str, framework: str | None) -> PipelineAdapter:
    """Instantiate an adapter, skipping when its framework is missing."""
    if framework is not None:
        pytest.importorskip(framework)
    module = importlib.import_module(module_path)
    factory: Callable[[], PipelineAdapter] = getattr(module, class_name)
    return factory()


class TestCrossAdapterFairness:
    """Every adapter must serve the source bytes in source order."""

    @pytest.mark.parametrize(
        ("module_path", "class_name", "framework"),
        _COMPARABLE_ADAPTERS,
        ids=[entry[1] for entry in _COMPARABLE_ADAPTERS],
    )
    def test_first_batch_is_source_prefix(
        self, module_path: str, class_name: str, framework: str | None
    ):
        """With no transforms, the first batch equals the first source rows."""
        adapter = _load_adapter(module_path, class_name, framework)
        data = _image_data()
        batch = _first_batch(adapter, _plain_config(), data)
        expected = data["image"][:_BATCH_SIZE]
        np.testing.assert_array_equal(
            np.asarray(batch, dtype=np.uint8).reshape(expected.shape), expected
        )


class TestTransformFidelity:
    """Adapters must run every requested transform or refuse the scenario.

    Silently skipping unimplemented transforms would benchmark a lighter
    pipeline than requested — unfair to adapters that run the full chain.
    """

    @pytest.mark.parametrize(
        ("module_path", "class_name", "framework"),
        _COMPARABLE_ADAPTERS,
        ids=[entry[1] for entry in _COMPARABLE_ADAPTERS],
    )
    def test_setup_rejects_unknown_transform(
        self, module_path: str, class_name: str, framework: str | None
    ):
        """setup() with a transform the adapter lacks raises, never skips."""
        adapter = _load_adapter(module_path, class_name, framework)
        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["NotARealTransform"],
            extra={"variant_name": "fidelity"},
        )
        with pytest.raises(ValueError, match="NotARealTransform"):
            adapter.setup(config, _image_data())
