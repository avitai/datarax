"""P2: GPU augmentation parity target tests.

Target: Datarax within 2x of DALI for vision transforms on GPU.

Note: These tests require nvidia.dali and GPU hardware.
They are automatically skipped on CPU-only systems.
"""

import pytest
import jax
import jax.numpy as jnp

from datarax.core.element_batch import Batch


# Skip entire module if no GPU available
def _has_gpu():
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="GPU not available")


@pytest.mark.benchmark
@pytest.mark.gpu
class TestP2GPUAugmentation:
    """P2: Datarax vision transforms within 2x of DALI on GPU."""

    def _make_gpu_batch(self, batch_size: int = 64, shape: tuple = (224, 224, 3)) -> Batch:
        """Create a batch of random images on GPU."""
        data = jnp.ones((batch_size, *shape), dtype=jnp.float32)
        gpu = jax.devices("gpu")[0]
        data = jax.device_put(data, gpu)
        return Batch.from_parts(data={"image": data}, states={}, validate=False)

    def test_normalize_on_gpu(self):
        """Verify normalize transform runs on GPU without host roundtrip."""
        from datarax.operators.modality.image.functional import normalize

        batch = self._make_gpu_batch(64, (32, 32, 3))
        images = batch.data["image"]

        @jax.jit
        def apply_normalize(x):
            return normalize(
                x, mean=jnp.array([0.485, 0.456, 0.406]), std=jnp.array([0.229, 0.224, 0.225])
            )

        result = apply_normalize(images)
        assert result.shape == images.shape
        # Result should stay on GPU
        assert result.devices() == images.devices()

    def test_cast_to_float32_on_gpu(self):
        """Verify uint8→float32 cast runs efficiently on GPU."""
        gpu = jax.devices("gpu")[0]
        data = jax.device_put(jnp.ones((64, 32, 32, 3), dtype=jnp.uint8), gpu)

        @jax.jit
        def cast_and_scale(x):
            return x.astype(jnp.float32) / 255.0

        result = cast_and_scale(data)
        assert result.dtype == jnp.float32
        assert result.devices() == data.devices()

    def test_comparable_throughput_vs_dali(self, cv1_large_image_data):
        """Compare Datarax GPU transform throughput against DALI.

        This test requires nvidia.dali and is skipped if not available.
        """
        pytest.importorskip("nvidia.dali")

        from tests.benchmarks.performance_targets import (
            measure_adapter_throughput,
            assert_within_ratio,
        )
        from benchmarks.adapters.base import ScenarioConfig
        from benchmarks.adapters.datarax_adapter import DataraxAdapter
        from benchmarks.adapters.dali_adapter import DaliAdapter as DALIAdapter

        config = ScenarioConfig(
            scenario_id="CV-1",
            dataset_size=10_000,
            element_shape=(224, 224, 3),
            batch_size=64,
            transforms=["Normalize", "CastToFloat32"],
            seed=42,
        )

        datarax_tp = measure_adapter_throughput(DataraxAdapter(), config, cv1_large_image_data)
        dali_tp = measure_adapter_throughput(DALIAdapter(), config, cv1_large_image_data)

        assert_within_ratio(
            datarax_tp,
            dali_tp,
            max_ratio=2.0,
            metric_name="CV-1 GPU vision transforms",
        )
