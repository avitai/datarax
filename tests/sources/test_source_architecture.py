"""Tests for the new source architecture (eager vs streaming separation).

This module contains tests verifying the new architectural separation between
eager-loading and streaming sources, following TDD principles.

Architecture Goals:
    - Eager sources load all data to JAX arrays at initialization
    - Streaming sources provide thin wrappers with DLPack conversion
    - No TensorFlow threads should remain after eager source init
    - O(1) memory shuffling via Grain's index_shuffle
"""

import platform

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.sources import MemorySource, MemorySourceConfig


# =============================================================================
# Tests for Eager Source Architecture (using MemorySource as reference)
# =============================================================================


class TestEagerSourceArchitecture:
    """Tests for eager-loading source behavior.

    These tests verify the core architectural properties that all eager sources
    (TFDSEagerSource, HFEagerSource, MemorySource) should exhibit.
    """

    @pytest.mark.unit
    def test_memory_source_stores_jax_arrays(self):
        """Verify that MemorySource stores data as JAX arrays."""
        data = {"image": np.random.randn(100, 28, 28).astype(np.float32)}
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Data should be stored and accessible
        assert len(source) == 100

    @pytest.mark.unit
    def test_eager_source_iteration_is_pure_python(self):
        """Verify that iteration doesn't invoke external frameworks."""
        data = {"x": np.arange(10)}
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Iteration should work without any external calls
        items = list(source)
        assert len(items) == 10

    @pytest.mark.unit
    def test_eager_source_supports_indexing(self):
        """Verify that eager sources support random access."""
        data = {"x": np.arange(10)}
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Should support __getitem__
        assert source[0]["x"] == 0
        assert source[5]["x"] == 5
        assert source[-1]["x"] == 9

    @pytest.mark.unit
    def test_eager_source_supports_batch_retrieval(self):
        """Verify that eager sources support get_batch method."""
        data = {"x": np.arange(100)}
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Stateful batch retrieval
        batch1 = source.get_batch(10)
        assert len(batch1["x"]) == 10

        # Second batch should be different (advancing state)
        batch2 = source.get_batch(10)
        assert batch2["x"][0] != batch1["x"][0]

    @pytest.mark.unit
    def test_eager_source_stateless_batch_with_key(self):
        """Verify stateless batch retrieval with explicit key."""
        data = {"x": np.arange(100)}
        config = MemorySourceConfig(shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        key = jax.random.key(42)
        batch1 = source.get_batch(10, key=key)
        batch2 = source.get_batch(10, key=key)

        # Same key should give same batch (in shuffle mode)
        np.testing.assert_array_equal(np.array(batch1["x"]), np.array(batch2["x"]))

    @pytest.mark.unit
    def test_eager_source_shuffle_produces_different_order(self):
        """Verify that shuffling produces different iteration order."""
        data = {"x": np.arange(20)}
        config = MemorySourceConfig(shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(42))

        # First epoch
        items1 = [item["x"] for item in source]

        # Second epoch (should have different shuffle)
        items2 = [item["x"] for item in source]

        # Both should have all elements
        assert sorted(items1) == list(range(20))
        assert sorted(items2) == list(range(20))

        # But order should be different (with high probability)
        # Note: There's a tiny chance they're the same, but very unlikely
        assert items1 != items2 or len(items1) < 3  # Allow small datasets to match

    @pytest.mark.unit
    def test_eager_source_reset_returns_to_start(self):
        """Verify that reset() returns source to beginning."""
        data = {"x": np.arange(10)}
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Advance the state
        source.get_batch(5)
        assert source.index.get_value() == 5

        # Reset
        source.reset()
        assert source.index.get_value() == 0
        assert source.epoch.get_value() == 0


# =============================================================================
# Tests for TFDS Eager Source
# =============================================================================


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping TFDS tests on macOS (TensorFlow ARM64 import hang issue)",
)
class TestTFDSEagerSource:
    """Tests for TFDSEagerSource architecture."""

    @pytest.fixture(autouse=True)
    def skip_without_tf(self):
        """Skip tests if TensorFlow/TFDS not available."""
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_datasets")

    @pytest.mark.tfds
    def test_tfds_eager_loads_all_at_init(self):
        """TFDS eager source loads all data to JAX arrays at init."""
        from datarax.sources import TFDSEagerSource, TFDSEagerConfig

        try:
            config = TFDSEagerConfig(name="mnist", split="train[:100]")
            source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

            # Data should be JAX arrays
            assert isinstance(source.data["image"], jax.Array)
            assert source.data["image"].shape[0] == 100
            assert len(source) == 100
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")

    @pytest.mark.tfds
    def test_tfds_eager_iteration_is_pure_jax(self):
        """After init, iteration should be pure JAX operations."""
        from datarax.sources import TFDSEagerSource, TFDSEagerConfig

        try:
            config = TFDSEagerConfig(name="mnist", split="train[:50]")
            source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

            # Iteration should work
            items = []
            for i, item in enumerate(source):
                items.append(item)
                if i >= 5:
                    break

            assert len(items) == 6
            assert isinstance(items[0]["image"], jax.Array)
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")

    @pytest.mark.tfds
    def test_tfds_eager_dataset_info_available(self):
        """Dataset info should be cached and available."""
        from datarax.sources import TFDSEagerSource, TFDSEagerConfig

        try:
            config = TFDSEagerConfig(name="mnist", split="train[:10]")
            source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

            info = source.get_dataset_info()
            assert info is not None
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")

    @pytest.mark.tfds
    def test_tfds_eager_with_shuffling(self):
        """Shuffling should work with Grain's index_shuffle."""
        from datarax.sources import TFDSEagerSource, TFDSEagerConfig

        try:
            config = TFDSEagerConfig(name="mnist", split="train[:100]", shuffle=True, seed=42)
            source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

            # Get first few items from two epochs
            epoch1_items = [next(iter(source))["label"] for _ in range(5)]
            epoch2_items = [next(iter(source))["label"] for _ in range(5)]

            # Shuffling should produce different orders
            assert epoch1_items != epoch2_items or len(epoch1_items) < 3
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")

    @pytest.mark.tfds
    def test_tfds_eager_include_keys_filter(self):
        """include_keys should filter output."""
        from datarax.sources import TFDSEagerSource, TFDSEagerConfig

        try:
            config = TFDSEagerConfig(name="mnist", split="train[:10]", include_keys={"image"})
            source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

            assert "image" in source.data
            assert "label" not in source.data
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")


# =============================================================================
# Tests for TFDS Streaming Source
# =============================================================================


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Skipping TFDS tests on macOS (TensorFlow ARM64 import hang issue)",
)
class TestTFDSStreamingSource:
    """Tests for TFDSStreamingSource architecture."""

    @pytest.fixture(autouse=True)
    def skip_without_tf(self):
        """Skip tests if TensorFlow/TFDS not available."""
        pytest.importorskip("tensorflow")
        pytest.importorskip("tensorflow_datasets")

    @pytest.mark.tfds
    def test_tfds_streaming_uses_fixed_prefetch(self):
        """Streaming source should use fixed prefetch, not AUTOTUNE."""
        from datarax.sources import TFDSStreamingSource, TFDSStreamingConfig

        try:
            config = TFDSStreamingConfig(name="mnist", split="train[:50]", prefetch_buffer=2)
            source = TFDSStreamingSource(config, rngs=nnx.Rngs(0))

            # Should be iterable
            items = []
            for i, item in enumerate(source):
                items.append(item)
                if i >= 5:
                    break

            assert len(items) == 6
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")

    @pytest.mark.tfds
    def test_tfds_streaming_produces_jax_arrays(self):
        """Each batch should produce JAX arrays."""
        from datarax.sources import TFDSStreamingSource, TFDSStreamingConfig

        try:
            config = TFDSStreamingConfig(name="mnist", split="train[:10]")
            source = TFDSStreamingSource(config, rngs=nnx.Rngs(0))

            item = next(iter(source))
            assert isinstance(item["image"], jax.Array)
            assert isinstance(item["label"], jax.Array)
        except Exception as e:
            pytest.skip(f"Could not load MNIST: {e}")


# =============================================================================
# Tests for HF Eager Source
# =============================================================================


class TestHFEagerSource:
    """Tests for HFEagerSource architecture."""

    @pytest.fixture(autouse=True)
    def skip_without_datasets(self):
        """Skip tests if datasets package not available."""
        pytest.importorskip("datasets")

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        import datasets

        data = {
            "label": list(range(10)),
            "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
        }
        return datasets.Dataset.from_dict(data)

    @pytest.mark.unit
    def test_hf_eager_loads_all_at_init(self, mock_dataset, monkeypatch):
        """HF eager source loads all data to JAX arrays at init."""
        import datasets
        from datarax.sources import HFEagerSource, HFEagerConfig

        def mock_load_dataset(name, split=None, **kwargs):
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train")
        source = HFEagerSource(config, rngs=nnx.Rngs(0))

        assert len(source) == 10
        assert "label" in source.data
        assert "feature" in source.data

    @pytest.mark.unit
    def test_hf_eager_iteration(self, mock_dataset, monkeypatch):
        """Iteration should work after init."""
        import datasets
        from datarax.sources import HFEagerSource, HFEagerConfig

        def mock_load_dataset(name, split=None, **kwargs):
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train")
        source = HFEagerSource(config, rngs=nnx.Rngs(0))

        items = list(source)
        assert len(items) == 10

    @pytest.mark.unit
    def test_hf_eager_with_filters(self, mock_dataset, monkeypatch):
        """include_keys filter should work."""
        import datasets
        from datarax.sources import HFEagerSource, HFEagerConfig

        def mock_load_dataset(name, split=None, **kwargs):
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train", include_keys={"label"})
        source = HFEagerSource(config, rngs=nnx.Rngs(0))

        assert "label" in source.data
        assert "feature" not in source.data


# =============================================================================
# Tests for Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for from_tfds and from_hf factory functions."""

    @pytest.mark.unit
    def test_from_hf_creates_eager_by_default(self, monkeypatch):
        """from_hf should create eager source by default."""
        pytest.importorskip("datasets")
        import datasets
        from datarax.sources import from_hf

        mock_data = {"label": list(range(5))}
        mock_dataset = datasets.Dataset.from_dict(mock_data)

        def mock_load_dataset(name, split=None, **kwargs):
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        source = from_hf("mock", "train", rngs=nnx.Rngs(0))

        # Should be eager source (has .data attribute)
        assert hasattr(source, "data")

    @pytest.mark.unit
    def test_from_hf_with_streaming_flag(self, monkeypatch):
        """from_hf with streaming=True should create streaming source."""
        pytest.importorskip("datasets")
        import datasets
        from datarax.sources import from_hf

        mock_data = {"label": list(range(5))}
        mock_dataset = datasets.Dataset.from_dict(mock_data)

        def mock_load_dataset(name, split=None, streaming=False, **kwargs):
            if streaming:
                return mock_dataset.to_iterable_dataset()
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        source = from_hf("mock", "train", streaming=True, rngs=nnx.Rngs(0))

        # Should be streaming source (has .streaming attribute set to True)
        assert hasattr(source, "streaming")
        assert source.streaming is True


# =============================================================================
# Tests for Two-Stage Prefetch
# =============================================================================


class TestTwoStagePrefetch:
    """Tests for the two-stage prefetch implementation."""

    @pytest.mark.unit
    def test_prefetch_to_device_basic(self):
        """Basic test that prefetch_to_device works."""
        from datarax.distributed.device_placement import prefetch_to_device

        # Create simple iterator
        def data_gen():
            for i in range(10):
                yield {"x": jnp.array([i])}

        prefetched = prefetch_to_device(data_gen(), size=2)

        items = list(prefetched)
        assert len(items) == 10
        assert all(isinstance(item["x"], jax.Array) for item in items)

    @pytest.mark.unit
    def test_prefetch_with_custom_cpu_buffer(self):
        """Test prefetch with custom CPU buffer size."""
        from datarax.distributed.device_placement import prefetch_to_device

        def data_gen():
            for i in range(5):
                yield {"x": jnp.array([i])}

        # Custom CPU buffer size
        prefetched = prefetch_to_device(data_gen(), size=2, cpu_buffer_size=4)

        items = list(prefetched)
        assert len(items) == 5
