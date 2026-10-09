"""Tests for the new source architecture (eager vs streaming separation).

This module contains tests verifying the new architectural separation between
eager-loading and streaming sources, following TDD principles.

Architecture Goals:
    - Eager sources load all data into host NumPy columns at initialization
    - Streams return host Batches named by the stream (test_stream_contract)
    - The eager TFDS source reads prepared ArrayRecord and never imports TensorFlow
    - O(1) memory shuffling via a keyed Feistel bijection
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.index_words import to_words
from datarax.sources import MemorySource, MemorySourceConfig
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


# =============================================================================
# Tests for Eager Source Architecture (using MemorySource as reference)
# =============================================================================


class TestEagerSourceArchitecture:
    """Tests for eager-loading source behavior.

    These tests verify the core architectural properties that all eager sources
    (TFDSEagerSource, HFEagerSource, MemorySource) should exhibit.
    """

    @pytest.mark.unit
    def test_memory_source_stores_host_numpy_columns(self):
        """Verify that MemorySource holds its records as host NumPy columns, not JAX arrays."""
        data = {"image": np.random.randn(100, 28, 28).astype(np.float32)}
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        assert len(source) == 100
        assert type(source.data["image"]) is np.ndarray

    @pytest.mark.unit
    def test_eager_source_iteration_is_pure_python(self):
        """Verify that iteration doesn't invoke external frameworks."""
        data = {"x": np.arange(10)}
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        # Iteration should work without any external calls
        items = list(source)
        assert len(items) == 10

    @pytest.mark.unit
    def test_eager_source_supports_indexing(self):
        """Verify that eager sources support random access."""
        data = {"x": np.arange(10)}
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        # Should support __getitem__
        assert source[0]["x"] == 0
        assert source[5]["x"] == 5
        assert source[-1]["x"] == 9

    @pytest.mark.unit
    def test_eager_source_supports_batch_retrieval(self):
        """Eager sources read the records named by their indices, statelessly."""
        data = {"x": np.arange(100)}
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        batch1 = source.get_batch(to_words(np.arange(10, dtype=np.uint64)))
        assert batch1.batch_size == 10

        # The same indices read the same records; others read others
        np.testing.assert_array_equal(
            source.get_batch(to_words(np.arange(10, dtype=np.uint64)))["x"], batch1["x"]
        )
        batch2 = source.get_batch(to_words(np.arange(10, 20, dtype=np.uint64)))
        assert batch2["x"][0] != batch1["x"][0]


# =============================================================================
# Tests for TFDS Eager Source (the offline fixture, prepared as ArrayRecord)
# =============================================================================


@pytest.mark.tfds
class TestTFDSEagerSource:
    """Tests for TFDSEagerSource architecture."""

    @staticmethod
    def _source(fixture: TFDSFixture, split: str, **kwargs: object) -> Any:
        from datarax.sources import TFDSEagerConfig, TFDSEagerSource

        config = TFDSEagerConfig(
            name=FIXTURE,
            split=split,
            data_dir=str(fixture.array_record),
            **kwargs,  # type: ignore[arg-type]
        )
        return TFDSEagerSource(config)

    def test_tfds_eager_loads_all_at_init(self, tfds_fixture: TFDSFixture) -> None:
        """TFDS eager source loads all data at init, as host NumPy columns."""
        source = self._source(tfds_fixture, "train[:10]")

        assert isinstance(source.data["image"], np.ndarray)
        assert source.data["image"].shape[0] == 10
        assert len(source) == 10

    def test_tfds_eager_iteration_reads_the_host_columns(self, tfds_fixture: TFDSFixture) -> None:
        """After init, iteration reads the host columns, with no TFDS work."""
        source = self._source(tfds_fixture, "train")

        items = []
        for i, item in enumerate(source):
            items.append(item)
            if i >= 5:
                break

        assert len(items) == 6
        assert isinstance(items[0]["image"], np.ndarray)

    def test_tfds_eager_dataset_info_available(self, tfds_fixture: TFDSFixture) -> None:
        """Dataset info should be cached and available."""
        source = self._source(tfds_fixture, "train")

        assert source.get_dataset_info().name == FIXTURE

    def test_tfds_eager_include_keys_filter(self, tfds_fixture: TFDSFixture) -> None:
        """include_keys should filter output."""
        source = self._source(tfds_fixture, "train", include_keys={"image"})

        assert "image" in source.data
        assert "label" not in source.data


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

        from datarax.sources import HFEagerConfig, HFEagerSource

        def mock_load_dataset(name, split=None, **kwargs):
            del kwargs, name, split
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train")
        source = HFEagerSource(config)

        assert len(source) == 10
        assert "label" in source.data
        assert "feature" in source.data

    @pytest.mark.unit
    def test_hf_eager_iteration(self, mock_dataset, monkeypatch):
        """Iteration should work after init."""
        import datasets

        from datarax.sources import HFEagerConfig, HFEagerSource

        def mock_load_dataset(name, split=None, **kwargs):
            del kwargs, name, split
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train")
        source = HFEagerSource(config)

        items = list(source)
        assert len(items) == 10

    @pytest.mark.unit
    def test_hf_eager_with_filters(self, mock_dataset, monkeypatch):
        """include_keys filter should work."""
        import datasets

        from datarax.sources import HFEagerConfig, HFEagerSource

        def mock_load_dataset(name, split=None, **kwargs):
            del kwargs, name, split
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        config = HFEagerConfig(name="mock", split="train", include_keys={"label"})
        source = HFEagerSource(config)

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
            del kwargs, name, split
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        source = from_hf("mock", "train")

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
            del kwargs, name, split
            if streaming:
                return mock_dataset.to_iterable_dataset()
            return mock_dataset

        monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

        from datarax.sources import HFStreamingSource

        source = from_hf("mock", "train", streaming=True)

        assert isinstance(source, HFStreamingSource)


# =============================================================================
# Tests for Two-Stage Prefetch
# =============================================================================


class TestTwoStagePrefetch:
    """Tests for the two-stage prefetch implementation."""

    @pytest.mark.unit
    def test_prefetch_to_device_basic(self):
        """Basic test that prefetch_to_device works."""
        from datarax.control.prefetcher import prefetch_to_device

        # Create simple iterator
        def data_gen():
            for i in range(10):
                yield {"x": jnp.array([i])}

        prefetched = prefetch_to_device(data_gen(), size=2)

        items = list(prefetched)
        assert len(items) == 10
        assert all(isinstance(item["x"], jax.Array) for item in items)

    @pytest.mark.unit
    def test_prefetch_with_deeper_buffer(self):
        """A deeper device buffer yields every element once, in order."""
        from datarax.control.prefetcher import prefetch_to_device

        def data_gen():
            for i in range(5):
                yield {"x": jnp.array([i])}

        prefetched = prefetch_to_device(data_gen(), size=4)

        items = list(prefetched)
        assert [int(item["x"][0]) for item in items] == list(range(5))
