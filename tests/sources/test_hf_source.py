"""Unit tests for HF Sources (HFEagerSource and HFStreamingSource).

This module contains unit tests for the HuggingFace Datasets data source adapters,
testing both eager-loading and streaming source functionality.
"""

import numpy as np
import pytest
from flax import nnx
import jax

# Skip tests if datasets is not available
datasets = pytest.importorskip("datasets")

from datarax.sources import (
    HFEagerSource,
    HFEagerConfig,
    HFStreamingSource,
    HFStreamingConfig,
    from_hf,
)


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing with text (for streaming/filter tests)."""
    # Create a small synthetic dataset with 10 examples
    data = {
        "text": [f"This is text {i}" for i in range(10)],
        "label": list(range(10)),
        "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
    }
    return datasets.Dataset.from_dict(data)


@pytest.fixture
def mock_numeric_dataset():
    """Create a mock dataset with only numeric data (for eager source tests).

    JAX only supports numeric arrays, so eager sources that load all data to JAX
    arrays need datasets without string fields.
    """
    data = {
        "label": list(range(10)),
        "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
    }
    return datasets.Dataset.from_dict(data)


# =============================================================================
# Unit Tests for HFEagerSource Core Functionality
# =============================================================================


@pytest.mark.unit
def test_hf_eager_source_initialization(mock_numeric_dataset, monkeypatch):
    """Test basic HFEagerSource initialization with config-based API."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test with config-based initialization
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))
    assert source is not None
    assert source.dataset_name == "mock_dataset"
    assert source.split_name == "train"


@pytest.mark.unit
def test_hf_eager_source_iteration(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource iteration functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Get the first element
    data = next(iter(source))

    # Verify the data structure (numeric fields only)
    assert "label" in data or "feature" in data

    # HFEagerSource converts numeric data to JAX arrays
    if "feature" in data:
        assert isinstance(data["feature"], jax.Array)


@pytest.mark.unit
def test_hf_eager_source_batch_method(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's get_batch method."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Get a batch
    batch = source.get_batch(batch_size=4)

    # Verify batch structure
    assert "label" in batch or "feature" in batch

    # Verify batch contains data
    if "label" in batch and isinstance(batch["label"], jax.Array):
        assert batch["label"].shape[0] == 4


@pytest.mark.unit
def test_hf_eager_source_random_access(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's random access capability."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Test random access
    item_3 = source[3]
    if "label" in item_3:
        assert item_3["label"] == 3  # Labels are [0, 1, 2, ..., 9]

    # Test negative indexing
    last_item = source[-1]
    if "label" in last_item:
        assert last_item["label"] == 9


@pytest.mark.unit
def test_hf_eager_source_with_filters(mock_dataset, monkeypatch):
    """Test HFEagerSource with include/exclude key filters."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test include_keys
    config_include = HFEagerConfig(
        name="mock_dataset", split="train", include_keys={"label", "feature"}
    )
    source_include = HFEagerSource(config_include, rngs=nnx.Rngs(42))
    data_include = next(iter(source_include))
    assert "label" in data_include or "feature" in data_include
    assert "text" not in data_include  # Should be excluded

    # Test exclude_keys
    config_exclude = HFEagerConfig(name="mock_dataset", split="train", exclude_keys={"text"})
    source_exclude = HFEagerSource(config_exclude, rngs=nnx.Rngs(42))
    data_exclude = next(iter(source_exclude))
    assert "label" in data_exclude or "feature" in data_exclude
    assert "text" not in data_exclude  # Should be excluded


@pytest.mark.unit
def test_hf_eager_source_shuffling(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource with shuffling enabled."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create source without shuffle
    config_no_shuffle = HFEagerConfig(name="mock_dataset", split="train", shuffle=False)
    source_no_shuffle = HFEagerSource(config_no_shuffle, rngs=nnx.Rngs(42))

    # Create source with shuffle
    config_shuffle = HFEagerConfig(name="mock_dataset", split="train", shuffle=True, seed=43)
    source_shuffle = HFEagerSource(config_shuffle, rngs=nnx.Rngs(43))

    # Collect data from both - create iterator once and use it
    iter_no_shuffle = iter(source_no_shuffle)
    data_no_shuffle = [next(iter_no_shuffle)["label"] for _ in range(5)]

    # Create iterator for shuffled source
    iter_shuffle = iter(source_shuffle)
    data_shuffle = [next(iter_shuffle)["label"] for _ in range(5)]

    # Without shuffle, should be in order [0, 1, 2, 3, 4]
    assert data_no_shuffle == [0, 1, 2, 3, 4]

    # With shuffle, order may be different (though could randomly be same)
    # Just verify they're valid indices
    assert all(0 <= label < 10 for label in data_shuffle)


@pytest.mark.unit
def test_hf_eager_source_length(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's length functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Non-streaming dataset should have length
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))
    assert len(source) == 10


# =============================================================================
# Unit Tests for HFStreamingSource
# =============================================================================


@pytest.mark.unit
def test_hf_streaming_source_initialization(mock_dataset, monkeypatch):
    """Test HFStreamingSource initialization."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            return mock_dataset.to_iterable_dataset()
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFStreamingConfig(name="mock_dataset", split="train", streaming=False)
    source = HFStreamingSource(config, rngs=nnx.Rngs(42))
    assert source is not None
    assert source.dataset_name == "mock_dataset"


@pytest.mark.unit
def test_hf_streaming_source_iteration(mock_numeric_dataset, monkeypatch):
    """Test HFStreamingSource iteration."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFStreamingConfig(name="mock_dataset", split="train")
    source = HFStreamingSource(config, rngs=nnx.Rngs(42))

    # Get first element
    data = next(iter(source))
    assert "label" in data or "feature" in data


@pytest.mark.unit
def test_hf_streaming_source_streaming_mode(mock_dataset, monkeypatch):
    """Test HFStreamingSource with streaming=True."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            return mock_dataset.to_iterable_dataset()
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create streaming source
    config = HFStreamingConfig(name="mock_dataset", split="train", streaming=True)
    source = HFStreamingSource(config, rngs=nnx.Rngs(42))

    assert source.streaming is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
def test_hf_eager_source_empty_dataset(monkeypatch):
    """Test HFEagerSource with an empty dataset."""
    # Create an empty dataset
    empty_dataset = datasets.Dataset.from_dict({"label": []})

    def mock_load_dataset(name, split=None, **kwargs):
        return empty_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="empty_dataset", split="train")

    # Should raise error because no data to load
    with pytest.raises(Exception):  # Will fail due to empty data
        HFEagerSource(config, rngs=nnx.Rngs(42))


@pytest.mark.unit
def test_hf_eager_source_invalid_filters():
    """Test HFEagerSource config with invalid filter configurations."""
    # Should raise error when both include and exclude are specified
    with pytest.raises(ValueError, match="Cannot specify both"):
        HFEagerConfig(
            name="dataset",
            split="train",
            include_keys={"text"},
            exclude_keys={"label"},
        )


@pytest.mark.unit
def test_hf_streaming_source_no_random_access(mock_dataset, monkeypatch):
    """Test that streaming HFStreamingSource doesn't support random access."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            return mock_dataset.to_iterable_dataset()
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create streaming source
    config = HFStreamingConfig(name="mock_dataset", split="train", streaming=True)
    source = HFStreamingSource(config, rngs=nnx.Rngs(42))

    # Streaming sources don't have length, so should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Length unknown"):
        _ = len(source)


@pytest.mark.unit
def test_hf_eager_config_validation():
    """Test HFEagerConfig validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        HFEagerConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        HFEagerConfig(name="dataset")


@pytest.mark.unit
def test_hf_streaming_config_validation():
    """Test HFStreamingConfig validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        HFStreamingConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        HFStreamingConfig(name="dataset")


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.unit
def test_from_hf_creates_eager_by_default(mock_numeric_dataset, monkeypatch):
    """Test that from_hf creates eager source by default."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train", rngs=nnx.Rngs(0))

    # Should be eager source (has .data attribute)
    assert hasattr(source, "data")
    assert isinstance(source, HFEagerSource)


@pytest.mark.unit
def test_from_hf_with_streaming_flag(mock_dataset, monkeypatch):
    """Test that from_hf with streaming=True creates streaming source."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            return mock_dataset.to_iterable_dataset()
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train", streaming=True, rngs=nnx.Rngs(0))

    # Should be streaming source
    assert isinstance(source, HFStreamingSource)
    assert source.streaming is True


@pytest.mark.unit
def test_from_hf_force_eager(mock_numeric_dataset, monkeypatch):
    """Test that from_hf with eager=True creates eager source."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train", eager=True, rngs=nnx.Rngs(0))
    assert isinstance(source, HFEagerSource)


@pytest.mark.unit
def test_from_hf_force_streaming(mock_dataset, monkeypatch):
    """Test that from_hf with eager=False creates streaming source."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train", eager=False, rngs=nnx.Rngs(0))
    assert isinstance(source, HFStreamingSource)


@pytest.mark.unit
def test_from_hf_with_shuffling(mock_numeric_dataset, monkeypatch):
    """Test that from_hf passes shuffle parameter correctly."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train", shuffle=True, seed=42, rngs=nnx.Rngs(0))

    # Should have shuffling enabled
    assert source.shuffle is True
