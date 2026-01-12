"""Unit tests for HFSource.

This module contains unit tests for the HuggingFace Datasets data source adapter,
testing individual HFSource functionality in isolation.
"""

import numpy as np
import pytest
from flax import nnx
import jax

# Skip tests if datasets is not available
datasets = pytest.importorskip("datasets")

from datarax.sources import HFSource, HfDataSourceConfig  # noqa: E402


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    # Create a small synthetic dataset with 10 examples
    data = {
        "text": [f"This is text {i}" for i in range(10)],
        "label": list(range(10)),
        "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
    }
    return datasets.Dataset.from_dict(data)


# ============================================================================
# Unit Tests for HFSource Core Functionality
# ============================================================================


@pytest.mark.unit
def test_hf_source_initialization(mock_dataset, monkeypatch):
    """Test basic HFSource initialization with config-based API."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test with config-based initialization
    config = HfDataSourceConfig(name="mock_dataset", split="train")
    source = HFSource(config, rngs=nnx.Rngs(42))
    assert source is not None
    assert source.dataset_name == "mock_dataset"
    assert source.split == "train"


@pytest.mark.unit
def test_hf_source_iteration(mock_dataset, monkeypatch):
    """Test HFSource iteration functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HfDataSourceConfig(name="mock_dataset", split="train")
    source = HFSource(config, rngs=nnx.Rngs(42))

    # Get the first element
    data = next(iter(source))

    # Verify the data structure
    assert "text" in data
    assert "label" in data
    assert "feature" in data

    # HFSource converts numeric data to JAX arrays
    assert isinstance(data["label"], jax.Array | int | float)
    assert isinstance(data["feature"], jax.Array)


@pytest.mark.unit
def test_hf_source_batch_method(mock_dataset, monkeypatch):
    """Test HFSource's get_batch method."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HfDataSourceConfig(name="mock_dataset", split="train")
    source = HFSource(config, rngs=nnx.Rngs(42))

    # Get a batch
    batch = source.get_batch(batch_size=4)

    # Verify batch structure
    assert "text" in batch
    assert "label" in batch
    assert "feature" in batch

    # Verify batch contains data
    assert isinstance(batch["label"], jax.Array | list)
    assert isinstance(batch["feature"], jax.Array | list)

    # Check batch size
    if isinstance(batch["label"], jax.Array):
        assert batch["label"].shape[0] == 4
    else:
        assert len(batch["label"]) == 4


@pytest.mark.unit
def test_hf_source_random_access(mock_dataset, monkeypatch):
    """Test HFSource's random access capability."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HfDataSourceConfig(name="mock_dataset", split="train", streaming=False)
    source = HFSource(config, rngs=nnx.Rngs(42))

    # Test random access
    item_3 = source[3]
    assert item_3["label"] == 3  # Labels are [0, 1, 2, ..., 9]

    # Test negative indexing
    last_item = source[-1]
    assert last_item["label"] == 9


@pytest.mark.unit
def test_hf_source_with_filters(mock_dataset, monkeypatch):
    """Test HFSource with include/exclude key filters."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test include_keys
    config_include = HfDataSourceConfig(
        name="mock_dataset", split="train", include_keys={"label", "feature"}
    )
    source_include = HFSource(config_include, rngs=nnx.Rngs(42))
    data_include = next(iter(source_include))
    assert "label" in data_include
    assert "feature" in data_include
    assert "text" not in data_include  # Should be excluded

    # Test exclude_keys
    config_exclude = HfDataSourceConfig(name="mock_dataset", split="train", exclude_keys={"text"})
    source_exclude = HFSource(config_exclude, rngs=nnx.Rngs(42))
    data_exclude = next(iter(source_exclude))
    assert "label" in data_exclude
    assert "feature" in data_exclude
    assert "text" not in data_exclude  # Should be excluded


@pytest.mark.unit
def test_hf_source_shuffling(mock_dataset, monkeypatch):
    """Test HFSource with shuffling enabled."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create two sources - one shuffled, one not
    config_no_shuffle = HfDataSourceConfig(name="mock_dataset", split="train", shuffle=False)
    source_no_shuffle = HFSource(config_no_shuffle, rngs=nnx.Rngs(42))

    config_shuffle = HfDataSourceConfig(name="mock_dataset", split="train", shuffle=True)
    source_shuffle = HFSource(config_shuffle, rngs=nnx.Rngs(43))

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
def test_hf_source_length(mock_dataset, monkeypatch):
    """Test HFSource's length functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Non-streaming dataset should have length
    config = HfDataSourceConfig(name="mock_dataset", split="train", streaming=False)
    source = HFSource(config, rngs=nnx.Rngs(42))
    assert len(source) == 10


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


@pytest.mark.unit
def test_hf_source_empty_dataset(monkeypatch):
    """Test HFSource with an empty dataset."""
    # Create an empty dataset
    empty_dataset = datasets.Dataset.from_dict({"text": [], "label": []})

    def mock_load_dataset(name, split=None, **kwargs):
        return empty_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HfDataSourceConfig(name="empty_dataset", split="train")
    source = HFSource(config, rngs=nnx.Rngs(42))

    # Iteration should work but produce no items
    items = list(iter(source))
    assert len(items) == 0

    # get_batch should return empty dict for empty dataset
    batch = source.get_batch(batch_size=4)
    assert len(batch) == 0  # Empty dataset returns empty dict


@pytest.mark.unit
def test_hf_source_invalid_filters():
    """Test HFSource config with invalid filter configurations."""
    # Should raise error when both include and exclude are specified
    with pytest.raises(ValueError, match="Cannot specify both"):
        HfDataSourceConfig(
            name="dataset",
            split="train",
            include_keys={"text"},
            exclude_keys={"label"},
        )


@pytest.mark.unit
def test_hf_source_streaming_no_random_access(mock_dataset, monkeypatch):
    """Test that streaming datasets don't support random access."""

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            # Create an iterable dataset for streaming
            return mock_dataset.to_iterable_dataset()
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create streaming source
    config = HfDataSourceConfig(name="mock_dataset", split="train", streaming=True)
    source = HFSource(config, rngs=nnx.Rngs(42))

    # Random access should raise error
    with pytest.raises(NotImplementedError, match="Random access is not supported"):
        _ = source[0]


@pytest.mark.unit
def test_hf_source_config_validation():
    """Test HFSource config validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        HfDataSourceConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        HfDataSourceConfig(name="dataset")
