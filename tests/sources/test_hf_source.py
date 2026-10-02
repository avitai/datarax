"""Unit tests for ``HFEagerSource`` and ``from_hf``'s default; the stream is ``test_hf_stream``."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image
from substrax.testing.compiles import compiled_programs


# Skip tests if datasets is not available
datasets = pytest.importorskip("datasets")

from datarax.core.index_words import to_words
from datarax.sources import from_hf, HFEagerConfig, HFEagerSource


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
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test with config-based initialization
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)
    assert source is not None
    assert source.dataset_name == "mock_dataset"
    assert source.split_name == "train"


@pytest.mark.unit
def test_hf_eager_source_iteration(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource iteration functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)

    # Get the first element
    data = next(iter(source))

    # Verify the data structure (numeric fields only)
    assert "label" in data or "feature" in data

    # HFEagerSource holds numeric data as host NumPy columns
    if "feature" in data:
        assert isinstance(data["feature"], np.ndarray)


@pytest.mark.unit
def test_hf_eager_source_preserves_text_columns(mock_dataset, monkeypatch):
    """Text columns are kept as the records' provenance, beside the array columns."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)
    item = next(iter(source))

    assert len(source) == 10
    assert "text" not in item
    assert source._provenance.value[0]["text"] == "This is text 0"
    assert isinstance(item["label"], np.generic)  # a scalar row of a host column


@pytest.mark.unit
def test_hf_eager_source_batch_method(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's host read, ``get_batch(indices)``."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)

    # Read records 0..3
    batch = source.get_batch(to_words(np.arange(4, dtype=np.uint64)))

    assert set(batch.data) == {"label", "feature"}
    assert batch["label"].shape[0] == 4
    np.testing.assert_array_equal(batch["label"], np.arange(4))


@pytest.mark.unit
def test_hf_eager_source_random_access(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's random access capability."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)

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
        del kwargs, name, split
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Test include_keys
    config_include = HFEagerConfig(
        name="mock_dataset", split="train", include_keys={"label", "feature"}
    )
    source_include = HFEagerSource(config_include)
    data_include = next(iter(source_include))
    assert "label" in data_include or "feature" in data_include
    assert "text" not in data_include  # Should be excluded

    # Test exclude_keys
    config_exclude = HFEagerConfig(name="mock_dataset", split="train", exclude_keys={"text"})
    source_exclude = HFEagerSource(config_exclude)
    data_exclude = next(iter(source_exclude))
    assert "label" in data_exclude or "feature" in data_exclude
    assert "text" not in data_exclude  # Should be excluded


@pytest.mark.unit
def test_hf_eager_source_length(mock_numeric_dataset, monkeypatch):
    """Test HFEagerSource's length functionality."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Non-streaming dataset should have length
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config)
    assert len(source) == 10


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
def test_hf_eager_source_empty_dataset(monkeypatch):
    """Test HFEagerSource with an empty dataset."""
    # Create an empty dataset
    empty_dataset = datasets.Dataset.from_dict({"label": []})

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return empty_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="empty_dataset", split="train")

    # Should raise error because no data to load
    with pytest.raises(ValueError, match="produced no elements"):
        HFEagerSource(config)


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
def test_hf_eager_config_validation():
    """Test HFEagerConfig validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        HFEagerConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        HFEagerConfig(name="dataset")


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.unit
def test_from_hf_creates_eager_by_default(mock_numeric_dataset, monkeypatch):
    """Test that from_hf creates eager source by default."""

    def mock_load_dataset(name, split=None, **kwargs):
        del kwargs, name, split
        return mock_numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    source = from_hf("mock", "train")

    # Should be eager source (has .data attribute)
    assert hasattr(source, "data")
    assert isinstance(source, HFEagerSource)


# =============================================================================
# Image columns: loaded once per column
# =============================================================================


@pytest.fixture
def image_dataset():
    """A dataset with an image column, decoded by datasets as PIL images, and a label column."""
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, size=(16, 8, 8), dtype=np.uint8)
    features = datasets.Features({"image": datasets.Image(), "label": datasets.Value("int64")})
    dataset = datasets.Dataset.from_dict(
        {"image": [Image.fromarray(row) for row in pixels], "label": list(range(16))},
        features=features,
    )
    return dataset, pixels


@pytest.mark.unit
def test_hf_eager_source_image_column_matches_the_decoded_images(image_dataset, monkeypatch):
    """Every image lands in one array, in row order, with its pixel values and dtype."""
    dataset, pixels = image_dataset
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split=None, **kwargs: dataset)

    source = HFEagerSource(HFEagerConfig(name="images", split="train"))

    np.testing.assert_array_equal(np.asarray(source.data["image"]), pixels)
    assert source.data["image"].dtype == jnp.uint8
    np.testing.assert_array_equal(np.asarray(source.data["label"]), np.arange(16))


@pytest.mark.unit
def test_hf_eager_source_construction_stacks_no_device_array_per_row(image_dataset, monkeypatch):
    """A column is built once, not by stacking one device array per row.

    Stacking per-row JAX arrays compiles a ``jit(stack)`` program whose input count is the row
    count, which made building MNIST's training split take tens of minutes.
    """
    dataset, _ = image_dataset
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split=None, **kwargs: dataset)

    # An earlier test may have compiled the same programs; clearing the in-memory caches makes
    # every program this test needs compile here, whatever ran before.
    jax.clear_caches()
    with compiled_programs() as control:
        jnp.stack([jnp.arange(3), jnp.arange(3)])
    assert "jit(stack)" in control, "positive control: a device stack must be recorded"
    jax.clear_caches()

    with compiled_programs() as compiled:
        HFEagerSource(HFEagerConfig(name="images", split="train"))

    assert "jit(stack)" not in compiled
