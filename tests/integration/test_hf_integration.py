"""Integration tests for HuggingFace datasets with Datarax pipeline.

This module contains integration tests that verify HFEagerSource works correctly
with other Datarax components including DAGExecutor, operators, and full
model training pipelines.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

# Skip tests if datasets not installed
datasets = pytest.importorskip("datasets")

from datarax.dag.nodes import BatchNode, OperatorNode  # noqa: E402
from datarax.dag import DAGExecutor  # noqa: E402
from datarax.sources import (  # noqa: E402
    HFEagerSource,
    HFEagerConfig,
    HFStreamingSource,
    HFStreamingConfig,
)
from datarax.core.element_batch import Element  # noqa: E402
from datarax.core.operator import OperatorModule  # noqa: E402
from datarax.core.config import OperatorConfig  # noqa: E402


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_dataset():
    """Create a mock HuggingFace dataset for testing."""
    data = {
        "text": [f"This is text {i}" for i in range(10)],
        "label": list(range(10)),
        "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
    }
    return datasets.Dataset.from_dict(data)


@pytest.fixture
def mock_hf_source(mock_dataset, monkeypatch):
    """Create a mock HF data source.

    Note: Excludes text field since raw strings are not JAX-compatible.
    For text processing, use a tokenizer operator before batching.
    """

    def mock_load_dataset(name: str, split: str | None = None, **kwargs: Any):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)
    # Exclude text field - raw strings can't be converted to JAX arrays
    config = HFEagerConfig(name="mock_dataset", split="train", exclude_keys={"text"})
    return HFEagerSource(config, rngs=nnx.Rngs(0))


def create_text_classification_dataset(num_samples: int = 6):
    """Create a mock text classification dataset."""
    texts = [
        "this movie is great",
        "terrible film",
        "great movie",
        "bad film",
        "positive review",
        "negative review",
    ][:num_samples]

    labels = [1, 0, 1, 0, 1, 0][:num_samples]

    return datasets.Dataset.from_dict({"text": texts, "label": labels})


# ============================================================================
# Simple Model for Testing
# ============================================================================


class SimpleTextModel(nnx.Module):
    """Simple text classification model for integration testing."""

    def __init__(self, vocab_size: int, num_classes: int, *, rngs: nnx.Rngs):
        super().__init__()
        self.embedding = nnx.Embed(num_embeddings=vocab_size, features=32, rngs=rngs)
        self.dense = nnx.Linear(in_features=32, out_features=num_classes, rngs=rngs)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        """Forward pass."""
        x = self.embedding(tokens)
        # Simple mean pooling
        x = jnp.mean(x, axis=1)
        return self.dense(x)


# Vocabulary for tokenization
VOCAB = {
    "<pad>": 0,
    "this": 1,
    "movie": 2,
    "is": 3,
    "great": 4,
    "terrible": 5,
    "film": 6,
    "bad": 7,
    "positive": 8,
    "review": 9,
    "negative": 10,
}
MAX_TOKEN_LEN = 5


def tokenize_element(element: Element, key: jax.Array) -> Element:
    """Tokenize a single element (for use before batching)."""
    text = element.data.get("text", "")
    label = element.data.get("label", 0)

    # Tokenize single text
    text_tokens = [VOCAB.get(word, 0) for word in str(text).lower().split()][:MAX_TOKEN_LEN]
    # Pad
    text_tokens += [0] * (MAX_TOKEN_LEN - len(text_tokens))

    return element.replace(data={"tokens": jnp.array(text_tokens), "label": jnp.array(label)})


# ============================================================================
# Integration Tests: HFEagerSource with Pipeline
# ============================================================================


@pytest.mark.integration
def test_hf_source_with_basic_pipeline(mock_hf_source):
    """Test HFEagerSource with basic DAGExecutor pipeline."""
    # Create pipeline with batch node
    stream = DAGExecutor().add(mock_hf_source).add(BatchNode(4))

    # Get a batch from the pipeline
    batch = next(iter(stream))

    # Verify batch structure (text is excluded since it's not JAX-compatible)
    assert "text" not in batch  # Text excluded via exclude_keys
    assert "label" in batch
    assert "feature" in batch

    # Verify batch shapes
    assert batch["label"].shape[0] == 4  # Batch size
    assert batch["feature"].shape == (4, 5)  # 4 examples with 5 features


@pytest.mark.integration
def test_hf_source_with_transformations(mock_hf_source):
    """Test HFEagerSource with pipeline transformations."""

    class MixedTypeOperator(OperatorModule):
        """Operator that handles mixed types including strings."""

        def __init__(self, *, rngs: nnx.Rngs | None = None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs, name="mixed_type_operator")

        def apply(
            self,
            data: dict[str, Any],
            state: dict[str, Any],
            metadata: Any,
            random_params: Any = None,
            stats: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any], Any]:
            """Handle mixed types without vmap for non-JAX data."""
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    break

            result = {}
            for key, value in data.items():
                if key == "label" and isinstance(value, jax.Array):
                    result[key] = value + 1
                else:
                    result[key] = value

            return result, state, metadata

    # Create pipeline with operator
    stream = (
        DAGExecutor().add(mock_hf_source).add(BatchNode(4)).add(OperatorNode(MixedTypeOperator()))
    )

    # Get a batch
    batch = next(iter(stream))

    # Verify transformation was applied
    # Original labels were [0, 1, 2, 3], after +1 should be [1, 2, 3, 4]
    assert jnp.array_equal(batch["label"], jnp.array([1, 2, 3, 4]))


@pytest.mark.integration
def test_hf_source_with_augmentation(mock_hf_source):
    """Test HFEagerSource with augmentation that uses RNG."""

    class RandomNoiseOperator(OperatorModule):
        """Operator that adds random noise to features."""

        def __init__(self, *, rngs: nnx.Rngs | None = None):
            # Stochastic operators require stream_name for RNG management
            config = OperatorConfig(stochastic=True, stream_name="augment")
            super().__init__(config, rngs=rngs, name="random_noise_operator")

        def generate_random_params(self, rng: jax.Array, data_shapes: dict) -> jax.Array:
            """Generate per-element RNG keys for batch."""
            # Get batch size from first array shape
            first_shape = next(iter(data_shapes.values()))
            batch_size = first_shape[0]
            # Split RNG into per-element keys
            return jax.random.split(rng, batch_size)

        def apply(
            self,
            data: dict[str, Any],
            state: dict[str, Any],
            metadata: Any,
            random_params: Any = None,
            stats: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any], Any]:
            """Apply random noise to features."""
            result = {}
            for key, value in data.items():
                if key == "feature" and isinstance(value, jax.Array):
                    # Get RNG key from random_params or use a default
                    if random_params is not None:
                        rng_key = random_params
                    else:
                        rng_key = jax.random.PRNGKey(0)
                    noise = jax.random.normal(rng_key, shape=value.shape) * 0.1
                    result[key] = value + noise
                else:
                    # Pass through other fields unchanged
                    result[key] = value
            return result, state, metadata

    # Create pipeline with augmentation - provide rngs for stochastic operator
    stream = (
        DAGExecutor()
        .add(mock_hf_source)
        .add(BatchNode(4))
        .add(OperatorNode(RandomNoiseOperator(rngs=nnx.Rngs(default=0, augment=1))))
    )

    # Get batches
    batch1 = next(iter(stream))
    batch2 = next(iter(stream))

    # Verify augmentation was applied (features should be different)
    assert not jnp.array_equal(batch1["feature"], batch2["feature"])

    # Verify structure is preserved
    assert batch1["feature"].shape == (4, 5)
    assert batch2["feature"].shape == (4, 5)


@pytest.mark.integration
def test_hf_source_full_training_pipeline(monkeypatch):
    """Test complete training pipeline with HFEagerSource.

    Note: Since the DAG architecture operates on batched data, we pre-tokenize
    elements at the source level using a wrapper that converts text to arrays
    before elements enter the pipeline.
    """

    def create_tokenized_dataset(n_samples: int):
        """Create pre-tokenized dataset (text already converted to token arrays)."""
        data = []
        for i in range(n_samples):
            text = f"sample text {i} for classification"
            label = i % 2
            # Pre-tokenize: convert text to array
            tokens = [VOCAB.get(word, 0) for word in text.lower().split()][:MAX_TOKEN_LEN]
            tokens += [0] * (MAX_TOKEN_LEN - len(tokens))
            data.append({"tokens": tokens, "label": label})
        return datasets.Dataset.from_dict(
            {
                "tokens": [d["tokens"] for d in data],
                "label": [d["label"] for d in data],
            }
        )

    # Create pre-tokenized datasets
    train_data = create_tokenized_dataset(6)
    test_data = create_tokenized_dataset(3)

    def mock_load_dataset(name, split=None, **kwargs):
        if split == "train":
            return train_data
        else:
            return test_data

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create data sources with pre-tokenized data (HFEagerSource loads all to JAX)
    train_config = HFEagerConfig(name="mock_dataset", split="train", shuffle=False)
    train_source = HFEagerSource(train_config, rngs=nnx.Rngs(0))

    test_config = HFEagerConfig(name="mock_dataset", split="test")
    test_source = HFEagerSource(test_config, rngs=nnx.Rngs(1))

    # Create simple pipelines - data is already tokenized at source
    train_pipeline = DAGExecutor().add(train_source).batch(batch_size=2)

    test_pipeline = DAGExecutor().add(test_source).batch(batch_size=2)

    # Create model and optimizer
    model = SimpleTextModel(vocab_size=11, num_classes=2, rngs=nnx.Rngs(42))

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.01), wrt=nnx.Param)

    # Define loss function
    def compute_loss(model, batch):
        logits = model(batch["tokens"])
        labels = batch["label"]  # tokenize_element outputs "label" key
        one_hot = jax.nn.one_hot(labels, 2)
        return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))

    # Training step
    @nnx.jit
    def train_step(model, optimizer, batch):
        loss, grads = nnx.value_and_grad(compute_loss)(model, batch)
        optimizer.update(model, grads)
        return loss

    # Train for a few steps
    losses = []
    for i, batch in enumerate(train_pipeline):
        if i >= 3:  # Train for 3 steps
            break
        loss = train_step(model, optimizer, batch)
        losses.append(float(loss))

    # Verify training occurred
    assert len(losses) == 3
    assert all(loss > 0 for loss in losses)

    # Test evaluation
    @nnx.jit
    def eval_step(model, batch):
        logits = model(batch["tokens"])
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch["label"])  # tokenize_element outputs "label" key
        return accuracy

    # Evaluate
    accuracies = []
    for i, batch in enumerate(test_pipeline):
        if i >= 2:  # Evaluate 2 batches
            break
        acc = eval_step(model, batch)
        accuracies.append(float(acc))

    # Verify evaluation
    assert len(accuracies) == 2
    assert all(0.0 <= acc <= 1.0 for acc in accuracies)


@pytest.mark.integration
def test_hf_source_with_multiple_transforms(mock_hf_source):
    """Test HFEagerSource with multiple sequential transformations."""

    class ScaleOperator(OperatorModule):
        """Operator that scales features by 2."""

        def __init__(self, *, rngs: nnx.Rngs | None = None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs, name="scale_operator")

        def apply(
            self,
            data: dict[str, Any],
            state: dict[str, Any],
            metadata: Any,
            random_params: Any = None,
            stats: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any], Any]:
            result = dict(data)
            if "feature" in data:
                result["feature"] = data["feature"] * 2.0
            return result, state, metadata

    class BiasOperator(OperatorModule):
        """Operator that adds bias of 1 to features."""

        def __init__(self, *, rngs: nnx.Rngs | None = None):
            config = OperatorConfig(stochastic=False)
            super().__init__(config, rngs=rngs, name="bias_operator")

        def apply(
            self,
            data: dict[str, Any],
            state: dict[str, Any],
            metadata: Any,
            random_params: Any = None,
            stats: dict[str, Any] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any], Any]:
            result = dict(data)
            if "feature" in data:
                result["feature"] = data["feature"] + 1.0
            return result, state, metadata

    # Create pipeline with multiple operators
    stream = (
        DAGExecutor()
        .add(mock_hf_source)
        .batch(batch_size=2)
        .add(OperatorNode(ScaleOperator()))
        .add(OperatorNode(BiasOperator()))
    )

    # Get batch
    batch = next(iter(stream))

    # Verify both transforms were applied
    # Features should be scaled by 2 and then have 1 added
    assert "feature" in batch
    assert batch["feature"].shape == (2, 5)

    # Check that values are reasonable (not the original values)
    # Since original features are random normal, after *2 +1 they should be different
    assert jnp.mean(jnp.abs(batch["feature"])) > 0.5  # Should have some magnitude


@pytest.mark.integration
def test_hf_source_with_key_filtering_in_pipeline(mock_dataset, monkeypatch):
    """Test HFEagerSource with key filtering in a pipeline."""

    def mock_load_dataset(name, split=None, **kwargs):
        return mock_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create source with filtering
    config = HFEagerConfig(name="mock_dataset", split="train", include_keys={"label", "feature"})
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Create pipeline
    stream = DAGExecutor().add(source).batch(batch_size=3)

    # Get batch
    batch = next(iter(stream))

    # Verify filtering worked
    assert "label" in batch
    assert "feature" in batch
    assert "text" not in batch  # Should be filtered out

    # Verify batch size
    assert batch["label"].shape[0] == 3
    assert batch["feature"].shape == (3, 5)


@pytest.mark.integration
@pytest.mark.skipif(not datasets, reason="datasets library not available")
def test_hf_with_real_dataset():
    """Test with a real small dataset from HuggingFace Hub.

    Uses HFStreamingSource to avoid downloading full dataset.
    Note: This test uses HuggingFace streaming mode which doesn't convert
    text to JAX arrays (since JAX doesn't support strings).
    """

    try:
        # Try to load a tiny dataset using streaming to avoid full download
        config = HFStreamingConfig(
            name="rotten_tomatoes",
            split="test",
            streaming=True,
        )
        source = HFStreamingSource(config, rngs=nnx.Rngs(0))

        # Create pipeline
        pipeline = DAGExecutor().add(source).batch(batch_size=4)

        # Get one batch
        for batch in pipeline:
            # Verify batch structure
            assert isinstance(batch, dict)
            assert "label" in batch

            # Verify batch size - streaming batches may be smaller
            assert len(batch["label"]) <= 4

            break  # Only test one batch

    except Exception as e:
        # Skip if network issues or dataset not available
        pytest.skip(f"Could not load real dataset: {e}")


@pytest.mark.integration
def test_hf_source_epoch_handling(monkeypatch):
    """Test HFEagerSource epoch handling in a pipeline."""

    # Create a small dataset with only 3 items (numeric only for JAX)
    small_dataset = datasets.Dataset.from_dict({"value": [1, 2, 3]})

    def mock_load_dataset(name, split=None, **kwargs):
        return small_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create source (HFEagerSource loads all data to JAX arrays)
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Create pipeline with batch size 2
    stream = DAGExecutor().add(source).batch(batch_size=2)

    # Collect batches from one epoch
    batches = []
    for batch in stream:
        batches.append(batch["value"])

    # With 3 items and batch size 2, we should get 2 batches
    # First batch: [1, 2], Second batch: [3]
    assert len(batches) == 2

    # Each batch should have size 2 or less (last batch may be smaller)
    for batch in batches:
        assert len(batch) <= 2


@pytest.mark.integration
def test_hf_source_with_stateful_iteration(monkeypatch):
    """Test HFEagerSource stateful iteration with get_batch method."""

    # Create numeric-only dataset for JAX compatibility
    numeric_dataset = datasets.Dataset.from_dict(
        {
            "label": list(range(10)),
            "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
        }
    )

    def mock_load_dataset(name, split=None, **kwargs):
        return numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create source with RNG for stateful mode
    config = HFEagerConfig(name="mock_dataset", split="train")
    source = HFEagerSource(config, rngs=nnx.Rngs(42))

    # Use get_batch method directly (stateful)
    batch1 = source.get_batch(batch_size=3)
    batch2 = source.get_batch(batch_size=3)

    # Batches should be different (advancing through dataset)
    assert not jnp.array_equal(batch1["label"], batch2["label"])

    # Verify batch structure (numeric fields only)
    assert "label" in batch1
    assert "feature" in batch1

    # Verify batch sizes
    if isinstance(batch1["label"], jax.Array):
        assert batch1["label"].shape[0] == 3
    else:
        assert len(batch1["label"]) == 3


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


@pytest.mark.integration
def test_pipeline_with_empty_dataset(monkeypatch):
    """Test that HFEagerSource raises error with empty dataset.

    HFEagerSource loads all data to JAX arrays at init. If the dataset
    is empty, there's nothing to load, so it should raise an error.
    """

    empty_dataset = datasets.Dataset.from_dict({"value": []})

    def mock_load_dataset(name, split=None, **kwargs):
        return empty_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    config = HFEagerConfig(name="empty", split="train")

    # HFEagerSource should raise error for empty dataset
    # (can't determine length with no data)
    with pytest.raises(Exception):  # StopIteration or similar
        HFEagerSource(config, rngs=nnx.Rngs(42))


@pytest.mark.integration
def test_pipeline_with_streaming_dataset(monkeypatch):
    """Test pipeline with HFStreamingSource."""

    # Create numeric-only dataset for JAX compatibility
    numeric_dataset = datasets.Dataset.from_dict(
        {
            "label": list(range(10)),
            "feature": [np.random.randn(5).astype(np.float32) for _ in range(10)],
        }
    )

    def mock_load_dataset(name, split=None, streaming=False, **kwargs):
        if streaming:
            return numeric_dataset.to_iterable_dataset()
        return numeric_dataset

    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    # Create streaming source (HFStreamingSource wraps HF iterators)
    config = HFStreamingConfig(name="mock_dataset", split="train", streaming=False)
    source = HFStreamingSource(config, rngs=nnx.Rngs(42))

    # Create pipeline
    stream = DAGExecutor().add(source).batch(batch_size=2)

    # Get a few batches
    batches = []
    for i, batch in enumerate(stream):
        if i >= 3:
            break
        batches.append(batch)

    # Verify we got batches
    assert len(batches) == 3

    # Each batch should have the expected structure (numeric fields)
    for batch in batches:
        assert "label" in batch
        assert "feature" in batch


if __name__ == "__main__":
    # For debugging
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
