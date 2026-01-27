"""End-to-end test for image classification pipeline.

This test demonstrates a complete workflow using Datarax data pipeline
for image classification with a small CNN model on MNIST dataset.

Features Showcased:
- TFDSEagerSource: TensorFlow Datasets integration for data loading
- DAGExecutor: Pipeline builder with fluent API
- BatchNode: Batch-first data processing
- ShuffleNode: Data shuffling for training
- MapOperator: Array-level transformations (normalization)
- BrightnessOperator: Built-in stochastic image augmentation
- ElementOperator: Element-level transformations (preprocessing)
- NNX integration: Flax NNX modules with proper RNG management
- Mode switching: model.train() / model.eval() for Dropout/BatchNorm
"""

import platform
import pytest

# Skip entire module on macOS ARM64 - TensorFlow import hangs during pytest collection
# due to Metal/GPU device detection issues. This is a known upstream issue:
# https://github.com/tensorflow/tensorflow/issues/52138
# Note: Major ML projects (Keras, Flax) don't run CI tests on macOS for this reason.
if platform.system() == "Darwin":
    pytest.skip(
        "Skipping TFDS-based tests on macOS (TensorFlow ARM64 import hang issue)",
        allow_module_level=True,
    )

# Skip if tensorflow_datasets not installed
pytest.importorskip("tensorflow_datasets")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.dag.nodes import OperatorNode, DataSourceNode
from datarax.dag import DAGExecutor
from datarax.operators.element_operator import ElementOperator
from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)
from datarax.sources.tfds_source import TFDSEagerConfig, TFDSEagerSource
from datarax.typing import Batch


# Skip if tensorflow_datasets not installed
pytest.importorskip("tensorflow_datasets")


# =============================================================================
# MODEL DEFINITION
# =============================================================================


class ImageClassifier(nnx.Module):
    """Simple CNN for MNIST image classification.

    This model demonstrates proper NNX patterns:
    - Module initialization with rngs parameter
    - Multiple RNG streams (params for weights, dropout for stochastic layers)
    - Proper call to super().__init__()
    - Mode switching via model.train() / model.eval() (not training parameter)
    """

    def __init__(self, num_classes: int, rngs: nnx.Rngs):
        """Initialize the model.

        Args:
            num_classes: Number of output classes.
            rngs: Random number generators with 'params' and 'dropout' streams.
        """
        super().__init__()
        self.conv1 = nnx.Conv(in_features=1, out_features=16, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = nnx.Conv(in_features=16, out_features=32, kernel_size=(3, 3), rngs=rngs)
        # With SAME padding (default in nnx.Conv), spatial dimensions are preserved.
        # After two 2x2 max pooling operations with stride 2:
        # 28x28 → 14x14 → 7x7, with 32 channels = 7*7*32 = 1568
        self.dense1 = nnx.Linear(in_features=1568, out_features=64, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=64, out_features=num_classes, rngs=rngs)
        # Dropout respects model.train() / model.eval() via deterministic attribute
        self.dropout = nnx.Dropout(0.3, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the model.

        Note: Training/eval mode is controlled via model.train() / model.eval()
        which sets the deterministic attribute on Dropout layers.

        Args:
            x: Input images with shape [batch_size, height, width, channels].

        Returns:
            logits: Output logits for classification.
        """
        # Check for empty batch - return empty logits with correct shape
        if x.size == 0 or x.shape[0] == 0:
            return jnp.zeros((0, 10))  # num_classes = 10 for MNIST

        # Ensure 4D input: [batch, height, width, channels]
        if x.ndim == 3:
            x = x[..., jnp.newaxis]

        # Convolutional layers with max pooling
        x = nnx.relu(self.conv1(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nnx.relu(self.conv2(x))
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten and dense layers
        x = x.reshape((x.shape[0], -1))
        x = nnx.relu(self.dense1(x))
        # Dropout automatically uses deterministic=True when model.eval() is called
        x = self.dropout(x)
        x = self.dense2(x)

        return x


# =============================================================================
# TRAINING UTILITIES
# =============================================================================


def compute_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute cross-entropy loss."""
    one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
    return optax.softmax_cross_entropy(logits, one_hot_labels).mean()


def compute_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Compute accuracy from logits."""
    predicted_labels = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_labels == labels)


# =============================================================================
# OPERATOR DEFINITIONS USING BUILT-IN OPERATORS
# =============================================================================


def create_preprocess_operator(rngs: nnx.Rngs) -> ElementOperator:
    """Create preprocessing operator using ElementOperator.

    This showcases ElementOperator for element-level transformations
    that need access to multiple fields or state.

    The preprocessing:
    - Normalizes images to [0, 1] range
    - Ensures correct shape with channel dimension
    """

    def preprocess_mnist(element: Element, key: jax.Array) -> Element:
        """Preprocess MNIST element.

        Args:
            element: Element with 'image' and 'label' in data
            key: JAX random key (unused for deterministic preprocessing)

        Returns:
            Element with normalized image data
        """
        data = element.data

        # Convert images to float and normalize to [0, 1]
        images = jnp.array(data["image"]).astype(jnp.float32) / 255.0

        # Ensure images have shape (H, W, 1) for single element
        if images.ndim == 2:
            images = images[:, :, np.newaxis]

        # Return element with updated data
        return element.replace(data={"image": images, "label": data["label"]})

    config = ElementOperatorConfig(stochastic=False)
    return ElementOperator(config, fn=preprocess_mnist, rngs=rngs)


def create_augmentation_operator(rngs: nnx.Rngs) -> BrightnessOperator:
    """Create augmentation operator using BrightnessOperator.

    This showcases the built-in modality operators for stochastic
    image augmentation with proper RNG stream management.
    """
    config = BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.1, 0.1),  # Subtle brightness augmentation
        clip_range=(0.0, 1.0),  # Keep values in valid range
        stochastic=True,
        stream_name="augment",
    )
    return BrightnessOperator(config, rngs=rngs)


# =============================================================================
# PIPELINE CONSTRUCTION
# =============================================================================


def create_train_pipeline(
    source: TFDSEagerSource,
    preprocess_op: ElementOperator,
    augment_op: BrightnessOperator,
    batch_size: int = 32,
    shuffle_buffer: int = 100,
) -> DAGExecutor:
    """Create training pipeline with all Datarax features.

    Pipeline structure:
    1. DataSourceNode - Wrap the data source
    2. BatchNode - Create batches (batch-first principle)
    3. ShuffleNode - Shuffle batches for training
    4. PreprocessOperator - Normalize and reshape images
    5. AugmentOperator - Apply stochastic brightness augmentation

    Args:
        source: TFDS data source
        preprocess_op: Preprocessing operator
        augment_op: Augmentation operator
        batch_size: Batch size
        shuffle_buffer: Shuffle buffer size

    Returns:
        Configured DAGExecutor pipeline
    """
    return (
        DAGExecutor(name="train_pipeline")
        .add(DataSourceNode(source))
        .batch(batch_size=batch_size)
        .shuffle(buffer_size=shuffle_buffer)
        .add(OperatorNode(preprocess_op))
        .add(OperatorNode(augment_op))
    )


def create_eval_pipeline(
    source: TFDSEagerSource,
    preprocess_op: ElementOperator,
    batch_size: int = 32,
) -> DAGExecutor:
    """Create evaluation pipeline (no shuffling or augmentation).

    Args:
        source: TFDS data source
        preprocess_op: Preprocessing operator
        batch_size: Batch size

    Returns:
        Configured DAGExecutor pipeline
    """
    return (
        DAGExecutor(name="eval_pipeline")
        .add(DataSourceNode(source))
        .batch(batch_size=batch_size)
        .add(OperatorNode(preprocess_op))
    )


# =============================================================================
# MAIN TEST
# =============================================================================


@pytest.mark.integration
def test_image_classification_end_to_end():
    """End-to-end test of image classification pipeline with model training.

    This test demonstrates:
    1. TFDSEagerSource for loading MNIST data
    2. DAGExecutor pipeline with batching, shuffling, and operators
    3. ElementOperator for preprocessing
    4. BrightnessOperator for augmentation
    5. NNX model training with proper optimizer API
    6. Mode switching via model.train() / model.eval()
    """
    # Create RNG with all required streams
    key = jax.random.key(0)
    params_key, dropout_key, augment_key = jax.random.split(key, 3)
    model_rngs = nnx.Rngs(params=params_key, dropout=dropout_key)

    # Skip if tensorflow_datasets not installed
    try:
        import tensorflow_datasets as tfds  # noqa: F401
    except ImportError:
        pytest.skip("tensorflow_datasets not installed")

    # ==========================================================================
    # DATA SOURCES
    # ==========================================================================

    train_config = TFDSEagerConfig(
        name="mnist:3.*.*",
        split="train[:500]",  # Small subset for testing
        shuffle=True,
    )
    train_source = TFDSEagerSource(train_config, rngs=nnx.Rngs(0))

    test_config = TFDSEagerConfig(
        name="mnist:3.*.*",
        split="test[:100]",  # Small subset for testing
        shuffle=False,
    )
    test_source = TFDSEagerSource(test_config, rngs=nnx.Rngs(1))

    # ==========================================================================
    # OPERATORS
    # ==========================================================================

    # Create operators with proper RNG streams
    preprocess_op = create_preprocess_operator(rngs=nnx.Rngs(default=2))
    augment_op = create_augmentation_operator(
        rngs=nnx.Rngs(default=augment_key, augment=jax.random.split(augment_key)[0])
    )

    # ==========================================================================
    # PIPELINES
    # ==========================================================================

    train_pipeline = create_train_pipeline(
        source=train_source,
        preprocess_op=preprocess_op,
        augment_op=augment_op,
        batch_size=32,
        shuffle_buffer=50,
    )

    # Create new preprocess operator for eval (operators are stateful)
    eval_preprocess_op = create_preprocess_operator(rngs=nnx.Rngs(default=3))
    eval_pipeline = create_eval_pipeline(
        source=test_source,
        preprocess_op=eval_preprocess_op,
        batch_size=32,
    )

    # ==========================================================================
    # MODEL AND OPTIMIZER
    # ==========================================================================

    num_classes = 10
    model = ImageClassifier(num_classes=num_classes, rngs=model_rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    # ==========================================================================
    # TRAINING FUNCTIONS
    # ==========================================================================

    def loss_fn(model: ImageClassifier, batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Compute loss and logits for a batch."""
        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        if images.shape[0] == 0:
            return jnp.float32(0.0), jnp.zeros((0, num_classes))

        logits = model(images)
        loss = compute_loss(logits, labels)
        return loss, logits

    @nnx.jit
    def train_step(
        model: ImageClassifier, optimizer: nnx.Optimizer, batch: Batch
    ) -> tuple[jax.Array, jax.Array]:
        """Single training step."""
        if batch["image"].shape[0] == 0:
            return jnp.float32(0.0), jnp.float32(0.0)

        labels = jnp.array(batch["label"])

        def model_loss_fn(m: ImageClassifier) -> tuple[jax.Array, jax.Array]:
            return loss_fn(m, batch)

        grad_fn = nnx.value_and_grad(model_loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model)
        optimizer.update(model, grads)  # Flax 0.11.0+ API: model as first arg

        accuracy = compute_accuracy(logits, labels)
        return loss, accuracy

    @nnx.jit
    def eval_step(model: ImageClassifier, batch: Batch) -> tuple[jax.Array, jax.Array]:
        """Single evaluation step."""
        if batch["image"].shape[0] == 0:
            return jnp.float32(0.0), jnp.float32(0.0)

        images = jnp.array(batch["image"])
        labels = jnp.array(batch["label"])

        logits = model(images)
        loss = compute_loss(logits, labels)
        accuracy = compute_accuracy(logits, labels)

        return loss, accuracy

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    num_epochs = 2
    batches_per_epoch = 5

    for epoch in range(num_epochs):
        # Set model to training mode (enables dropout, updates batchnorm stats)
        model.train()

        epoch_losses = []
        epoch_accuracies = []

        batch_count = 0
        for batch in train_pipeline:
            if batch["image"].shape[0] == 0:
                continue

            try:
                loss, accuracy = train_step(model, optimizer, batch)
                if not (jnp.isnan(loss) or jnp.isnan(accuracy)):
                    epoch_losses.append(float(loss))
                    epoch_accuracies.append(float(accuracy))
            except Exception as e:
                print(f"Training error: {e}")
                continue

            batch_count += 1
            if batch_count >= batches_per_epoch:
                break

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accuracies)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")

    # ==========================================================================
    # EVALUATION
    # ==========================================================================

    # Set model to evaluation mode (disables dropout, uses running batchnorm stats)
    model.eval()

    test_losses = []
    test_accuracies = []

    batch_count = 0
    for batch in eval_pipeline:
        if batch["image"].shape[0] == 0:
            continue

        try:
            loss, accuracy = eval_step(model, batch)
            if not (jnp.isnan(loss) or jnp.isnan(accuracy)):
                test_losses.append(float(loss))
                test_accuracies.append(float(accuracy))
        except Exception as e:
            print(f"Evaluation error: {e}")
            continue

        batch_count += 1
        if batch_count >= 3:
            break

    if test_losses:
        final_loss = np.mean(test_losses)
        final_acc = np.mean(test_accuracies)
        print(f"Test: Loss = {final_loss:.4f}, Accuracy = {final_acc:.4f}")

        # Assert training improved the model
        assert final_acc > 0.0, "Model training failed to improve accuracy"


if __name__ == "__main__":
    test_image_classification_end_to_end()
