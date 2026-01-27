"""Example demonstrating HuggingFace Datasets integration with model training.

This example shows how to use Datarax with HuggingFace Datasets to create a
data pipeline for model training, complete with checkpointing and evaluation.
"""

import os

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from datarax.core import Pipeline
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import HFEagerConfig, HFEagerSource


# Define a simple text classifier model
class TextClassifier(nnx.Module):
    """Simple text classifier with embedding layer and MLP layers."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, *, rngs: nnx.Rngs):
        """Initialize the text classifier."""
        super().__init__()

        self.embedding = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.dense1 = nnx.Linear(in_features=embed_dim, out_features=hidden_dim, rngs=rngs)
        # Create Dropout layer
        self.dropout = nnx.Dropout(rate=0.1)
        self.dense2 = nnx.Linear(in_features=hidden_dim, out_features=num_classes, rngs=rngs)

    def __call__(self, x, training=False):
        """Forward pass of the text classifier."""
        x = self.embedding(x)
        # Average over token dimension
        x = jnp.mean(x, axis=1)
        x = nnx.relu(self.dense1(x))
        if training:
            # Apply dropout during training
            dropout_rng = nnx.Rngs(dropout=jax.random.key(0))
            x = self.dropout(x, deterministic=not training, rngs=dropout_rng)
        x = self.dense2(x)
        return x


def tokenize_glue_sst2(examples, key=None):
    """Tokenizer for SST-2 sentiment analysis dataset.

    Args:
        examples: Input examples dictionary
        key: Optional RNG key (unused for deterministic tokenization)

    Returns:
        Tokenized examples
    """
    # Simple vocabulary for demo purposes
    # In real applications, use a proper tokenizer like SentencePiece or WordPiece
    vocab = {}
    # Add special tokens
    vocab["<pad>"] = 0
    vocab["<unk>"] = 1

    # Add common words (simplified for demo)
    common_words = [
        "the",
        "a",
        "an",
        "and",
        "is",
        "was",
        "it",
        "to",
        "of",
        "in",
        "movie",
        "film",
        "great",
        "good",
        "bad",
        "terrible",
        "excellent",
        "poor",
        "amazing",
        "awful",
        "wonderful",
        "horrible",
        "best",
        "worst",
        "like",
        "love",
        "hate",
        "enjoy",
        "boring",
        "exciting",
        "interesting",
        "dull",
        "fun",
        "not",
        "very",
        "really",
        "quite",
        "so",
        "much",
        "this",
    ]

    for i, word in enumerate(common_words):
        vocab[word] = i + 2  # Start after special tokens

    # Ensure inputs are lists
    if isinstance(examples["sentence"], list):
        sentences = examples["sentence"]
    else:
        sentences = [examples["sentence"]]

    # Get labels
    if "label" in examples:
        if isinstance(examples["label"], list):
            labels = examples["label"]
        else:
            labels = [examples["label"]]
    else:
        # Default labels if not found (should not happen)
        labels = [0] * len(sentences)

    # Tokenize each sentence
    tokenized_inputs = []
    for sentence in sentences:
        # Convert to lowercase, split by space, and map to vocab ids
        tokens = []
        for word in sentence.lower().split()[:50]:  # Limit to 50 words
            tokens.append(vocab.get(word, vocab["<unk>"]))

        # Truncate or pad to fixed length (30 tokens)
        if len(tokens) > 30:
            tokens = tokens[:30]
        else:
            tokens = tokens + [vocab["<pad>"]] * (30 - len(tokens))

        tokenized_inputs.append(tokens)

    # Return tokenized inputs with labels
    return {"tokens": tokenized_inputs, "label": labels}


def compute_loss(logits, labels):
    """Compute cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, num_classes=2)
    return optax.softmax_cross_entropy(logits, one_hot).mean()


def compute_accuracy(logits, labels):
    """Compute accuracy."""
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


def main():
    """Run the example."""
    print("Datarax HuggingFace Datasets Integration Example")
    print("===============================================")

    # Check for datasets package
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Error: HuggingFace datasets package not installed.")
        print("Install with: pip install datasets")
        return

    print("\nLoading SST-2 dataset...")

    # Create data source for SST-2 dataset using config-based API
    train_config = HFEagerConfig(
        name="glue",
        split="train[:sst2]",  # SST-2 subset of GLUE
        streaming=False,
        shuffle=True,
        seed=42,
    )
    train_source = HFEagerSource(train_config, rngs=nnx.Rngs(0))

    val_config = HFEagerConfig(
        name="glue",
        split="validation[:sst2]",
        streaming=False,
    )
    val_source = HFEagerSource(val_config, rngs=nnx.Rngs(1))

    print("Creating data streams...")

    # Create ElementOperator for tokenization
    tokenizer_config = ElementOperatorConfig(stochastic=False)
    tokenizer = ElementOperator(tokenizer_config, fn=tokenize_glue_sst2, rngs=nnx.Rngs(0))

    # Create data streams with transformations using the fluent API
    train_stream = Pipeline(train_source).map(tokenizer).batch(batch_size=32)

    val_stream = Pipeline(val_source).map(tokenizer).batch(batch_size=64)

    print("Initializing model...")

    # Model hyperparameters
    vocab_size = 42  # Size of our toy vocabulary
    embed_dim = 64
    hidden_dim = 128
    num_classes = 2  # Binary sentiment classification

    # Create model
    param_key = jax.random.key(0)
    model_rngs = nnx.Rngs(params=param_key)

    model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        rngs=model_rngs,
    )

    # Create optimizer with nnx
    learning_rate = 1e-3
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    # Create metrics
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    # Define the loss function
    def loss_fn(model, batch):
        tokens = jnp.array(batch["tokens"])
        labels = jnp.array(batch["label"])
        logits = model(tokens, training=True)
        loss = compute_loss(logits, labels)
        return loss, logits

    @nnx.jit
    def train_step(model, optimizer, metrics, batch):
        """Single training step using nnx and optax."""
        # Use value_and_grad to compute loss and gradients
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)

        # Update metrics (in-place)
        metrics.update(loss=loss, logits=logits, labels=jnp.array(batch["label"]))

        # Update optimizer (which updates the model)
        optimizer.update(model, grads)

    @nnx.jit
    def eval_step(model, metrics, batch):
        """Evaluate the model on a batch."""
        tokens = jnp.array(batch["tokens"])
        labels = jnp.array(batch["label"])

        logits = model(tokens, training=False)
        loss = compute_loss(logits, labels)

        # Update metrics (in-place)
        metrics.update(loss=loss, logits=logits, labels=labels)

    # Use repository-based directory for checkpointing instead of a temporary one
    temp_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "temp", "hf_model_training"
    )
    os.makedirs(temp_dir, exist_ok=True)
    ckpt_dir = os.path.abspath(temp_dir)
    print(f"Using directory: {ckpt_dir}")

    # Training loop
    num_epochs = 3
    steps_per_epoch = 10  # Limit for example purposes

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Reset metrics for training
        metrics.reset()

        # Training phase
        print("Training...")

        # Process a fixed number of batches per epoch
        batch_count = 0
        for batch in train_stream:
            train_step(model, optimizer, metrics, batch)

            batch_count += 1
            if batch_count % 2 == 0:
                # Print intermediate metrics
                metric_values = metrics.compute()
                print(
                    f"  Step {batch_count}/{steps_per_epoch}, "
                    f"Loss: {metric_values['loss']:.4f}, "
                    f"Accuracy: {metric_values['accuracy']:.4f}"
                )

            if batch_count >= steps_per_epoch:
                break

        # Print training metrics
        train_metrics = metrics.compute()
        print(
            f"  Training summary - Loss: {train_metrics['loss']:.4f}, "
            f"Accuracy: {train_metrics['accuracy']:.4f}"
        )

        # Reset metrics for validation
        metrics.reset()

        # Validation phase
        print("Validating...")

        # Use a fixed number of batches for validation
        val_batch_count = 0
        for batch in val_stream:
            eval_step(model, metrics, batch)
            val_batch_count += 1
            if val_batch_count >= 5:  # Limit for example purposes
                break

        # Print validation results
        val_metrics = metrics.compute()
        print(
            f"  Validation summary - Loss: {val_metrics['loss']:.4f}, "
            f"Accuracy: {val_metrics['accuracy']:.4f}"
        )

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
