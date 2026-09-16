"""Train a text classifier on GLUE SST-2 loaded through HuggingFace Datasets.

JAX arrays cannot hold strings, so a text dataset is tokenized before it is batched:
``HFEagerSource`` loads and filters the columns, the sentences become fixed-length token
ids once at load time, and a ``MemorySource`` batches the token ids and labels through a
``Pipeline`` for a Flax NNX classifier that trains for three epochs with validation after
each one.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources import HFEagerConfig, HFEagerSource, MemorySource, MemorySourceConfig


# A toy vocabulary keeps the example self-contained; a real application uses a trained
# tokenizer (SentencePiece, WordPiece) and a vocabulary of tens of thousands of entries.
COMMON_WORDS = (
    "the", "a", "an", "and", "is", "was", "it", "to", "of", "in",
    "movie", "film", "great", "good", "bad", "terrible", "excellent", "poor", "amazing",
    "awful", "wonderful", "horrible", "best", "worst", "like", "love", "hate", "enjoy",
    "boring", "exciting", "interesting", "dull", "fun", "not", "very", "really", "quite",
    "so", "much", "this",
)  # fmt: skip
PAD, UNK = 0, 1
VOCAB = {word: index + 2 for index, word in enumerate(COMMON_WORDS)}
VOCAB_SIZE = len(COMMON_WORDS) + 2
MAX_TOKENS = 30

NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64


class TextClassifier(nnx.Module):
    """Mean-pooled token embeddings followed by a two-layer MLP."""

    def __init__(
        self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int, *, rngs: nnx.Rngs
    ):
        """Build the embedding, the hidden layer, its dropout and the output layer."""
        super().__init__()
        self.embedding = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        self.dense1 = nnx.Linear(in_features=embed_dim, out_features=hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.dense2 = nnx.Linear(in_features=hidden_dim, out_features=num_classes, rngs=rngs)

    def __call__(self, tokens: jax.Array, *, training: bool = False) -> jax.Array:
        """Class logits for a batch of token ids."""
        x = jnp.mean(self.embedding(tokens), axis=1)
        x = nnx.relu(self.dense1(x))
        x = self.dropout(x, deterministic=not training)
        return self.dense2(x)


def tokenize(sentences: Sequence[str]) -> np.ndarray:
    """Map sentences to ``(len(sentences), MAX_TOKENS)`` int32 ids, truncated or padded."""
    ids = np.full((len(sentences), MAX_TOKENS), PAD, dtype=np.int32)
    for row, sentence in enumerate(sentences):
        words = sentence.lower().split()[:MAX_TOKENS]
        ids[row, : len(words)] = [VOCAB.get(word, UNK) for word in words]
    return ids


def load_sst2(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one SST-2 split and return its token ids and labels.

    GLUE is one dataset with many configurations; SST-2 is the ``sst2`` configuration,
    which ``datasets.load_dataset`` takes as its ``name`` argument. The eager source keeps
    a text column as Python strings for inspection, and only numeric columns can be batched,
    so the sentences are tokenized here.
    """
    source = HFEagerSource(
        HFEagerConfig(
            name="nyu-mll/glue",
            split=split,
            download_kwargs={"name": "sst2"},
            include_keys={"sentence", "label"},
        ),
        rngs=nnx.Rngs(0),
    )
    return tokenize(source.data["sentence"]), np.asarray(source.data["label"], dtype=np.int32)


def make_pipeline(
    tokens: np.ndarray, labels: np.ndarray, *, batch_size: int, shuffle: bool, seed: int
) -> Pipeline:
    """Batch token ids and labels from memory, shuffled per epoch when asked."""
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data={"tokens": tokens, "label": labels},
        rngs=nnx.Rngs(seed),
    )
    return Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(seed))


def loss_fn(model: TextClassifier, batch: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    """Softmax cross-entropy and the logits it was computed from."""
    logits = model(batch["tokens"], training=True)
    one_hot = jax.nn.one_hot(batch["label"], num_classes=2)
    return optax.softmax_cross_entropy(logits, one_hot).mean(), logits


@nnx.jit
def train_step(
    model: TextClassifier,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    batch: dict[str, jax.Array],
) -> None:
    """One gradient step; the metrics accumulate the batch's loss and accuracy."""
    (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch["label"])
    optimizer.update(model, grads)


@nnx.jit
def eval_step(model: TextClassifier, metrics: nnx.MultiMetric, batch: dict[str, jax.Array]) -> None:
    """Accumulate the batch's loss and accuracy without updating the model."""
    logits = model(batch["tokens"], training=False)
    one_hot = jax.nn.one_hot(batch["label"], num_classes=2)
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    metrics.update(loss=loss, logits=logits, labels=batch["label"])


def main() -> None:
    """Run the example."""
    print("Datarax HuggingFace Datasets Integration Example")
    print("===============================================")

    print("\nLoading SST-2 dataset...")
    train_tokens, train_labels = load_sst2("train")
    val_tokens, val_labels = load_sst2("validation")
    print(f"Train: {len(train_labels)} sentences, validation: {len(val_labels)} sentences")
    print(f"Tokens per sentence: {MAX_TOKENS}, vocabulary: {VOCAB_SIZE} entries")

    train_stream = make_pipeline(
        train_tokens, train_labels, batch_size=TRAIN_BATCH_SIZE, shuffle=True, seed=2
    )
    val_stream = make_pipeline(
        val_tokens, val_labels, batch_size=VAL_BATCH_SIZE, shuffle=False, seed=3
    )

    print("Initializing model...")
    model = TextClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=64,
        hidden_dim=128,
        num_classes=2,
        rngs=nnx.Rngs(params=0, dropout=1),
    )
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average("loss"),
    )

    for epoch in range(NUM_EPOCHS):
        metrics.reset()
        train_stream.reset()
        for batch in train_stream:
            train_step(model, optimizer, metrics, batch)
        train_metrics = metrics.compute()

        metrics.reset()
        val_stream.reset()
        for batch in val_stream:
            eval_step(model, metrics, batch)
        val_metrics = metrics.compute()

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}: "
            f"train loss {train_metrics['loss']:.4f}, accuracy {train_metrics['accuracy']:.4f} | "
            f"validation loss {val_metrics['loss']:.4f}, accuracy {val_metrics['accuracy']:.4f}"
        )

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
