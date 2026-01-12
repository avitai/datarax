"""Sample data generation utilities for examples.

This module provides functions to generate sample data for use in examples,
allowing examples to be self-contained and runnable without external datasets.
"""

import numpy as np


def create_mnist_like(
    num_samples: int = 1000,
    image_size: tuple[int, int] = (28, 28),
    num_classes: int = 10,
    seed: int = 42,
) -> dict:
    """Create MNIST-like sample data.

    Args:
        num_samples: Number of samples to generate.
        image_size: Height and width of images.
        num_classes: Number of classification classes.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'image' and 'label' keys.

    Example:
        >>> data = create_mnist_like(num_samples=100)
        >>> print(data['image'].shape)
        (100, 28, 28, 1)
    """
    rng = np.random.default_rng(seed)
    height, width = image_size
    image = rng.integers(0, 255, (num_samples, height, width, 1)).astype(np.float32)
    label = rng.integers(0, num_classes, (num_samples,)).astype(np.int32)
    return {"image": image, "label": label}


def create_imagenet_like(
    num_samples: int = 100,
    image_size: tuple[int, int] = (224, 224),
    num_classes: int = 1000,
    seed: int = 42,
) -> dict:
    """Create ImageNet-like sample data.

    Args:
        num_samples: Number of samples to generate.
        image_size: Height and width of images.
        num_classes: Number of classification classes.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'image' and 'label' keys.

    Example:
        >>> data = create_imagenet_like(num_samples=10)
        >>> print(data['image'].shape)
        (10, 224, 224, 3)
    """
    rng = np.random.default_rng(seed)
    height, width = image_size
    image = rng.integers(0, 255, (num_samples, height, width, 3)).astype(np.float32)
    label = rng.integers(0, num_classes, (num_samples,)).astype(np.int32)
    return {"image": image, "label": label}


def create_text_batch(
    num_samples: int = 500,
    max_length: int = 128,
    vocab_size: int = 30000,
    seed: int = 42,
) -> dict:
    """Create tokenized text-like sample data.

    Args:
        num_samples: Number of samples to generate.
        max_length: Maximum sequence length.
        vocab_size: Size of the vocabulary.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'label' keys.

    Example:
        >>> data = create_text_batch(num_samples=100)
        >>> print(data['input_ids'].shape)
        (100, 128)
    """
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(0, vocab_size, (num_samples, max_length)).astype(np.int32)
    # Create attention mask (all ones for simplicity)
    attention_mask = np.ones((num_samples, max_length), dtype=np.int32)
    label = rng.integers(0, 2, (num_samples,)).astype(np.int32)  # Binary classification
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}
