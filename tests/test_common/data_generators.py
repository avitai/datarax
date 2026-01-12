"""Test data generators for Datarax tests."""

import os
import random

import numpy as np


def generate_image_data(
    num_samples: int = 100,
    image_height: int = 32,
    image_width: int = 32,
    num_channels: int = 3,
    num_classes: int = 10,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic image data for testing.

    Args:
        num_samples: Number of samples to generate.
        image_height: Height of the generated images.
        image_width: Width of the generated images.
        num_channels: Number of channels in the images.
        num_classes: Number of classes for labels.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with image data and labels.
    """
    np.random.seed(seed)
    images = np.random.rand(num_samples, image_height, image_width, num_channels).astype(np.float32)

    labels = np.random.randint(0, num_classes, size=(num_samples,))

    return {
        "image": images,
        "label": labels,
    }


def generate_text_data(
    num_samples: int = 100,
    vocab_size: int = 1000,
    max_seq_length: int = 50,
    num_classes: int = 2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic text data for testing.

    Args:
        num_samples: Number of samples to generate.
        vocab_size: Size of the vocabulary.
        max_seq_length: Maximum sequence length.
        num_classes: Number of classes for labels.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with tokenized text and labels.
    """
    np.random.seed(seed)

    # Generate sequences of token ids
    sequences = []
    sequence_lengths = []

    for _ in range(num_samples):
        seq_length = np.random.randint(10, max_seq_length + 1)
        sequence = np.random.randint(1, vocab_size, size=seq_length)

        # Pad to max_seq_length
        padded_sequence = np.zeros(max_seq_length, dtype=np.int32)
        padded_sequence[:seq_length] = sequence

        sequences.append(padded_sequence)
        sequence_lengths.append(seq_length)

    sequences_array = np.stack(sequences)

    # Generate labels
    labels = np.random.randint(0, num_classes, size=(num_samples,))

    return {
        "tokens": sequences_array,
        "length": np.array(sequence_lengths, dtype=np.int32),
        "label": labels,
    }


def generate_tabular_data(
    num_samples: int = 100,
    num_numerical_features: int = 8,
    num_categorical_features: int = 4,
    num_categories: int = 5,
    num_classes: int = 2,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic tabular data for testing.

    Args:
        num_samples: Number of samples to generate.
        num_numerical_features: Number of numerical features.
        num_categorical_features: Number of categorical features.
        num_categories: Number of categories per categorical feature.
        num_classes: Number of classes for labels.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with numerical features, categorical features, and labels.
    """
    np.random.seed(seed)

    numerical = np.random.randn(num_samples, num_numerical_features).astype(np.float32)

    categorical = np.random.randint(0, num_categories, size=(num_samples, num_categorical_features))

    labels = np.random.randint(0, num_classes, size=(num_samples,))

    return {
        "numerical": numerical,
        "categorical": categorical,
        "label": labels,
    }


def write_image_files(
    output_dir: str,
    num_samples: int = 100,
    image_height: int = 32,
    image_width: int = 32,
    num_channels: int = 3,
    num_classes: int = 10,
    seed: int = 42,
) -> tuple[list[str], list[int]]:
    """Generate image files for testing file-based data sources.

    Args:
        output_dir: Directory to write the files to.
        num_samples: Number of samples to generate.
        image_height: Height of the generated images.
        image_width: Width of the generated images.
        num_channels: Number of channels in the images.
        num_classes: Number of classes for labels.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (file_paths, labels).
    """
    try:
        import PIL.Image
    except ImportError as e:
        raise ImportError("PIL is required for write_image_files") from e

    np.random.seed(seed)
    random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    file_paths = []
    labels = []

    for i in range(num_samples):
        # Generate a random class label
        label = random.randint(0, num_classes - 1)  # nosec B311
        labels.append(label)

        # Create class directory if it doesn't exist
        class_dir = os.path.join(output_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)

        # Generate random image data
        if num_channels == 1:
            # Grayscale image
            img_data = np.random.randint(0, 256, size=(image_height, image_width)).astype(np.uint8)
            img = PIL.Image.fromarray(img_data, mode="L")
        else:
            # Color image
            img_data = np.random.randint(
                0, 256, size=(image_height, image_width, num_channels)
            ).astype(np.uint8)
            img = PIL.Image.fromarray(img_data, mode="RGB")

        # Save the image
        file_path = os.path.join(class_dir, f"image_{i}.png")
        img.save(file_path)
        file_paths.append(file_path)

    return file_paths, labels


def create_test_tfrecord(output_path: str, num_samples: int = 100, seed: int = 42) -> None:
    """Create TFRecord file for testing.

    Args:
        output_path: Path to save the TFRecord file.
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("TensorFlow is required for create_test_tfrecord") from e

    np.random.seed(seed)

    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(num_samples):
            # Generate random features
            feature = np.random.randn(10).astype(np.float32)

            # Generate random label
            label = np.random.randint(0, 5)

            # Create example
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "feature": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }
                )
            )

            # Write example
            writer.write(example.SerializeToString())


def create_npz_file(output_path: str, num_samples: int = 100, seed: int = 42) -> None:
    """Create NumPy .npz file for testing.

    Args:
        output_path: Path to save the .npz file.
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
    """
    np.random.seed(seed)

    features = np.random.randn(num_samples, 16).astype(np.float32)
    labels = np.random.randint(0, 5, size=(num_samples,))

    np.savez(output_path, features=features, labels=labels)
