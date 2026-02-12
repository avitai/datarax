"""Deterministic synthetic data generation for all benchmark modalities.

Uses ``np.random.default_rng(seed)`` for thread-safe, deterministic output.
Same seed + same calls = byte-identical results across runs.

Design ref: Section 6.3 of the benchmark report.
"""

from __future__ import annotations

import numpy as np


class SyntheticDataGenerator:
    """Deterministic synthetic data for all modalities.

    Args:
        seed: Random seed for reproducibility (default 42).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def images(
        self,
        n: int,
        h: int,
        w: int,
        c: int = 3,
        dtype: str = "uint8",
    ) -> np.ndarray:
        """Generate synthetic images with realistic pixel distributions.

        Args:
            n: Number of images.
            h: Height in pixels.
            w: Width in pixels.
            c: Number of channels (default 3 for RGB).
            dtype: 'uint8' for [0, 255] integers, 'float32' for standard normal.

        Returns:
            Array of shape ``(n, h, w, c)``.
        """
        if dtype == "uint8":
            return self.rng.integers(0, 256, (n, h, w, c), dtype=np.uint8)
        return self.rng.standard_normal((n, h, w, c)).astype(np.float32)

    def token_sequences(
        self,
        n: int,
        seq_len: int,
        vocab_size: int = 32000,
    ) -> np.ndarray:
        """Generate pre-tokenized integer sequences.

        Args:
            n: Number of sequences.
            seq_len: Sequence length.
            vocab_size: Vocabulary size (tokens are in [0, vocab_size)).

        Returns:
            Array of shape ``(n, seq_len)`` with dtype int32.
        """
        return self.rng.integers(0, vocab_size, (n, seq_len), dtype=np.int32)

    def variable_length_tokens(
        self,
        n: int,
        min_len: int = 10,
        max_len: int = 512,
        vocab_size: int = 32000,
    ) -> list[np.ndarray]:
        """Generate variable-length token sequences.

        Args:
            n: Number of sequences.
            min_len: Minimum sequence length.
            max_len: Maximum sequence length (inclusive).
            vocab_size: Vocabulary size.

        Returns:
            List of n arrays, each with dtype int32 and varying length.
        """
        lengths = self.rng.integers(min_len, max_len + 1, n)
        return [self.rng.integers(0, vocab_size, (length,), dtype=np.int32) for length in lengths]

    def tabular(
        self,
        n: int,
        features: int,
        dtype: str = "float32",
    ) -> np.ndarray:
        """Generate dense tabular data from standard normal distribution.

        Args:
            n: Number of rows.
            features: Number of feature columns.
            dtype: Output dtype string.

        Returns:
            Array of shape ``(n, features)``.
        """
        return self.rng.standard_normal((n, features)).astype(dtype)

    def sparse_features(
        self,
        n: int,
        num_dense: int = 13,
        num_sparse: int = 26,
        embedding_sizes: list[int] | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Generate DLRM-style mixed dense + sparse features.

        Args:
            n: Number of samples.
            num_dense: Number of dense float features.
            num_sparse: Number of sparse categorical features.
            embedding_sizes: Max value for each sparse feature. Defaults to
                [1000] * num_sparse.

        Returns:
            Tuple of (dense_array, list_of_sparse_arrays).
        """
        dense = self.rng.standard_normal((n, num_dense)).astype(np.float32)
        if embedding_sizes is None:
            embedding_sizes = [1000] * num_sparse
        sparse = [self.rng.integers(0, s, n, dtype=np.int64) for s in embedding_sizes]
        return dense, sparse

    def audio_waveforms(
        self,
        n: int,
        sample_rate: int = 16000,
        duration_sec: float = 5.0,
    ) -> np.ndarray:
        """Generate synthetic audio waveforms.

        Args:
            n: Number of waveforms.
            sample_rate: Samples per second.
            duration_sec: Duration in seconds.

        Returns:
            Array of shape ``(n, sample_rate * duration_sec)``.
        """
        length = int(sample_rate * duration_sec)
        return self.rng.standard_normal((n, length)).astype(np.float32)

    def volumes_3d(self, n: int, d: int, h: int, w: int) -> np.ndarray:
        """Generate 3D medical-imaging-style volumes.

        Args:
            n: Number of volumes.
            d: Depth (number of slices).
            h: Height.
            w: Width.

        Returns:
            Array of shape ``(n, d, h, w)`` with dtype float32.
        """
        return self.rng.standard_normal((n, d, h, w)).astype(np.float32)

    def image_text_pairs(
        self,
        n: int,
        img_shape: tuple[int, ...] = (224, 224, 3),
        text_len: int = 77,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate CLIP-style image-text pairs.

        Args:
            n: Number of pairs.
            img_shape: Image shape (h, w, c).
            text_len: Token sequence length per text.

        Returns:
            Tuple of (images_float32, token_sequences_int32).
        """
        images = self.images(n, img_shape[0], img_shape[1], img_shape[2], dtype="float32")
        tokens = self.token_sequences(n, text_len)
        return images, tokens
