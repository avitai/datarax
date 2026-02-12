"""Tests for SyntheticDataGenerator.

TDD: Write tests first, then implement.
Design ref: Section 6.3 of the benchmark report.
"""

import numpy as np

from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator


class TestSyntheticDataGeneratorInit:
    """Test initialization and determinism."""

    def test_default_seed(self):
        """Test default seed is 42."""
        gen = SyntheticDataGenerator()
        # Two generators with same seed produce identical output
        gen2 = SyntheticDataGenerator(seed=42)
        np.testing.assert_array_equal(gen.images(5, 32, 32), gen2.images(5, 32, 32))

    def test_custom_seed(self):
        """Test custom seed produces different output."""
        gen1 = SyntheticDataGenerator(seed=1)
        gen2 = SyntheticDataGenerator(seed=2)
        # Different seeds → different data
        assert not np.array_equal(gen1.images(5, 32, 32), gen2.images(5, 32, 32))

    def test_deterministic_across_calls(self):
        """Test same seed + same calls = identical results."""
        gen1 = SyntheticDataGenerator(seed=123)
        gen2 = SyntheticDataGenerator(seed=123)
        np.testing.assert_array_equal(
            gen1.token_sequences(10, 128),
            gen2.token_sequences(10, 128),
        )


class TestImages:
    """Test images() generator."""

    def test_uint8_shape_and_dtype(self):
        """Test uint8 image generation."""
        gen = SyntheticDataGenerator()
        imgs = gen.images(10, 256, 256, c=3, dtype="uint8")
        assert imgs.shape == (10, 256, 256, 3)
        assert imgs.dtype == np.uint8

    def test_uint8_value_range(self):
        """Test uint8 images are in [0, 255]."""
        gen = SyntheticDataGenerator()
        imgs = gen.images(100, 32, 32, dtype="uint8")
        assert imgs.min() >= 0
        assert imgs.max() <= 255

    def test_float32_shape_and_dtype(self):
        """Test float32 image generation."""
        gen = SyntheticDataGenerator()
        imgs = gen.images(10, 64, 64, c=1, dtype="float32")
        assert imgs.shape == (10, 64, 64, 1)
        assert imgs.dtype == np.float32

    def test_default_channels(self):
        """Test default 3 channels."""
        gen = SyntheticDataGenerator()
        imgs = gen.images(5, 32, 32)
        assert imgs.shape == (5, 32, 32, 3)

    def test_single_image(self):
        """Test generating a single image."""
        gen = SyntheticDataGenerator()
        imgs = gen.images(1, 16, 16)
        assert imgs.shape == (1, 16, 16, 3)


class TestTokenSequences:
    """Test token_sequences() generator."""

    def test_shape_and_dtype(self):
        """Test token sequence shape and int32 dtype."""
        gen = SyntheticDataGenerator()
        tokens = gen.token_sequences(20, 512)
        assert tokens.shape == (20, 512)
        assert tokens.dtype == np.int32

    def test_value_range_default_vocab(self):
        """Test tokens are in [0, vocab_size)."""
        gen = SyntheticDataGenerator()
        tokens = gen.token_sequences(100, 128, vocab_size=32000)
        assert tokens.min() >= 0
        assert tokens.max() < 32000

    def test_custom_vocab_size(self):
        """Test custom vocabulary size."""
        gen = SyntheticDataGenerator()
        tokens = gen.token_sequences(50, 64, vocab_size=100)
        assert tokens.max() < 100


class TestVariableLengthTokens:
    """Test variable_length_tokens() generator."""

    def test_returns_list_of_arrays(self):
        """Test return type is list of numpy arrays."""
        gen = SyntheticDataGenerator()
        sequences = gen.variable_length_tokens(10)
        assert isinstance(sequences, list)
        assert len(sequences) == 10
        assert all(isinstance(s, np.ndarray) for s in sequences)

    def test_lengths_within_bounds(self):
        """Test all sequences have lengths within [min_len, max_len]."""
        gen = SyntheticDataGenerator()
        sequences = gen.variable_length_tokens(50, min_len=10, max_len=100)
        for seq in sequences:
            assert 10 <= len(seq) <= 100

    def test_variable_lengths(self):
        """Test sequences have varying lengths."""
        gen = SyntheticDataGenerator()
        sequences = gen.variable_length_tokens(50, min_len=10, max_len=500)
        lengths = {len(s) for s in sequences}
        assert len(lengths) > 1  # Not all same length

    def test_dtype(self):
        """Test token dtype is int32."""
        gen = SyntheticDataGenerator()
        sequences = gen.variable_length_tokens(5)
        for seq in sequences:
            assert seq.dtype == np.int32


class TestTabular:
    """Test tabular() generator."""

    def test_shape_and_dtype(self):
        """Test tabular data shape and dtype."""
        gen = SyntheticDataGenerator()
        data = gen.tabular(1000, 50)
        assert data.shape == (1000, 50)
        assert data.dtype == np.float32

    def test_custom_dtype(self):
        """Test custom dtype."""
        gen = SyntheticDataGenerator()
        data = gen.tabular(100, 10, dtype="float64")
        assert data.dtype == np.float64

    def test_standard_normal_distribution(self):
        """Test data follows approximate standard normal."""
        gen = SyntheticDataGenerator()
        data = gen.tabular(10000, 1)
        assert abs(data.mean()) < 0.1
        assert abs(data.std() - 1.0) < 0.1


class TestSparseFeatures:
    """Test sparse_features() generator."""

    def test_returns_dense_and_sparse(self):
        """Test returns (dense, sparse) tuple."""
        gen = SyntheticDataGenerator()
        dense, sparse = gen.sparse_features(100)
        assert isinstance(dense, np.ndarray)
        assert isinstance(sparse, list)

    def test_dense_shape(self):
        """Test dense features shape."""
        gen = SyntheticDataGenerator()
        dense, _ = gen.sparse_features(100, num_dense=13)
        assert dense.shape == (100, 13)
        assert dense.dtype == np.float32

    def test_sparse_count(self):
        """Test number of sparse features."""
        gen = SyntheticDataGenerator()
        _, sparse = gen.sparse_features(100, num_sparse=26)
        assert len(sparse) == 26

    def test_sparse_dtype(self):
        """Test sparse feature dtype is int64."""
        gen = SyntheticDataGenerator()
        _, sparse = gen.sparse_features(100)
        for s in sparse:
            assert s.dtype == np.int64

    def test_custom_embedding_sizes(self):
        """Test custom embedding sizes."""
        gen = SyntheticDataGenerator()
        sizes = [500, 1000, 2000]
        _, sparse = gen.sparse_features(100, num_sparse=3, embedding_sizes=sizes)
        assert len(sparse) == 3
        for s, size in zip(sparse, sizes):
            assert s.max() < size

    def test_default_embedding_sizes(self):
        """Test default embedding sizes of 1000."""
        gen = SyntheticDataGenerator()
        _, sparse = gen.sparse_features(200, num_sparse=5)
        for s in sparse:
            assert s.max() < 1000


class TestAudioWaveforms:
    """Test audio_waveforms() generator."""

    def test_shape(self):
        """Test audio waveform shape."""
        gen = SyntheticDataGenerator()
        audio = gen.audio_waveforms(10, sample_rate=16000, duration_sec=5.0)
        expected_length = int(16000 * 5.0)
        assert audio.shape == (10, expected_length)

    def test_dtype(self):
        """Test audio dtype is float32."""
        gen = SyntheticDataGenerator()
        audio = gen.audio_waveforms(5)
        assert audio.dtype == np.float32

    def test_custom_parameters(self):
        """Test custom sample rate and duration."""
        gen = SyntheticDataGenerator()
        audio = gen.audio_waveforms(3, sample_rate=44100, duration_sec=2.0)
        expected_length = int(44100 * 2.0)
        assert audio.shape == (3, expected_length)


class TestVolumes3D:
    """Test volumes_3d() generator."""

    def test_shape(self):
        """Test 3D volume shape."""
        gen = SyntheticDataGenerator()
        vols = gen.volumes_3d(5, d=64, h=128, w=128)
        assert vols.shape == (5, 64, 128, 128)

    def test_dtype(self):
        """Test volume dtype is float32."""
        gen = SyntheticDataGenerator()
        vols = gen.volumes_3d(2, d=16, h=16, w=16)
        assert vols.dtype == np.float32


class TestImageTextPairs:
    """Test image_text_pairs() generator."""

    def test_returns_images_and_tokens(self):
        """Test returns (images, tokens) tuple."""
        gen = SyntheticDataGenerator()
        images, tokens = gen.image_text_pairs(10)
        assert isinstance(images, np.ndarray)
        assert isinstance(tokens, np.ndarray)

    def test_image_shape(self):
        """Test image shape matches img_shape parameter."""
        gen = SyntheticDataGenerator()
        images, _ = gen.image_text_pairs(5, img_shape=(224, 224, 3))
        assert images.shape == (5, 224, 224, 3)

    def test_token_shape(self):
        """Test token shape matches text_len parameter."""
        gen = SyntheticDataGenerator()
        _, tokens = gen.image_text_pairs(5, text_len=77)
        assert tokens.shape == (5, 77)

    def test_image_dtype_is_float(self):
        """Test image-text pair images are float32."""
        gen = SyntheticDataGenerator()
        images, _ = gen.image_text_pairs(3)
        assert images.dtype == np.float32

    def test_custom_shape_and_length(self):
        """Test custom image shape and text length."""
        gen = SyntheticDataGenerator()
        images, tokens = gen.image_text_pairs(4, img_shape=(64, 64, 1), text_len=128)
        assert images.shape == (4, 64, 64, 1)
        assert tokens.shape == (4, 128)
