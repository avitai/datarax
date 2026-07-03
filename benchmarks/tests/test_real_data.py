"""Tests for the framework-neutral real-data provider.

RealDataProvider mirrors the SyntheticDataGenerator output contract: raw
numpy arrays that every adapter wraps its own way, so all frameworks see
byte-identical input. Unit tests mock the dataset loaders; integration
tests run only against locally cached datasets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pytest

from benchmarks.fixtures import real_data
from benchmarks.fixtures.real_data import (
    hash_tokenize,
    RealDataProvider,
    RealDataUnavailableError,
)
from benchmarks.tests.conftest import FAKE_CIFAR_COUNT


# ---------------------------------------------------------------------------
# hash_tokenize
# ---------------------------------------------------------------------------


class TestHashTokenize:
    """Deterministic hash-based word -> token-id mapping."""

    def test_dtype_and_range(self):
        """Token ids are int32 in [0, vocab_size)."""
        ids = hash_tokenize(["the", "quick", "brown", "fox"], vocab_size=32000)
        assert ids.dtype == np.int32
        assert ids.shape == (4,)
        assert ids.min() >= 0
        assert ids.max() < 32000

    def test_deterministic(self):
        """Same words always map to the same ids."""
        words = ["alpha", "beta", "gamma", "alpha"]
        first = hash_tokenize(words, vocab_size=1000)
        second = hash_tokenize(words, vocab_size=1000)
        np.testing.assert_array_equal(first, second)

    def test_repeated_word_same_id(self):
        """Word identity is preserved: repeated words share one id."""
        ids = hash_tokenize(["dog", "cat", "dog"], vocab_size=32000)
        assert ids[0] == ids[2]
        assert ids[0] != ids[1]

    def test_small_vocab_respected(self):
        """All ids stay below a small vocab_size."""
        words = [f"word{i}" for i in range(100)]
        ids = hash_tokenize(words, vocab_size=7)
        assert ids.max() < 7
        assert ids.min() >= 0

    def test_empty_input(self):
        """Empty word list yields an empty int32 array."""
        ids = hash_tokenize([], vocab_size=100)
        assert ids.shape == (0,)
        assert ids.dtype == np.int32


# ---------------------------------------------------------------------------
# Download guard configuration
# ---------------------------------------------------------------------------


class TestAllowDownload:
    """allow_download resolves from arg, then env var, then False."""

    def test_default_false(self, monkeypatch: pytest.MonkeyPatch):
        """Without env var or arg, downloads are disallowed."""
        monkeypatch.delenv(real_data.DOWNLOAD_ENV_VAR, raising=False)
        assert RealDataProvider().allow_download is False

    def test_env_var_enables(self, monkeypatch: pytest.MonkeyPatch):
        """DATARAX_BENCH_DOWNLOAD=1 enables downloads."""
        monkeypatch.setenv(real_data.DOWNLOAD_ENV_VAR, "1")
        assert RealDataProvider().allow_download is True

    def test_env_var_zero_disables(self, monkeypatch: pytest.MonkeyPatch):
        """DATARAX_BENCH_DOWNLOAD=0 keeps downloads disabled."""
        monkeypatch.setenv(real_data.DOWNLOAD_ENV_VAR, "0")
        assert RealDataProvider().allow_download is False

    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch):
        """An explicit constructor arg wins over the env var."""
        monkeypatch.setenv(real_data.DOWNLOAD_ENV_VAR, "1")
        assert RealDataProvider(allow_download=False).allow_download is False


# ---------------------------------------------------------------------------
# cifar10_images
# ---------------------------------------------------------------------------


class TestCifar10Images:
    """cifar10_images mirrors SyntheticDataGenerator.images (uint8 NHWC)."""

    def test_native_shape_and_dtype(self, fake_cifar: np.ndarray):
        """Native 32x32 output has synthetic-contract shape and dtype."""
        images = RealDataProvider(seed=1).cifar10_images(10)
        assert images.shape == (10, 32, 32, 3)
        assert images.dtype == np.uint8

    def test_resize(self, fake_cifar: np.ndarray):
        """Requesting a different size resizes every image."""
        images = RealDataProvider(seed=1).cifar10_images(4, h=64, w=64)
        assert images.shape == (4, 64, 64, 3)
        assert images.dtype == np.uint8

    def test_rows_come_from_source(self, fake_cifar: np.ndarray):
        """With n == available, output is a permutation of the source rows."""
        images = RealDataProvider(seed=1).cifar10_images(FAKE_CIFAR_COUNT)
        source_rows = {row.tobytes() for row in fake_cifar}
        output_rows = {row.tobytes() for row in images}
        assert output_rows == source_rows

    def test_deterministic_same_seed(self, fake_cifar: np.ndarray):
        """Same seed yields byte-identical output across providers."""
        first = RealDataProvider(seed=7).cifar10_images(20)
        second = RealDataProvider(seed=7).cifar10_images(20)
        np.testing.assert_array_equal(first, second)

    def test_different_seed_changes_order(self, fake_cifar: np.ndarray):
        """Different seeds select/order rows differently."""
        first = RealDataProvider(seed=1).cifar10_images(20)
        second = RealDataProvider(seed=2).cifar10_images(20)
        assert not np.array_equal(first, second)

    def test_tiling_repeats_permutation(self, fake_cifar: np.ndarray):
        """n beyond the source size tiles the permuted rows deterministically."""
        n = FAKE_CIFAR_COUNT * 2 + 50
        images = RealDataProvider(seed=3).cifar10_images(n)
        assert images.shape == (n, 32, 32, 3)
        np.testing.assert_array_equal(
            images[FAKE_CIFAR_COUNT : 2 * FAKE_CIFAR_COUNT],
            images[:FAKE_CIFAR_COUNT],
        )

    def test_non_positive_n_raises(self, fake_cifar: np.ndarray):
        """n must be positive."""
        with pytest.raises(ValueError, match="n must be positive"):
            RealDataProvider().cifar10_images(0)


# ---------------------------------------------------------------------------
# wikitext_tokens
# ---------------------------------------------------------------------------


class TestWikitextTokens:
    """wikitext_tokens mirrors SyntheticDataGenerator.token_sequences."""

    def test_shape_and_dtype(self, fake_wikitext: list[str]):
        """Output is (n, seq_len) int32 within the vocab."""
        tokens = RealDataProvider(seed=1).wikitext_tokens(8, seq_len=16)
        assert tokens.shape == (8, 16)
        assert tokens.dtype == np.int32
        assert tokens.min() >= 0
        assert tokens.max() < 32000

    def test_vocab_size_respected(self, fake_wikitext: list[str]):
        """Custom vocab_size bounds every token id."""
        tokens = RealDataProvider(seed=1).wikitext_tokens(4, seq_len=8, vocab_size=50)
        assert tokens.max() < 50

    def test_deterministic_same_seed(self, fake_wikitext: list[str]):
        """Same seed yields identical packed sequences."""
        first = RealDataProvider(seed=5).wikitext_tokens(6, seq_len=12)
        second = RealDataProvider(seed=5).wikitext_tokens(6, seq_len=12)
        np.testing.assert_array_equal(first, second)

    def test_word_identity_preserved(self, fake_wikitext: list[str]):
        """Real text structure survives: distinct ids <= distinct words."""
        distinct_words = {word for line in fake_wikitext for word in line.split()}
        tokens = RealDataProvider(seed=1).wikitext_tokens(4, seq_len=16)
        assert len(np.unique(tokens)) <= len(distinct_words)

    def test_tiling_when_corpus_small(self, fake_wikitext: list[str]):
        """Requests larger than the corpus tile the token stream."""
        tokens = RealDataProvider(seed=1).wikitext_tokens(50, seq_len=64)
        assert tokens.shape == (50, 64)

    def test_non_positive_n_raises(self, fake_wikitext: list[str]):
        """n must be positive."""
        with pytest.raises(ValueError, match="n must be positive"):
            RealDataProvider().wikitext_tokens(0, seq_len=8)


# ---------------------------------------------------------------------------
# criteo_features
# ---------------------------------------------------------------------------


class TestCriteoFeatures:
    """criteo_features yields DLRM-style dense + sparse arrays."""

    def test_shapes_and_dtypes(self, fake_criteo: list[str]):
        """Dense is (n, 13) float32; sparse is (n, 26) int64."""
        dense, sparse = RealDataProvider(seed=1).criteo_features(3)
        assert dense.shape == (3, 13)
        assert dense.dtype == np.float32
        assert sparse.shape == (3, 26)
        assert sparse.dtype == np.int64

    def test_sparse_within_hash_buckets(self, fake_criteo: list[str]):
        """Sparse ids are hashed into [0, hash_buckets)."""
        _, sparse = RealDataProvider(seed=1).criteo_features(3, hash_buckets=100)
        assert sparse.min() >= 0
        assert sparse.max() < 100

    def test_missing_dense_is_zero(self, fake_criteo: list[str]):
        """Empty dense fields parse to 0.0, and no NaN leaks through."""
        dense, _ = RealDataProvider(seed=1).criteo_features(3)
        assert not np.isnan(dense).any()
        source_rows = {tuple(row) for row in dense}
        # Line 0 has dense field I3 empty; line 1 has I1 empty.
        assert any(row[2] == 0.0 for row in source_rows)
        assert any(row[0] == 0.0 for row in source_rows)

    def test_deterministic_same_seed(self, fake_criteo: list[str]):
        """Same seed yields identical dense and sparse arrays."""
        provider_a = RealDataProvider(seed=9)
        provider_b = RealDataProvider(seed=9)
        dense_a, sparse_a = provider_a.criteo_features(3)
        dense_b, sparse_b = provider_b.criteo_features(3)
        np.testing.assert_array_equal(dense_a, dense_b)
        np.testing.assert_array_equal(sparse_a, sparse_b)

    def test_tiling(self, fake_criteo: list[str]):
        """n beyond the sample size tiles rows deterministically."""
        dense, sparse = RealDataProvider(seed=1).criteo_features(10)
        assert dense.shape == (10, 13)
        assert sparse.shape == (10, 26)

    def test_malformed_line_raises(self, monkeypatch: pytest.MonkeyPatch):
        """A line with the wrong field count fails fast."""

        def _fake_load(allow_download: bool) -> Iterable[str]:
            return ["0\t1\t2"]

        monkeypatch.setattr(real_data, "_load_criteo_lines", _fake_load)
        with pytest.raises(ValueError, match="DAC"):
            RealDataProvider().criteo_features(1)


# ---------------------------------------------------------------------------
# coco_image_text
# ---------------------------------------------------------------------------


class TestCocoImageText:
    """coco_image_text mirrors SyntheticDataGenerator.image_text_pairs."""

    def test_float32_shapes_and_range(self, fake_coco: list[tuple[np.ndarray, str]]):
        """Default float32 images are (n, h, w, 3) scaled to [0, 1]."""
        images, tokens = RealDataProvider(seed=1).coco_image_text(8, h=64, w=64)
        assert images.shape == (8, 64, 64, 3)
        assert images.dtype == np.float32
        assert images.min() >= 0.0
        assert images.max() <= 1.0
        assert tokens.shape == (8, 77)
        assert tokens.dtype == np.int32

    def test_uint8_mode(self, fake_coco: list[tuple[np.ndarray, str]]):
        """image_dtype='uint8' keeps raw pixel values."""
        images, _ = RealDataProvider(seed=1).coco_image_text(4, h=32, w=32, image_dtype="uint8")
        assert images.dtype == np.uint8
        assert images.shape == (4, 32, 32, 3)

    def test_token_vocab_and_length(self, fake_coco: list[tuple[np.ndarray, str]]):
        """Tokens honor text_len and vocab_size."""
        _, tokens = RealDataProvider(seed=1).coco_image_text(
            4, h=16, w=16, text_len=12, vocab_size=200
        )
        assert tokens.shape == (4, 12)
        assert tokens.min() >= 0
        assert tokens.max() < 200

    def test_deterministic_same_seed(self, fake_coco: list[tuple[np.ndarray, str]]):
        """Same seed yields identical images and tokens."""
        images_a, tokens_a = RealDataProvider(seed=4).coco_image_text(6, h=32, w=32)
        images_b, tokens_b = RealDataProvider(seed=4).coco_image_text(6, h=32, w=32)
        np.testing.assert_array_equal(images_a, images_b)
        np.testing.assert_array_equal(tokens_a, tokens_b)

    def test_tiling(self, fake_coco: list[tuple[np.ndarray, str]]):
        """n beyond the available pairs tiles deterministically."""
        images, tokens = RealDataProvider(seed=1).coco_image_text(50, h=16, w=16)
        assert images.shape == (50, 16, 16, 3)
        assert tokens.shape == (50, 77)


# ---------------------------------------------------------------------------
# Integration tests (locally cached data only; never download in CI)
# ---------------------------------------------------------------------------


@pytest.mark.tfds
@pytest.mark.skipif(
    not real_data.cifar10_is_cached(),
    reason="cifar10 not cached in the local TFDS data dir",
)
class TestCifar10Integration:
    """Integration against the locally cached TFDS cifar10."""

    def test_real_load_contract(self):
        """Cached cifar10 loads and matches the synthetic contract."""
        images = RealDataProvider(seed=1, allow_download=False).cifar10_images(64)
        assert images.shape == (64, 32, 32, 3)
        assert images.dtype == np.uint8
        # Real photographs: not constant, plausible dynamic range.
        assert images.std() > 10

    def test_identical_bytes_across_providers(self):
        """Two providers hand every adapter the exact same bytes."""
        first = RealDataProvider(seed=42, allow_download=False).cifar10_images(32)
        second = RealDataProvider(seed=42, allow_download=False).cifar10_images(32)
        np.testing.assert_array_equal(first, second)


@pytest.mark.tfds
def test_cifar10_missing_data_raises(tmp_path):
    """With downloads disabled and an empty data dir, fail fast and clear."""
    provider = RealDataProvider(data_dir=tmp_path, allow_download=False)
    with pytest.raises(RealDataUnavailableError, match="cifar10"):
        provider.cifar10_images(4)


@pytest.mark.hf
class TestCachedRealDataSmoke:
    """Small offline loads of each real dataset, skipped when not cached.

    These run wherever the datasets have been materialized (developer
    machines, benchmark hosts) and skip cleanly in CI.
    """

    def _smoke(self, load: Callable[[], tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
        try:
            return load()
        except RealDataUnavailableError as e:
            pytest.skip(str(e))

    def test_wikitext_smoke(self):
        """Cached wikitext yields in-contract token sequences."""
        provider = RealDataProvider(seed=1, allow_download=False)
        (tokens,) = self._smoke(lambda: (provider.wikitext_tokens(16, seq_len=32),))
        assert tokens.shape == (16, 32)
        assert tokens.dtype == np.int32
        # Real text: token ids repeat (Zipfian), so uniques < elements.
        assert len(np.unique(tokens)) < tokens.size

    def test_criteo_smoke(self):
        """Cached criteo yields in-contract dense + sparse features."""
        provider = RealDataProvider(seed=1, allow_download=False)
        dense, sparse = self._smoke(lambda: provider.criteo_features(64))
        assert dense.shape == (64, 13)
        assert sparse.shape == (64, 26)
        assert not np.isnan(dense).any()

    def test_coco_smoke(self):
        """Cached coco yields in-contract image/token pairs."""
        provider = RealDataProvider(seed=1, allow_download=False)
        images, tokens = self._smoke(lambda: provider.coco_image_text(8, h=32, w=32))
        assert images.shape == (8, 32, 32, 3)
        assert tokens.shape == (8, 77)
        # Real photographs are not constant images.
        assert images.std() > 0.01
