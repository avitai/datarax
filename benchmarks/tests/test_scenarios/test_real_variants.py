"""Tests for real-data scenario variants.

Real-data variants live alongside the synthetic ones (synthetic stays the
fast CI default) and are fed by the framework-neutral RealDataProvider so
every adapter receives byte-identical input. Generator tests run against
the fake loaders from the shared conftest, so they are fast and offline;
dataset sizes are exercised at small n through the factory functions.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from benchmarks.scenarios import real_data_variants
from benchmarks.tests.test_scenarios.conftest import assert_valid_variant


# (module path, real variant, synthetic sibling, element_shape, dataset_size)
_REAL_VARIANTS: list[tuple[str, str, str, tuple[int, ...], int]] = [
    (
        "benchmarks.scenarios.vision.cv1_image_classification",
        "real_cifar10",
        "small",
        (32, 32, 3),
        10_000,
    ),
    (
        "benchmarks.scenarios.vision.cv3_batch_mixing",
        "real_cifar10",
        "default",
        (64, 64, 3),
        5_000,
    ),
    (
        "benchmarks.scenarios.vision.hcv1_imagenet_classification",
        "real_cifar10",
        "imagenet_small",
        (224, 224, 3),
        50_000,
    ),
    (
        "benchmarks.scenarios.nlp.nlp1_llm_pretraining",
        "real_wikitext",
        "small",
        (128,),
        10_000,
    ),
    (
        "benchmarks.scenarios.nlp.hnlp1_llm_pretraining",
        "real_wikitext",
        "short_context",
        (2048,),
        100_000,
    ),
    (
        "benchmarks.scenarios.tabular.tab1_dense_features",
        "real_criteo",
        "small",
        (13,),
        10_000,
    ),
    (
        "benchmarks.scenarios.tabular.htab1_recommendation",
        "real_criteo",
        "small",
        (39,),
        1_000_000,
    ),
    (
        "benchmarks.scenarios.multimodal.mm1_image_text",
        "real_coco",
        "default",
        (64, 64, 3),
        5_000,
    ),
    (
        "benchmarks.scenarios.multimodal.hmm1_vision_language",
        "real_coco",
        "clip_small",
        (224, 224, 3),
        50_000,
    ),
]

_PARAM_IDS = [f"{path.rsplit('.', 1)[-1]}:{variant}" for path, variant, *_ in _REAL_VARIANTS]


@pytest.mark.parametrize(
    ("module_path", "variant_name", "sibling_name", "element_shape", "dataset_size"),
    _REAL_VARIANTS,
    ids=_PARAM_IDS,
)
class TestRealVariantRegistration:
    """Every target scenario registers a real-data variant."""

    def test_variant_registered_and_valid(
        self,
        module_path: str,
        variant_name: str,
        sibling_name: str,
        element_shape: tuple[int, ...],
        dataset_size: int,
    ):
        """The real variant exists with the expected config."""
        module = importlib.import_module(module_path)
        assert variant_name in module.VARIANTS
        variant = module.VARIANTS[variant_name]
        assert_valid_variant(variant)
        config = variant.config
        assert config.extra["variant_name"] == variant_name
        assert tuple(config.element_shape) == element_shape
        assert config.dataset_size == dataset_size

    def test_variant_mirrors_synthetic_sibling(
        self,
        module_path: str,
        variant_name: str,
        sibling_name: str,
        element_shape: tuple[int, ...],
        dataset_size: int,
    ):
        """Transforms and capabilities match the synthetic sibling variant.

        The real variant must benchmark the same pipeline on real bytes:
        only the data source differs.
        """
        module = importlib.import_module(module_path)
        real_config = module.VARIANTS[variant_name].config
        sibling_config = module.VARIANTS[sibling_name].config
        assert real_config.transforms == sibling_config.transforms
        assert real_config.required_capabilities == sibling_config.required_capabilities
        assert real_config.batch_size == sibling_config.batch_size


class TestCifar10ImageData:
    """cifar10_image_data factory yields the CV image contract."""

    def test_contract(self, fake_cifar: np.ndarray):
        """Images are (n, h, w, 3) uint8 under the 'image' key."""
        data = real_data_variants.cifar10_image_data(8, h=64, w=64)()
        assert set(data) == {"image"}
        assert data["image"].shape == (8, 64, 64, 3)
        assert data["image"].dtype == np.uint8

    def test_deterministic(self, fake_cifar: np.ndarray):
        """Two invocations produce byte-identical data."""
        generate = real_data_variants.cifar10_image_data(8)
        np.testing.assert_array_equal(generate()["image"], generate()["image"])


class TestWikitextTokenData:
    """wikitext_token_data factory yields the NLP token contract."""

    def test_contract(self, fake_wikitext: list[str]):
        """Tokens are (n, seq_len) int32 under the 'tokens' key."""
        data = real_data_variants.wikitext_token_data(4, seq_len=16)()
        assert set(data) == {"tokens"}
        assert data["tokens"].shape == (4, 16)
        assert data["tokens"].dtype == np.int32

    def test_deterministic(self, fake_wikitext: list[str]):
        """Two invocations produce byte-identical data."""
        generate = real_data_variants.wikitext_token_data(4, seq_len=16)
        np.testing.assert_array_equal(generate()["tokens"], generate()["tokens"])


class TestCriteoData:
    """Criteo factories yield the tabular feature contracts."""

    def test_dense_contract(self, fake_criteo: list[str]):
        """Dense-only features are (n, 13) float32."""
        data = real_data_variants.criteo_dense_data(6)()
        assert set(data) == {"features"}
        assert data["features"].shape == (6, 13)
        assert data["features"].dtype == np.float32

    def test_recommendation_contract(self, fake_criteo: list[str]):
        """Dense + hashed-sparse features concatenate to (n, 39) float32."""
        data = real_data_variants.criteo_recommendation_data(6)()
        features = data["features"]
        assert features.shape == (6, 39)
        assert features.dtype == np.float32
        # Sparse half holds hashed ids: non-negative integral values.
        sparse_half = features[:, 13:]
        assert (sparse_half >= 0).all()
        np.testing.assert_array_equal(sparse_half, np.round(sparse_half))

    def test_deterministic(self, fake_criteo: list[str]):
        """Two invocations produce byte-identical data."""
        generate = real_data_variants.criteo_recommendation_data(6)
        np.testing.assert_array_equal(generate()["features"], generate()["features"])


class TestCocoPairData:
    """coco_pair_data factory yields the multimodal pair contract."""

    def test_float32_contract(self, fake_coco: list[tuple[np.ndarray, str]]):
        """Float32 images in [0, 1] plus int32 tokens."""
        data = real_data_variants.coco_pair_data(4, h=32, w=32)()
        assert set(data) == {"image", "tokens"}
        assert data["image"].shape == (4, 32, 32, 3)
        assert data["image"].dtype == np.float32
        assert data["image"].max() <= 1.0
        assert data["tokens"].shape == (4, 77)
        assert data["tokens"].dtype == np.int32

    def test_uint8_contract(self, fake_coco: list[tuple[np.ndarray, str]]):
        """uint8 mode keeps raw pixels and honors vocab_size."""
        data = real_data_variants.coco_pair_data(
            4, h=32, w=32, vocab_size=49408, image_dtype="uint8"
        )()
        assert data["image"].dtype == np.uint8
        assert data["tokens"].max() < 49408

    def test_deterministic(self, fake_coco: list[tuple[np.ndarray, str]]):
        """Two invocations produce byte-identical data."""
        generate = real_data_variants.coco_pair_data(4, h=32, w=32)
        first = generate()
        second = generate()
        np.testing.assert_array_equal(first["image"], second["image"])
        np.testing.assert_array_equal(first["tokens"], second["tokens"])
