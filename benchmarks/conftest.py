"""Pytest fixtures for benchmark application layer tests."""

from pathlib import Path

import numpy as np
import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator


@pytest.fixture
def tmp_baselines_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for baseline storage."""
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    return baselines


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for benchmark output."""
    output = tmp_path / "output"
    output.mkdir()
    return output


@pytest.fixture
def synth_gen() -> SyntheticDataGenerator:
    """Deterministic synthetic data generator with seed=42."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def small_image_data(synth_gen: SyntheticDataGenerator) -> dict[str, np.ndarray]:
    """Small synthetic image data for fast tests (100 x 32x32x3)."""
    return {"image": synth_gen.images(100, 32, 32, 3)}


@pytest.fixture
def small_token_data(synth_gen: SyntheticDataGenerator) -> dict[str, np.ndarray]:
    """Small synthetic token data for fast tests (100 x 128)."""
    return {"tokens": synth_gen.token_sequences(100, 128)}


@pytest.fixture
def small_tabular_data(synth_gen: SyntheticDataGenerator) -> dict[str, np.ndarray]:
    """Small synthetic tabular data for fast tests (100 x 100)."""
    return {"features": synth_gen.tabular(100, 100)}


@pytest.fixture
def datarax_adapter() -> DataraxAdapter:
    """Fresh DataraxAdapter instance."""
    return DataraxAdapter()


@pytest.fixture
def cv1_small_config() -> ScenarioConfig:
    """CV-1 small scenario config for tests."""
    return ScenarioConfig(
        scenario_id="CV-1",
        dataset_size=100,
        element_shape=(32, 32, 3),
        batch_size=10,
        transforms=[],
        seed=42,
        extra={"variant_name": "small"},
    )


@pytest.fixture
def nlp1_small_config() -> ScenarioConfig:
    """NLP-1 small scenario config for tests."""
    return ScenarioConfig(
        scenario_id="NLP-1",
        dataset_size=100,
        element_shape=(128,),
        batch_size=10,
        transforms=[],
        seed=42,
        extra={"variant_name": "small"},
    )


@pytest.fixture
def tab1_small_config() -> ScenarioConfig:
    """TAB-1 small scenario config for tests."""
    return ScenarioConfig(
        scenario_id="TAB-1",
        dataset_size=100,
        element_shape=(100,),
        batch_size=10,
        transforms=[],
        seed=42,
        extra={"variant_name": "small"},
    )
