"""Throughput regression guard against the committed baselines.

Runs the Tier-1 scenarios through the datarax adapter with the same
measurement shape as the baseline runs and fails when throughput drops
below 1/1.5 of the committed baseline. Guarded to the hardware the
baselines were recorded on: on any other GPU (or CPU) the comparison is
meaningless and the tests skip.
"""

from __future__ import annotations

import pytest

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.baselines import BaselineStore
from benchmarks.core.environment import capture_environment
from benchmarks.scenarios.base import run_scenario


BASELINES_DIR = "benchmarks/baselines"
MAX_SLOWDOWN = 1.5
_FAILURE_RATIO = 1.0 / MAX_SLOWDOWN

# (baseline name, scenario module, variant) — fast Tier-1 anchors.
_GUARDED = [
    ("CV-1_small", "benchmarks.scenarios.vision.cv1_image_classification", "small"),
    ("NLP-1_small", "benchmarks.scenarios.nlp.nlp1_llm_pretraining", "small"),
    ("TAB-1_small", "benchmarks.scenarios.tabular.tab1_dense_features", "small"),
]


def _gpu_matches_baseline(store: BaselineStore, name: str) -> bool:
    """True when the local GPU matches the one the baseline was recorded on."""
    baseline = store.load(name)
    if baseline is None:
        return False
    baseline_gpu = str(baseline.get("metadata", {}).get("environment", {}).get("gpu", ""))
    current_gpu = str(capture_environment().get("gpu", ""))
    baseline_model = baseline_gpu.split(",")[0].strip()
    current_model = current_gpu.split(",")[0].strip()
    return bool(baseline_model) and baseline_model == current_model


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(
    ("baseline_name", "module_path", "variant_name"),
    _GUARDED,
    ids=[entry[0] for entry in _GUARDED],
)
def test_throughput_within_regression_budget(
    baseline_name: str, module_path: str, variant_name: str
):
    """Current datarax throughput stays within 1.5x of the baseline."""
    import importlib
    import sys

    import jax

    if "coverage" in sys.modules:
        # Instrumentation roughly halves throughput; timings under coverage
        # would flag phantom regressions.
        pytest.skip("throughput is not meaningful under coverage instrumentation")
    if jax.default_backend() != "gpu":
        pytest.skip("regression guard runs on the baseline GPU only")
    store = BaselineStore(BASELINES_DIR)
    if not _gpu_matches_baseline(store, baseline_name):
        pytest.skip(f"local GPU differs from the {baseline_name} baseline hardware")

    module = importlib.import_module(module_path)
    variant = module.get_variant(variant_name)
    result = run_scenario(
        DataraxAdapter(),
        variant,
        num_batches=40,
        warmup_batches=6,
        num_repetitions=3,
    )
    verdict = store.compare(baseline_name, result, failure_ratio=_FAILURE_RATIO)
    assert verdict is not None
    assert verdict["status"] != "failure", (
        f"{baseline_name}: throughput {verdict['current_throughput']:.0f} elem/s is below "
        f"1/{MAX_SLOWDOWN}x of the baseline {verdict['baseline_throughput']:.0f} elem/s "
        f"(ratio {verdict['throughput_ratio']:.2f})"
    )
