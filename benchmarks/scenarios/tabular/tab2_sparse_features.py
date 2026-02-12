"""TAB-2: Sparse Features (DLRM-style) scenario.

Mixed dense + sparse categorical features simulating DLRM-style
recommendation workloads. Sparse features are flattened into a dict.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "TAB-2"
TIER1_VARIANT: str | None = None

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_NUM_DENSE = 13
_NUM_SPARSE = 26

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "default": {
        "dataset_size": 1_000_000,
        "batch_size": 1024,
    },
}


def _make_data_generator(
    dataset_size: int,
    num_dense: int,
    num_sparse: int,
    seed: int = DEFAULT_SEED,
) -> callable:
    """Create a lazy data generator for sparse feature data."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        dense, sparse_list = gen.sparse_features(
            dataset_size, num_dense=num_dense, num_sparse=num_sparse
        )
        data: dict[str, Any] = {"dense": dense}
        data.update({f"sparse_{i}": s for i, s in enumerate(sparse_list)})
        return data

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all TAB-2 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=(_NUM_DENSE,),
            batch_size=spec["batch_size"],
            transforms=[],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(spec["dataset_size"], _NUM_DENSE, _NUM_SPARSE),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
