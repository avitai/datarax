"""HTAB-1: Large-Scale Recommendation System Pipeline (Heavy).

DLRM/DCN-v2 style recommendation model data pipeline with extremely large
batch sizes (up to 65K) and dataset sizes (100M+). Tests Datarax's throughput
ceiling with mixed dense + sparse features and non-trivial transforms
(log transform, feature hashing).

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HTAB-1"
TIER1_VARIANT: str | None = "small"

_NUM_DENSE = 13
_NUM_SPARSE = 26
_TOTAL_FEATURES = _NUM_DENSE + _NUM_SPARSE


def _rec_data(n: int) -> dict[str, np.ndarray]:
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    dense = gen.rng.standard_normal((n, _NUM_DENSE)).astype(np.float32)
    sparse = gen.rng.integers(0, 10000, (n, _NUM_SPARSE), dtype=np.int32)
    # Concatenate into single feature array for the adapter pipeline
    features = np.concatenate([dense, sparse.astype(np.float32)], axis=1)
    return {"features": features}


VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=1_000_000,
            element_shape=(_TOTAL_FEATURES,),
            batch_size=4096,
            transforms=["LogTransform", "Normalize"],
            extra={
                "variant_name": "small",
                "num_dense": _NUM_DENSE,
                "num_sparse": _NUM_SPARSE,
            },
        ),
        data_generator=lambda: _rec_data(1_000_000),
    ),
    "medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000_000,
            element_shape=(_TOTAL_FEATURES,),
            batch_size=8192,
            transforms=["LogTransform", "Normalize"],
            extra={
                "variant_name": "medium",
                "num_dense": _NUM_DENSE,
                "num_sparse": _NUM_SPARSE,
            },
        ),
        data_generator=lambda: _rec_data(10_000_000),
    ),
}


get_variant = make_get_variant(VARIANTS)
