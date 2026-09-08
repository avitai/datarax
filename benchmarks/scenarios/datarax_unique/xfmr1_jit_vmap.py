"""XFMR-1: JIT + vmap Transform Acceleration.

Benchmarks the effect of JAX transform compilation modes (eager, vmap,
jit+vmap, jit+vmap+fused) on data pipeline throughput at three resolutions.
This scenario bypasses the adapter for compilation-mode comparison; the
standard adapter interface still works for baseline measurements.

Design ref: Section 7.3.9 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "XFMR-1"
TIER1_VARIANT: str | None = None

_COMPILATION_MODES = ["eager", "vmap", "jit_vmap", "jit_vmap_fused"]

# Resolution -> (dataset_size, element_shape, batch_size)
_RESOLUTION_CONFIGS = {
    "32x32": (50_000, (32, 32, 3), 128),
    "128x128": (50_000, (128, 128, 3), 64),
    "512x512": (10_000, (512, 512, 3), 16),
}


def _build_variants() -> dict[str, ScenarioVariant]:
    """One variant per resolution, each with a generator bound to its own shape."""
    variants: dict[str, ScenarioVariant] = {}
    for res_name, (ds_size, shape, bs) in _RESOLUTION_CONFIGS.items():
        # Bind the loop values through defaults so each generator keeps its own shape.
        def make_gen(n: int = ds_size, h: int = shape[0], w: int = shape[1]) -> dict:
            gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
            return {"image": gen.images(n, h, w, 3, dtype="float32")}

        variants[res_name] = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id=SCENARIO_ID,
                dataset_size=ds_size,
                element_shape=shape,
                batch_size=bs,
                transforms=["Normalize", "RandomCrop", "AffineRotation"],
                seed=DEFAULT_SEED,
                extra={
                    "modes": list(_COMPILATION_MODES),
                    "variant_name": res_name,
                },
            ),
            data_generator=make_gen,
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()

get_variant = make_get_variant(VARIANTS)
