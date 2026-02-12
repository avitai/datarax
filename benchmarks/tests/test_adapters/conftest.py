"""Shared fixtures and helpers for adapter tests.

Provides common configurations, data, and assertion helpers for all 15
adapter test files. Extends the parent conftest (benchmarks/conftest.py)
with adapter-specific fixtures.

DRY: All adapter tests import these fixtures instead of creating their own.
"""

import numpy as np
import pytest

from benchmarks.adapters.base import IterationResult, ScenarioConfig


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------


def assert_valid_iteration_result(result: IterationResult) -> None:
    """Assert that an IterationResult has valid, non-degenerate values.

    Checks all structural invariants that every adapter must satisfy:
    - Correct type
    - Positive batch count, element count, wall clock, first batch time
    - per_batch_times length matches num_batches
    """
    assert isinstance(result, IterationResult)
    assert result.num_batches > 0
    assert result.num_elements > 0
    assert result.wall_clock_sec > 0
    assert result.first_batch_time > 0
    assert len(result.per_batch_times) == result.num_batches


# ---------------------------------------------------------------------------
# Ray cleanup fixture — follows Ray's own ``shutdown_only`` pattern
# (see https://docs.ray.io/en/latest/ray-contribute/testing-tips.html)
# ---------------------------------------------------------------------------


@pytest.fixture
def ray_shutdown():
    """Ensure Ray is shut down after a test.

    Disconnects from the cluster and stops the CLI-started Ray head node.
    The adapter uses ``ray start --head`` (external process) instead of
    in-process ``ray.init()`` to avoid fork-safety issues with native
    extensions like deeplake's Rust runtime (244 OS threads at import).
    """
    yield
    import ray

    ray.shutdown()

    from benchmarks.adapters.ray_data_adapter import _stop_ray

    _stop_ray()


# ---------------------------------------------------------------------------
# Multi-modal fixtures (not in parent conftest)
# ---------------------------------------------------------------------------


@pytest.fixture
def mm1_small_config() -> ScenarioConfig:
    """MM-1 small scenario config — image-text pairs."""
    return ScenarioConfig(
        scenario_id="MM-1",
        dataset_size=50,
        element_shape=(32, 32, 3),
        batch_size=10,
        transforms=[],
        seed=42,
        extra={"variant_name": "small"},
    )


@pytest.fixture
def small_multimodal_data() -> dict[str, np.ndarray]:
    """50 synthetic image-text pairs."""
    rng = np.random.default_rng(42)
    return {
        "image": rng.integers(0, 256, (50, 32, 32, 3), dtype=np.uint8),
        "tokens": rng.integers(0, 32000, (50, 77), dtype=np.int32),
    }


# ---------------------------------------------------------------------------
# CV-1 with transforms
# ---------------------------------------------------------------------------


@pytest.fixture
def cv1_transform_config() -> ScenarioConfig:
    """CV-1 config with Normalize + CastToFloat32 transforms."""
    return ScenarioConfig(
        scenario_id="CV-1",
        dataset_size=100,
        element_shape=(32, 32, 3),
        batch_size=10,
        transforms=["Normalize", "CastToFloat32"],
        seed=42,
        extra={"variant_name": "small"},
    )
