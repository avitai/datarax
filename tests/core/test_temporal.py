"""Tests for TimeSeriesSpec — driver/solution rate contract for time-series operators."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from datarax.core.temporal import TimeSeriesSpec


def test_timeseries_spec_requires_integer_downsample() -> None:
    """driver_length must be an integer multiple of solution_length.

    A non-integer ratio produces ambiguous alignment between driver and
    solution (e.g., a length-10 driver and length-3 solution can't be aligned
    by simple downsampling). The constructor must reject such cases at init time.
    """
    with pytest.raises(ValueError, match="integer multiple"):
        TimeSeriesSpec(driver_length=10, solution_length=3)


def test_timeseries_spec_computes_factor() -> None:
    """downsample_factor = driver_length // solution_length.

    Used by Neural CDE / Neural ODE models to size their interpolators.
    """
    spec = TimeSeriesSpec(driver_length=64, solution_length=8)
    assert spec.downsample_factor == 8


def test_timeseries_spec_is_frozen() -> None:
    """Mutating a spec after construction must raise FrozenInstanceError.

    Specs flow through pipeline construction as compile-time constants;
    runtime mutation would invalidate the contract that downstream operators
    rely on for shape inference.
    """
    spec = TimeSeriesSpec(driver_length=64, solution_length=8)
    with pytest.raises(FrozenInstanceError):
        spec.driver_length = 128  # type: ignore[misc]


def test_timeseries_spec_rejects_zero_solution_length() -> None:
    """solution_length must be positive (zero would make downsample_factor undefined)."""
    with pytest.raises(ValueError, match="positive"):
        TimeSeriesSpec(driver_length=64, solution_length=0)
