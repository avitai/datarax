"""Tests for ``datarax.utils.multirate`` — preprocessing-time multirate alignment.

Each channel is upsampled to a common rate via ``np.repeat`` during dataset
preprocessing, then the aligned arrays are cached and consumed by standard
batching downstream. Runtime multirate batching is intentionally NOT provided
so the rate-conversion cost is paid once per dataset, not per batch.
"""

from __future__ import annotations

import numpy as np
import pytest

from datarax.utils.multirate import multirate_align


def test_multirate_align_repeats_each_channel_by_its_factor() -> None:
    """rate_factors={"slow": 16, "fast": 1} → slow repeated 16x along time axis.

    Models a typical wearable-sensor case: ACC at half-rate (factor 2), BVP
    at base rate (factor 1), EDA/TEMP at 1/16 rate (factor 16).
    """
    channels = {
        "slow": np.arange(2, dtype=np.float32),  # length 2
        "fast": np.arange(32, dtype=np.float32),  # length 32
    }
    rate_factors = {"slow": 16, "fast": 1}

    aligned = multirate_align(channels, rate_factors)

    assert aligned["slow"].shape == (32,)
    assert aligned["fast"].shape == (32,)
    np.testing.assert_array_equal(aligned["slow"][:16], np.zeros(16, dtype=np.float32))
    np.testing.assert_array_equal(aligned["slow"][16:], np.ones(16, dtype=np.float32))


def test_multirate_align_unfactored_channels_pass_through() -> None:
    """Channels not in ``rate_factors`` are returned unchanged."""
    channels = {
        "slow": np.arange(3, dtype=np.float32),
        "label": np.arange(8, dtype=np.int32),  # not in rate_factors
    }
    aligned = multirate_align(channels, {"slow": 4})

    assert aligned["slow"].shape == (12,)
    np.testing.assert_array_equal(aligned["label"], channels["label"])


def test_multirate_align_works_on_2d_arrays_along_axis_zero() -> None:
    """Default ``axis=0`` upsampling matches ``np.repeat(..., axis=0)`` semantics."""
    acc = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)  # (2, 3)
    aligned = multirate_align({"acc": acc}, {"acc": 2}, axis=0)

    assert aligned["acc"].shape == (4, 3)
    np.testing.assert_array_equal(aligned["acc"][0], aligned["acc"][1])
    np.testing.assert_array_equal(aligned["acc"][2], aligned["acc"][3])


def test_multirate_align_rejects_non_positive_factor() -> None:
    """Rate factors must be positive integers."""
    with pytest.raises(ValueError, match="positive"):
        multirate_align({"x": np.zeros(4)}, {"x": 0})
    with pytest.raises(ValueError, match="positive"):
        multirate_align({"x": np.zeros(4)}, {"x": -1})


def test_multirate_align_typical_wearable_pattern() -> None:
    """Reproduce a typical wearable-sensor alignment (BVP / ACC / EDA / TEMP).

    Channel rates (example from a 64 Hz BVP-anchored configuration):
    - BVP at base rate (factor 1, kept as-is)
    - ACC at 32 Hz vs BVP 64 Hz → factor 2
    - EDA at 4 Hz vs BVP 64 Hz → factor 16
    - TEMP at 4 Hz → factor 16

    After alignment, every channel has length 64 (matching BVP).
    """
    bvp = np.linspace(0, 1, 64, dtype=np.float32)  # 64 samples (base rate)
    acc = np.linspace(0, 1, 32, dtype=np.float32)  # 32 samples
    eda = np.linspace(0, 1, 4, dtype=np.float32)  # 4 samples
    temp = np.linspace(0, 1, 4, dtype=np.float32)  # 4 samples

    aligned = multirate_align(
        {"BVP": bvp, "ACC": acc, "EDA": eda, "TEMP": temp},
        {"ACC": 2, "EDA": 16, "TEMP": 16},
        axis=0,
    )

    assert aligned["BVP"].shape == (64,)
    assert aligned["ACC"].shape == (64,)
    assert aligned["EDA"].shape == (64,)
    assert aligned["TEMP"].shape == (64,)
