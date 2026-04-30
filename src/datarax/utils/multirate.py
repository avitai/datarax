"""Multirate signal alignment — preprocessing-time helper.

For sensor-fusion / wearable workloads, channels arrive at different sample
rates. Each channel array is upsampled along the time axis by an integer
factor via ``np.repeat`` so all channels share a common rate post-alignment.
The result is then cached (e.g., as ``.npy``) and consumed by standard
batching downstream — runtime multirate batching is intentionally NOT
provided so the rate-conversion cost is paid once per dataset, not per batch.
"""

from __future__ import annotations

import numpy as np


def multirate_align(
    channels: dict[str, np.ndarray],
    rate_factors: dict[str, int],
    *,
    axis: int = 0,
) -> dict[str, np.ndarray]:
    """Upsample each rate-factored channel along ``axis`` via ``np.repeat``.

    Channels absent from ``rate_factors`` (or with factor 1) pass through
    unchanged. Used at dataset preprocessing time to align signals captured at
    different sample rates onto a common timebase before batching.

    Args:
        channels: Mapping from channel name to ndarray.
        rate_factors: Mapping from channel name to positive integer upsample
            factor. Missing keys default to factor 1 (passthrough).
        axis: Axis along which to apply the repeat (default 0).

    Returns:
        New dict with each channel's array upsampled per its rate factor.

    Raises:
        ValueError: If any rate factor is not a positive integer.
    """
    for key, factor in rate_factors.items():
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError(f"rate_factors[{key!r}] must be a positive int, got {factor!r}.")

    out: dict[str, np.ndarray] = {}
    for key, value in channels.items():
        factor = rate_factors.get(key, 1)
        out[key] = value if factor == 1 else np.repeat(value, factor, axis=axis)
    return out


__all__ = ["multirate_align"]
