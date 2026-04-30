"""Driver/solution rate contracts for time-series operators.

Time-series workloads (Neural CDE / ODE, dynamical systems, multi-rate sensor
fusion) emit a dense ``driver`` signal at one rate and a sparse ``solution``
(target) signal at a downsampled rate. ``TimeSeriesSpec`` declares the
relationship as a compile-time constant so downstream operators can size
their interpolators and learnable layers without hard-coding shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class TimeSeriesSpec:
    """Driver/solution rate contract for a time-series sample.

    Attributes:
        driver_length: Number of timesteps in the dense input signal (the
            "control path" in CDE terminology).
        solution_length: Number of timesteps in the sparse target signal
            (the "trajectory" in CDE terminology). Must divide ``driver_length``
            so the downsample factor is an integer.

    Properties:
        downsample_factor: ``driver_length // solution_length``. Used by models
            (e.g., Neural CDE) to size interpolators and time grids.

    Raises:
        ValueError: If ``solution_length <= 0`` or ``driver_length`` is not an
            integer multiple of ``solution_length``.
    """

    driver_length: int
    solution_length: int

    def __post_init__(self) -> None:
        """Validate that ``driver_length`` is a positive integer multiple of ``solution_length``."""
        if self.solution_length <= 0:
            raise ValueError("solution_length must be positive.")
        if self.driver_length % self.solution_length != 0:
            raise ValueError(
                "driver_length must be an integer multiple of solution_length "
                f"(got driver_length={self.driver_length}, "
                f"solution_length={self.solution_length})."
            )

    @property
    def downsample_factor(self) -> int:
        """Integer ratio ``driver_length // solution_length`` used to size interpolators."""
        return self.driver_length // self.solution_length


__all__ = ["TimeSeriesSpec"]
