"""Validation helpers shared by sampler configs."""

import logging


logger = logging.getLogger(__name__)


def validate_sampler_bounds(num_records: int | None, num_epochs: int) -> None:
    """Validate common sampler config bounds for record and epoch counts."""
    if num_records is None:
        raise ValueError("num_records is required")
    if num_records <= 0:
        raise ValueError(f"num_records must be positive, got {num_records}")
    if num_epochs < -1 or num_epochs == 0:
        raise ValueError(f"num_epochs must be positive or -1 (infinite), got {num_epochs}")


def validate_seed(seed: int) -> None:
    """Validate a shuffle seed against Grain's accepted range ``[0, 2**32)``.

    Matches ``grain``'s own validation (``seed < 0 or seed.bit_length() > 32``),
    so a seed accepted here is accepted by Grain's ``IndexSampler``/shuffle.
    """
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}")
