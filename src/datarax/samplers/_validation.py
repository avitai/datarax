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
