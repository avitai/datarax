"""Grain-backed O(1) worker-invariant index shuffle."""

import logging

from grain.experimental import index_shuffle as grain_index_shuffle


logger = logging.getLogger(__name__)


def index_shuffle(index: int, seed: int, num_elements: int) -> int:
    """Compute Grain's shuffled position without materializing a permutation.

    Args:
        index: Original element index in [0, num_elements).
        seed: Seed for the permutation (same seed = same permutation).
        num_elements: Total number of elements N.

    Returns:
        Shuffled index in [0, num_elements).
    """
    if index < 0 or index >= num_elements:
        raise IndexError(f"Index {index} out of range for {num_elements} elements")
    if num_elements <= 1:
        return 0
    return grain_index_shuffle(
        index=index,
        max_index=num_elements - 1,
        seed=seed,
        rounds=4,
    )
