"""Feistel cipher-based index shuffle for O(1) worker-invariant permutations.

Adapted from Grain's `index_shuffle` algorithm. Uses a 3-round balanced Feistel
network to compute a permutation on-the-fly without materializing the full
permutation array.

Key properties:
- O(1) memory: no permutation array needed
- O(1) per element: compute shuffled position of any element directly
- Worker-count invariant: each worker gets the same mapping regardless of count
- Deterministic: same seed → same permutation
"""

import hashlib


def _round_fn(value: int, key: int, modulus: int) -> int:
    """Feistel round function using a keyed hash.

    Args:
        value: Input half of the Feistel block.
        key: Round key (derived from seed + round number).
        modulus: Size of the output space.

    Returns:
        Pseudo-random value in [0, modulus).
    """
    # Use a simple but effective hash mixing function
    h = hashlib.blake2b(
        f"{value}:{key}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little") % modulus


def index_shuffle(index: int, seed: int, num_elements: int) -> int:
    """Compute the shuffled position of element `index` without materializing the permutation.

    Uses a 3-round Feistel cipher to bijectively map indices [0, N) → [0, N).
    The Feistel network operates on the two halves of the index, with the
    domain adapted to N using the cycle-walking technique.

    Args:
        index: Original element index in [0, num_elements).
        seed: Seed for the permutation (same seed = same permutation).
        num_elements: Total number of elements N.

    Returns:
        Shuffled index in [0, num_elements).
    """
    if num_elements <= 1:
        return 0

    # Find the smallest power-of-2 domain that fits num_elements
    # Split into two halves for the Feistel network
    # Use ceil(sqrt(N)) for left half, ceil(N / left) for right half
    left_size = 1
    while left_size * left_size < num_elements:
        left_size += 1
    right_size = (num_elements + left_size - 1) // left_size

    # 3-round Feistel cipher with cycle-walking to stay in [0, N)
    NUM_ROUNDS = 3
    current = index

    # Cycle-walk: apply Feistel until result is in valid range
    while True:
        left = current // right_size
        right = current % right_size

        for round_num in range(NUM_ROUNDS):
            round_key = seed * 1000003 + round_num * 999983
            if round_num % 2 == 0:
                # Even round: modify left using right
                left = (left + _round_fn(right, round_key, left_size)) % left_size
            else:
                # Odd round: modify right using left
                right = (right + _round_fn(left, round_key, right_size)) % right_size

        result = left * right_size + right

        if result < num_elements:
            return result

        # Result out of range — cycle-walk: use result as new input
        current = result
