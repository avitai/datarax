"""Shared random-parameter helpers for operators."""

from collections.abc import Callable
from typing import Any

import jax


def per_element_params(
    element_keys: jax.Array,
    draw_one: Callable[[jax.Array], Any],
) -> Any:
    """Map a single-record draw over per-record PRNG keys.

    ``element_keys`` is the ``(batch_size,)`` array of per-record keys the base
    operator derives as ``fold_in(base_key, global_index)``. ``draw_one(key)``
    produces the random parameter(s) for ONE record from ONE key; this vmaps it
    to yield a batched result with leading dim ``batch_size``. Keying each
    record's randomness on its own key (not one shared per-batch draw) is what
    makes augmentation invariant to batch composition, shuffle, host count, and
    resume.
    """
    return jax.vmap(draw_one)(element_keys)
