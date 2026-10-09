"""The device arrays a block of code made: the live arrays not in a snapshot the caller holds.

A test asserting that a call makes no device array takes ``before = jax.live_arrays()``, runs the
call, and asserts ``arrays_made_since(before)`` holds none (or none of a given shape). Holding the
snapshot keeps every array in it alive, so an array freed during the call (by a garbage collection
of an unrelated reference cycle, say) neither reads as a change nor frees an id a new array could
reuse. Comparing two sets of ids, or two counts, has both faults.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax


def arrays_made_since(before: Sequence[jax.Array]) -> list[jax.Array]:
    """The live device arrays that are not in ``before``.

    Args:
        before: The result of ``jax.live_arrays()`` at the window's start, held by the caller
            until this returns.

    Returns:
        The arrays alive now that were not alive at the snapshot, in ``jax.live_arrays()`` order.
    """
    known = {id(array) for array in before}
    return [array for array in jax.live_arrays() if id(array) not in known]
