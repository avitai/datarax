"""Compile-count checks for a first call, independent of what ran earlier in the process.

jax keeps every program it builds in process-wide in-memory caches, so a program an earlier test
built is served to a later one without compiling, and an eager operation an earlier test ran is
already built when a later block calls it. A block asserting what a first call compiles therefore
measures the test order unless it starts from caches it cleared itself.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import jax
from substrax.testing.compiles import compiled_programs


@contextmanager
def expect_first_call_compiles(*programs: str) -> Generator[None]:
    """Require a block to build exactly ``programs``, in order, from cleared caches.

    ``jax.clear_caches()`` clears every in-memory compilation and staging cache: the
    ``util.cache``, ``weakref_lru_cache`` and ``lu.cache`` instances (traced jaxprs, lowerings,
    the compiled executables of ``pxla._cached_compilation``) and the C++ ``jit`` dispatch caches
    (``jax/_src/api.py`` ``clear_caches``, jax 0.11.1). It leaves the persistent compilation
    cache, whose hits ``compiled_programs`` records as builds too. The block therefore sees the
    programs a first call builds, eager operations included, whatever ran before.

    Args:
        *programs: The names of the programs the block must build, as ``jit(<function>)``.

    Yields:
        None: Control to the block.

    Raises:
        AssertionError: If the block built other programs, or the same ones in another order.
    """
    jax.clear_caches()
    with compiled_programs() as built:
        yield
    if built != list(programs):
        raise AssertionError(
            f"expected {list(programs)} built from cleared caches, observed {built}"
        )
