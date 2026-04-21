"""Console output helpers for scripts and benchmarks."""

from __future__ import annotations

import sys
from typing import TextIO


def emit(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    file: TextIO | None = None,
    flush: bool = False,
) -> None:
    """Write console output without using the print builtin."""
    stream = sys.stdout if file is None else file
    stream.write(sep.join(str(value) for value in values))
    stream.write(end)
    if flush:
        stream.flush()
