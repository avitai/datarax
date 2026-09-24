"""The XLA programs compiled inside a block, for tests that a repeated call compiles nothing."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import jax


@contextmanager
def compiled_programs() -> Iterator[list[str]]:
    """Names of the XLA programs compiled inside the block, read from JAX's compile log."""
    names: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Finished XLA compilation of" in message:
                names.append(message.split("compilation of ")[1].split(" in ")[0])

    logger = logging.getLogger("jax")
    handler, level = _Handler(), logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with jax.log_compiles(True):
            yield names
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
