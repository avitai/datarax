"""A spy on ``numpy.memmap.flush``, for worker processes the tests start.

A worker process imports ``sitecustomize`` at start-up from ``PYTHONPATH``. A test that puts this
directory there and names a log in ``DATARAX_TEST_FLUSH_LOG`` sees every ``flush`` a process
makes, as its process id per line; without the variable this module does nothing.
"""

from __future__ import annotations

import os
from pathlib import Path


_LOG = os.environ.get("DATARAX_TEST_FLUSH_LOG")

if _LOG:
    import numpy as np

    _PATH = Path(_LOG)
    _flush = np.memmap.flush

    def _logged(self):  # noqa: ANN001, ANN202 - the wrapped signature
        with _PATH.open("a") as log:
            log.write(f"{os.getpid()}\n")
        return _flush(self)

    np.memmap.flush = _logged
