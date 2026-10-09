"""A spy on a TFDS stream's payload reads, for worker processes the tests start.

A worker process imports ``sitecustomize`` at start-up from ``PYTHONPATH``. A test that puts this
directory there and names a log in ``DATARAX_TEST_PAYLOAD_LOG`` sees every record payload a worker
reads, as ``<shard file> <offset>`` per line; without the variable this module does nothing.
"""

from __future__ import annotations

import os
from pathlib import Path


_LOG = os.environ.get("DATARAX_TEST_PAYLOAD_LOG")

if _LOG:
    from datarax.sources import tfds_source

    _PATH = Path(_LOG)

    _read = tfds_source._payload  # noqa: SLF001 - the function the spy wraps

    def _logged(file, path, index, offset):  # noqa: ANN001, ANN202 - the wrapped signature
        with _PATH.open("a") as log:
            log.write(f"{Path(path).name} {offset}\n")
        return _read(file, path, index, offset)

    tfds_source._payload = _logged  # noqa: SLF001
