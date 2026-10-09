"""Fail unless this interpreter is the Python version named: a CI leg runs its matrix version.

uv takes its interpreter from ``.python-version`` unless told otherwise, so a leg named for one
Python version can run another and pass. Each leg of a Python version matrix runs this with its
matrix version, in the environment its tests run in:

    uv run python scripts/require_python.py 3.13
"""

from __future__ import annotations

import re
import sys


_MAJOR_MINOR = re.compile(r"\d+\.\d+")


def main(argv: list[str]) -> int:
    """Check the running interpreter against ``argv[0]``, a ``major.minor`` version.

    Args:
        argv: The command-line arguments after the script's name.

    Returns:
        0 when the interpreter is the version named.

    Raises:
        SystemExit: With a message, when the argument is not one ``major.minor`` version or the
            interpreter is another version.
    """
    if len(argv) != 1 or not _MAJOR_MINOR.fullmatch(argv[0]):
        raise SystemExit(f"usage: require_python.py <major.minor>, got {argv}")
    wanted = argv[0]
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"wanted Python {wanted}; interpreter {sys.executable}: {sys.version}")
    if running != wanted:
        raise SystemExit(f"this environment runs Python {running}, not {wanted}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
