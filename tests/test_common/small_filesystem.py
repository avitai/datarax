"""A script run with a real 1 MiB filesystem mounted, for the tests of what a full disk does.

:func:`run_on_small_filesystem` mounts a tmpfs of a given size at a directory inside a user
namespace (``unshare -rm``, no root) and runs a Python script there, with the test process's
``sys.path`` prepended, so the script imports datarax and the test helpers. A host that refuses
an unprivileged user namespace or the mount (Ubuntu 24.04's AppArmor default) skips the test.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - the tests run their own script under a user namespace
import sys
import textwrap
from pathlib import Path

import pytest


def run_on_small_filesystem(
    mount: Path, script: str, *, size: str = "1m"
) -> subprocess.CompletedProcess[str]:
    """Run ``script`` with a tmpfs of ``size`` mounted at ``mount``, skipping where refused.

    Args:
        mount: An empty directory, the mount point.
        script: Python source; it runs with the test process's ``sys.path`` first.
        size: The filesystem's size, as ``mount -o size=`` takes it.

    Returns:
        The finished run, its output captured.
    """
    if shutil.which("unshare") is None:
        pytest.skip("needs unshare(1) to mount a small filesystem without root")
    probe = subprocess.run(  # nosec B603 - fixed arguments
        ["unshare", "-rm", "true"], capture_output=True, text=True, timeout=30, check=False
    )
    if probe.returncode != 0:  # e.g. Ubuntu 24.04's AppArmor restriction on user namespaces
        pytest.skip(f"this host refuses a user namespace: {probe.stderr.strip()[:200]}")
    prelude = f"import sys\nsys.path[:0] = {[str(p) for p in sys.path if p]!r}\n"
    run = subprocess.run(  # nosec B603 - fixed arguments
        [
            "unshare",
            "-rm",
            "sh",
            "-c",
            f'mount -t tmpfs -o size={size} tmpfs "{mount}" && exec "{sys.executable}" -c "$0"',
            prelude + textwrap.dedent(script),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if run.returncode != 0 and "mount" in run.stderr:
        pytest.skip(f"this host refuses a user-namespace mount: {run.stderr.strip()[:200]}")
    return run
