"""Run a program as every process of one ``jax.distributed`` run, on the CPU platform.

Each process is a fresh interpreter started by ``substrax.testing.run_python`` with the program,
the coordinator's port and its process index as arguments, and two CPU devices; the processes run
concurrently and each one's last line is read as JSON of the caller's shape.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from substrax.runtime import JaxRuntime
from substrax.testing import run_python


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSES = 2
DEVICES_PER_PROCESS = 2
CHILD_SECONDS = 180.0


def free_port() -> int:
    """A port no process listens on now, for the run's coordinator."""
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


def run_processes[T](program: str, report: type[T], *args: str) -> list[T]:
    """``program`` as every process of one run, each one's last JSON line as ``report``.

    Args:
        program: Python code; ``sys.argv[1:3]`` are the port and the process index, then ``args``.
        report: The shape each process's last line must have, validated strictly.
        *args: More arguments, the same for every process.

    Returns:
        The reports, in process order.
    """
    port = free_port()
    runtime = JaxRuntime(platforms=("cpu",), cpu_devices=DEVICES_PER_PROCESS, preallocate=False)
    with ThreadPoolExecutor(max_workers=PROCESSES) as pool:
        children = [
            pool.submit(
                run_python,
                program,
                str(port),
                str(process),
                *args,
                timeout=CHILD_SECONDS,
                runtime=runtime,
                cwd=REPO_ROOT,
            )
            for process in range(PROCESSES)
        ]
        results = [child.result() for child in children]
    return [result.check().last_json_as(report) for result in results]


__all__ = ["CHILD_SECONDS", "PROCESSES", "REPO_ROOT", "free_port", "run_processes"]
