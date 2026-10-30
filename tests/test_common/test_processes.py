"""``run_processes`` returns what each process reported, whatever the process prints.

A library loaded by a process (Gloo, as ``jax.distributed`` starts) writes to standard output at
any moment, through Python or straight to file descriptor 1; a process's report travels on its own
channel, so a line written after it never replaces it.
"""

from __future__ import annotations

from typing import TypedDict

from tests.test_common.processes import PROCESSES, run_processes


class _Index(TypedDict):
    process: int


_REPORTS_THEN_PRINTS = """
import os, sys
from substrax.testing import report
report({"process": int(sys.argv[2])})
print("[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1")
sys.stdout.flush()
os.write(1, b"written to file descriptor 1 after the report\\n")
"""


def test_each_process_s_report_survives_what_it_prints_after_it() -> None:
    assert run_processes(_REPORTS_THEN_PRINTS, _Index) == [
        {"process": index} for index in range(PROCESSES)
    ]
