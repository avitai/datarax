"""The comparison sharding guide holds its claims when it runs as two processes.

The guide is one program every process runs, joined into one ``jax.distributed`` run by the
coordinator, process count and process index on its command line; the example tests run it as
one process. Here two processes with two CPU devices each run it as JAX's multi-controller guide
launches one, and each reports the guide's claims: Datarax's global batches and per-record draws
are the single-process ones, Grain's global batch depends on the process count, and both fits
reach the true weights.
"""

from __future__ import annotations

from typing import TypedDict

import pytest

from tests.test_common.processes import PROCESSES, REPO_ROOT, run_processes


GUIDE = REPO_ROOT / "examples" / "comparison" / "03_sharding_guide.py"

PROGRAM = """
import json, runpy, sys

port, process, guide = sys.argv[1:4]
sys.argv = [guide, f"localhost:{port}", "2", process]
claims = runpy.run_path(guide, run_name="sharding_guide")
print(json.dumps({
    "processes": claims["jax"].process_count(),
    "datarax_is_one_process_batch": bool(claims["datarax_is_one_process_batch"]),
    "grain_is_one_process_batch": bool(claims["grain_is_one_process_batch"]),
    "noise_independent_of_processes": bool(claims["noise_independent_of_processes"]),
    "noise_largest_difference": float(claims["noise_difference"].max()),
    "fits_reach_true_weights": bool(claims["fits_reach_true_weights"]),
}), flush=True)
"""


class GuideReport(TypedDict):
    processes: int
    datarax_is_one_process_batch: bool
    grain_is_one_process_batch: bool
    noise_independent_of_processes: bool
    noise_largest_difference: float
    fits_reach_true_weights: bool


@pytest.mark.slow
def test_the_guide_run_as_two_processes_holds_its_claims_in_each() -> None:
    reports = run_processes(PROGRAM, GuideReport, str(GUIDE))

    for report in reports:
        assert report["processes"] == PROCESSES
        assert report["datarax_is_one_process_batch"]
        assert not report["grain_is_one_process_batch"]
        assert report["noise_independent_of_processes"], report["noise_largest_difference"]
        assert report["fits_reach_true_weights"]
