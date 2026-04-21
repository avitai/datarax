"""CLI argument parser and entry point for the benchmark orchestrator."""

from __future__ import annotations

import argparse
import sys

from benchmarks.automation._orchestrator_utils import (
    DEFAULT_CAPTURE_LIMIT_CHARS,
    OrchestrationError,
)
from datarax.utils.console import emit


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the orchestrator CLI."""
    parser = argparse.ArgumentParser(description="Automate Vast.ai A100 benchmark runs.")
    parser.add_argument("--infra", default="vast", help="SkyPilot infra provider")
    parser.add_argument("--cluster", default="datarax-vast-a100", help="Sky cluster name")
    parser.add_argument(
        "--mode",
        choices=["two-pass", "subset", "full"],
        default="two-pass",
        help="Execution mode",
    )
    parser.add_argument(
        "--download-dir",
        default="benchmark-data/reports/vast/latest",
        help="Local destination root for downloaded artifacts",
    )
    parser.add_argument(
        "--template",
        default="benchmarks/sky/gpu-benchmark.yaml",
        help="Template Sky config path (relative to repo root)",
    )
    parser.add_argument("--subset-repetitions", type=int, default=3)
    parser.add_argument("--full-repetitions", type=int, default=5)
    parser.add_argument(
        "--allowed-gpu",
        action="append",
        default=None,
        help="Allowed GPU name token for verification (repeatable)",
    )
    parser.add_argument("--analyze", action="store_true", help="Run local analysis after sync")
    parser.add_argument("--keep-cluster", action="store_true", help="Skip automatic teardown")
    parser.add_argument("--run-id", default=None, help="Optional custom run id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate preflight and generate configs without provisioning resources",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Auto-confirm interactive prompts",
    )
    parser.add_argument(
        "--allow-spot-fallback",
        dest="allow_spot_fallback",
        action="store_true",
        help="Allow on-demand launch fallback to spot with confirmation",
    )
    parser.add_argument(
        "--no-spot-fallback",
        dest="allow_spot_fallback",
        action="store_false",
        help="Disable fallback to spot if on-demand launch fails",
    )
    parser.add_argument(
        "--launch-timeout-sec",
        type=int,
        default=300,
        help="Timeout for cluster launch command before failing/fallback",
    )
    parser.add_argument(
        "--stall-timeout-sec",
        type=int,
        default=300,
        help=(
            "Fail fast if launch emits no new output for this many seconds; "
            "runs sky queue/log diagnostics before aborting"
        ),
    )
    parser.add_argument(
        "--capture-limit-chars",
        type=int,
        default=DEFAULT_CAPTURE_LIMIT_CHARS,
        help=(
            "Maximum in-memory captured chars per stdout/stderr stream for command "
            "summaries. Full logs on disk are always complete. Use 0 to disable limit."
        ),
    )
    parser.add_argument(
        "--live-peek",
        dest="live_peek",
        action="store_true",
        help="Show live snippets from long-running command output",
    )
    parser.add_argument(
        "--no-live-peek",
        dest="live_peek",
        action="store_false",
        help="Disable live command output snippets",
    )
    parser.add_argument(
        "--peek-interval-sec",
        type=int,
        default=5,
        help="Minimum seconds between live command output snippets",
    )
    parser.add_argument(
        "--artifact-transfer",
        choices=["auto", "sky-rsync", "scp"],
        default="auto",
        help=(
            "Artifact download method. 'auto' selects based on SkyPilot capabilities "
            "detected in preflight"
        ),
    )

    placement = parser.add_mutually_exclusive_group()
    placement.add_argument(
        "--on-demand",
        dest="on_demand",
        action="store_true",
        help="Use on-demand instances",
    )
    placement.add_argument(
        "--spot",
        dest="on_demand",
        action="store_false",
        help="Use spot instances",
    )
    parser.set_defaults(on_demand=True, allow_spot_fallback=True, live_peek=True)
    return parser


def main() -> None:
    """Run the orchestrator CLI."""
    from benchmarks.automation.vast_orchestrator import orchestrate

    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(orchestrate(args))
    except OrchestrationError as exc:
        emit(f"ERROR: {exc}", flush=True)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        emit("ERROR: Interrupted by user", file=sys.stderr, flush=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
