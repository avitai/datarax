"""Command line interface for running Datarax benchmarks.

Uses TimingCollector for framework-agnostic timing (replaces PipelineBenchmark).
"""

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from calibrax.profiling import TimingCollector, TimingSample

from datarax.performance.synchronization import block_until_ready_tree


JSONValue = dict[str, "JSONValue"] | list["JSONValue"] | str | int | float | bool | None


def _emit(message: str = "", *, error: bool = False) -> None:
    """Write CLI output without bypassing the command output boundary."""
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


def _make_sync_fn() -> Callable[[Any], None]:
    """Create a JAX device sync function for accurate GPU timing."""

    def _sync(result: Any) -> None:
        block_until_ready_tree(result)

    return _sync


def _sample_to_dict(sample: TimingSample) -> dict[str, Any]:
    """Convert a TimingSample to a serializable dict."""
    return {
        "wall_clock_sec": sample.wall_clock_sec,
        "num_batches": sample.num_batches,
        "num_elements": sample.num_elements,
        "first_batch_time": sample.first_batch_time,
        "batches_per_second": sample.num_batches / sample.wall_clock_sec
        if sample.wall_clock_sec > 0
        else 0,
        "elements_per_second": sample.num_elements / sample.wall_clock_sec
        if sample.wall_clock_sec > 0
        else 0,
    }


def _emit_timing_summary(sample: TimingSample) -> None:
    """Print a TimingSample in a readable format."""
    bps = sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    eps = sample.num_elements / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    _emit(f"  Wall clock:     {sample.wall_clock_sec:.4f} s")
    _emit(f"  Batches:        {sample.num_batches}")
    _emit(f"  Elements:       {sample.num_elements}")
    _emit(f"  First batch:    {sample.first_batch_time * 1000:.2f} ms")
    _emit(f"  Batches/sec:    {bps:.2f}")
    _emit(f"  Elements/sec:   {eps:.2f}")


def save_benchmark_results_to_path(results: dict, output_path: str) -> None:
    """Save benchmark results to a JSON file."""
    Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(obj: Any) -> JSONValue:
        if isinstance(obj, int | float | str | bool | type(None)):
            return obj
        elif isinstance(obj, list | tuple):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)

    serializable_results = make_serializable(results)
    if not isinstance(serializable_results, dict):
        serializable_results = {"results": serializable_results}

    serializable_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    with Path(output_path).open("w") as f:
        json.dump(serializable_results, f, indent=2)

    _emit(f"Results saved to {output_path}")


def run_pipeline_benchmark(args: argparse.Namespace) -> None:
    """Run a pipeline benchmark using TimingCollector."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_module", args.module_path)
    if spec is None or spec.loader is None:
        _emit(f"Error: Could not load module from {args.module_path}", error=True)
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, args.setup_function):
        _emit(f"Error: Function '{args.setup_function}' not found in module", error=True)
        sys.exit(1)

    setup_function = getattr(module, args.setup_function)
    pipeline = setup_function()

    # Warmup
    _emit(f"Warming up with {args.warmup_batches} batches...")
    for i, _ in enumerate(pipeline):
        if i >= args.warmup_batches - 1:
            break

    # Measure
    _emit(f"Measuring {args.num_batches} batches...")
    collector = TimingCollector(sync_fn=_make_sync_fn())
    sample = collector.measure_iteration(iter(pipeline), num_batches=args.num_batches)

    _emit()
    _emit("Results:")
    _emit_timing_summary(sample)

    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        Path("temp/benchmarks").mkdir(parents=True, exist_ok=True)
        output_path = f"temp/benchmarks/pipeline_benchmark_{timestamp}.json"

    save_benchmark_results_to_path(
        {
            "type": "pipeline_benchmark",
            "module": args.module_path,
            "setup_function": args.setup_function,
            "config": {
                "num_batches": args.num_batches,
                "warmup_batches": args.warmup_batches,
                "seed": args.seed,
            },
            "results": _sample_to_dict(sample),
        },
        output_path,
    )


def main() -> None:
    """Run the Datarax benchmarking CLI."""
    parser = argparse.ArgumentParser(description="Datarax benchmarking tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--module-path",
        "-m",
        required=True,
        help="Path to the Python module containing benchmark setup functions",
    )
    common_args.add_argument(
        "--setup-function",
        "-f",
        required=True,
        help="Name of the function in the module that sets up the data stream",
    )
    common_args.add_argument(
        "--output",
        "-o",
        help="Path to save benchmark results as JSON. "
        "If not provided, results will be saved to temp/benchmarks/",
    )
    common_args.add_argument(
        "--seed", "-s", type=int, default=42, help="Random seed for the benchmark"
    )

    # Pipeline benchmark command
    pipeline_parser = subparsers.add_parser(
        "pipeline", parents=[common_args], help="Run a benchmark on a single pipeline"
    )
    pipeline_parser.add_argument(
        "--num-batches",
        "-n",
        type=int,
        default=50,
        help="Number of batches to process for the benchmark",
    )
    pipeline_parser.add_argument(
        "--warmup-batches",
        "-w",
        type=int,
        default=5,
        help="Number of batches to process before starting timing",
    )
    pipeline_parser.set_defaults(func=run_pipeline_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
