"""Command line interface for running Datarax benchmarks.

Uses TimingCollector for framework-agnostic timing (replaces PipelineBenchmark).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Union

import jax.numpy as jnp

from datarax.benchmarking.timing import TimingCollector, TimingSample


def _make_sync_fn():
    """Create a JAX device sync function for accurate GPU timing."""
    return lambda: jnp.array(0.0).block_until_ready()


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


def _print_sample(sample: TimingSample) -> None:
    """Print a TimingSample in a readable format."""
    bps = sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    eps = sample.num_elements / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
    print(f"  Wall clock:     {sample.wall_clock_sec:.4f} s")
    print(f"  Batches:        {sample.num_batches}")
    print(f"  Elements:       {sample.num_elements}")
    print(f"  First batch:    {sample.first_batch_time * 1000:.2f} ms")
    print(f"  Batches/sec:    {bps:.2f}")
    print(f"  Elements/sec:   {eps:.2f}")


def save_benchmark_results(results: dict, output_path: str) -> None:
    """Save benchmark results to a JSON file."""
    Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    def make_serializable(
        obj: Any,
    ) -> Union[dict[str, Any], list[Any], str, int, float, bool, None]:
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

    print(f"Results saved to {output_path}")


def run_pipeline_benchmark(args: argparse.Namespace) -> None:
    """Run a pipeline benchmark using TimingCollector."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("benchmark_module", args.module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {args.module_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, args.setup_function):
        print(f"Error: Function '{args.setup_function}' not found in module")
        sys.exit(1)

    setup_function = getattr(module, args.setup_function)
    pipeline = setup_function()

    # Warmup
    print(f"Warming up with {args.warmup_batches} batches...")
    for i, _ in enumerate(pipeline):
        if i >= args.warmup_batches - 1:
            break

    # Measure
    print(f"Measuring {args.num_batches} batches...")
    collector = TimingCollector(sync_fn=_make_sync_fn())
    sample = collector.measure_iteration(iter(pipeline), num_batches=args.num_batches)

    print("\nResults:")
    _print_sample(sample)

    # Save results
    output_path = args.output
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        Path("temp/benchmarks").mkdir(parents=True, exist_ok=True)
        output_path = f"temp/benchmarks/pipeline_benchmark_{timestamp}.json"

    save_benchmark_results(
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
