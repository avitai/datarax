"""Command line interface for running Datarax benchmarks."""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Union

from datarax.benchmarking.pipeline_throughput import (
    BatchSizeBenchmark,
    PipelineBenchmark,
    ProfileReport,
)


def save_benchmark_results(results: Dict, output_path: str) -> None:
    """Save benchmark results to a JSON file.

    Args:
        results: The benchmark results to save.
        output_path: Path to save the results to.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert any non-serializable objects to strings
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

    # Ensure serializable_results is a dictionary
    if not isinstance(serializable_results, dict):
        serializable_results = {"results": serializable_results}

    # Add timestamp
    serializable_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save to file
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Results saved to {output_path}")


def run_pipeline_benchmark(args: argparse.Namespace) -> None:
    """Run a pipeline benchmark based on command line arguments.

    Args:
        args: Command line arguments.
    """
    import importlib.util

    # Load the provided module
    spec = importlib.util.spec_from_file_location("benchmark_module", args.module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {args.module_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check if the setup function exists
    if not hasattr(module, args.setup_function):
        print(f"Error: Function '{args.setup_function}' not found in module")
        sys.exit(1)

    # Get the data stream from the setup function
    setup_function = getattr(module, args.setup_function)
    data_stream = setup_function()

    # Run the benchmark
    benchmark = PipelineBenchmark(
        data_stream,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )

    print(
        f"Running benchmark with {args.num_batches} batches "
        f"and {args.warmup_batches} warmup batches..."
    )

    results = benchmark.run(pipeline_seed=args.seed)
    benchmark.print_results()

    # Save results
    output_path = args.output
    if output_path is None:
        # Create default output path in temp/benchmarks
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("temp/benchmarks", exist_ok=True)
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
            "results": results,
        },
        output_path,
    )


def run_profile(args: argparse.Namespace) -> None:
    """Run a profile report based on command line arguments.

    Args:
        args: Command line arguments.
    """
    import importlib.util

    # Load the provided module
    spec = importlib.util.spec_from_file_location("benchmark_module", args.module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {args.module_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check if the setup function exists
    if not hasattr(module, args.setup_function):
        print(f"Error: Function '{args.setup_function}' not found in module")
        sys.exit(1)

    # Get the data stream from the setup function
    setup_function = getattr(module, args.setup_function)
    data_stream = setup_function()

    # Run the profile
    profile = ProfileReport(data_stream)

    print(f"Running profile with {args.num_batches} batches...")

    profile.run(num_batches=args.num_batches, pipeline_seed=args.seed)
    profile.print_report()

    # Save results
    output_path = args.output
    if output_path is None:
        # Create default output path in temp/benchmarks
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("temp/benchmarks", exist_ok=True)
        output_path = f"temp/benchmarks/profile_report_{timestamp}.json"

    save_benchmark_results(
        {
            "type": "profile_report",
            "module": args.module_path,
            "setup_function": args.setup_function,
            "config": {
                "num_batches": args.num_batches,
                "seed": args.seed,
            },
            "results": profile.metrics,
        },
        output_path,
    )


def run_batch_size_benchmark(args: argparse.Namespace) -> None:
    """Run a batch size benchmark based on command line arguments.

    Args:
        args: Command line arguments.
    """
    import importlib.util

    # Load the provided module
    spec = importlib.util.spec_from_file_location("benchmark_module", args.module_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load module from {args.module_path}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Check if the setup function exists
    if not hasattr(module, args.setup_function):
        print(f"Error: Function '{args.setup_function}' not found in module")
        sys.exit(1)

    # Get the data stream factory
    setup_function = getattr(module, args.setup_function)

    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]

    # Run the benchmark
    benchmark = BatchSizeBenchmark(
        data_stream_factory=setup_function,
        batch_sizes=batch_sizes,
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches,
    )

    print(f"Running batch size benchmark with sizes {batch_sizes}...")

    results = benchmark.run(pipeline_seed=args.seed)
    benchmark.print_results()

    # Save results
    output_path = args.output
    if output_path is None:
        # Create default output path in temp/benchmarks
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs("temp/benchmarks", exist_ok=True)
        output_path = f"temp/benchmarks/batch_size_benchmark_{timestamp}.json"

    save_benchmark_results(
        {
            "type": "batch_size_benchmark",
            "module": args.module_path,
            "setup_function": args.setup_function,
            "config": {
                "batch_sizes": batch_sizes,
                "num_batches": args.num_batches,
                "warmup_batches": args.warmup_batches,
                "seed": args.seed,
            },
            "results": {str(k): v for k, v in results.items()},
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

    # Profile command
    profile_parser = subparsers.add_parser(
        "profile",
        parents=[common_args],
        help="Generate a performance profile report for a pipeline",
    )
    profile_parser.add_argument(
        "--num-batches",
        "-n",
        type=int,
        default=10,
        help="Number of batches to process for the profile",
    )
    profile_parser.set_defaults(func=run_profile)

    # Batch size benchmark command
    batch_size_parser = subparsers.add_parser(
        "batch-size", parents=[common_args], help="Run benchmarks for multiple batch sizes"
    )
    batch_size_parser.add_argument(
        "--batch-sizes",
        "-b",
        required=True,
        help="Comma-separated list of batch sizes to benchmark",
    )
    batch_size_parser.add_argument(
        "--num-batches",
        "-n",
        type=int,
        default=30,
        help="Number of batches to process for each benchmark",
    )
    batch_size_parser.add_argument(
        "--warmup-batches",
        "-w",
        type=int,
        default=5,
        help="Number of batches to process before starting timing",
    )
    batch_size_parser.set_defaults(func=run_batch_size_benchmark)

    # Parse arguments and run the appropriate function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
