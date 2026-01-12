#!/usr/bin/env python3
"""Script for running Datarax benchmarks with the benchmarking utilities.

This script provides a convenient way to run benchmarks on Datarax pipelines
using the benchmark utilities.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np

from datarax import from_source
from datarax.sources import MemorySource, MemorySourceConfig


from datarax.benchmarking.pipeline_throughput import (
    BatchSizeBenchmark,
    PipelineBenchmark,
    ProfileReport,
)


def setup_example_pipeline():
    """Create an example pipeline for demonstration purposes."""

    # Generate some sample data
    # MemorySource needs a dict of arrays/lists
    data = {"value": np.arange(1000)}

    # Create a simple pipeline
    source_config = MemorySourceConfig(shuffle=False)
    source = MemorySource(config=source_config, data=data)

    # from_source returns a DAGExecutor
    # batch() adds a batch node and returns self (DAGExecutor)
    pipeline = from_source(source, enforce_batch=False).batch(batch_size=32)

    return pipeline


def main():
    """Run benchmarks."""
    parser = argparse.ArgumentParser(description="Run Datarax benchmarks")
    parser.add_argument(
        "--output-dir",
        "-o",
        default="temp/benchmarks",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--num-batches", "-n", type=int, default=50, help="Number of batches to process"
    )
    parser.add_argument("--run-all", "-a", action="store_true", help="Run all benchmark types")
    parser.add_argument("--pipeline", "-p", action="store_true", help="Run pipeline benchmark")
    parser.add_argument("--profile", "-f", action="store_true", help="Run profile report")
    parser.add_argument("--batch-size", "-b", action="store_true", help="Run batch size benchmark")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for benchmarks")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine which benchmarks to run
    run_pipeline = args.run_all or args.pipeline
    run_profile = args.run_all or args.profile
    run_batch_size = args.run_all or args.batch_size

    # If no benchmark specified, run pipeline benchmark
    if not (run_pipeline or run_profile or run_batch_size):
        run_pipeline = True

    # Run pipeline benchmark
    if run_pipeline:
        print("Running pipeline benchmark...")
        pipeline = setup_example_pipeline()
        benchmark = PipelineBenchmark(pipeline, num_batches=args.num_batches, warmup_batches=5)
        results = benchmark.run(pipeline_seed=args.seed)
        benchmark.print_results()

        # Save results
        output_path = os.path.join(args.output_dir, f"pipeline_benchmark_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Pipeline benchmark results saved to {output_path}")

    # Run profile report
    if run_profile:
        print("Running profile report...")
        pipeline = setup_example_pipeline()
        profile = ProfileReport(pipeline)
        profile.run(num_batches=args.num_batches, pipeline_seed=args.seed)
        profile.print_report()

        # Save results
        output_path = os.path.join(args.output_dir, f"profile_report_{timestamp}.json")
        with open(output_path, "w") as f:
            json.dump(profile.metrics, f, indent=2)
        print(f"Profile report saved to {output_path}")

    # Run batch size benchmark
    if run_batch_size:
        print("Running batch size benchmark...")

        # Create factory function for data stream with different batch sizes
        def data_stream_factory(batch_size):
            # Generate some sample data

            data = {"value": np.arange(1000)}

            # Create a simple pipeline with the specified batch size
            source_config = MemorySourceConfig(shuffle=False)
            source = MemorySource(config=source_config, data=data)

            pipeline = from_source(source, enforce_batch=False).batch(batch_size=batch_size)

            return pipeline

        batch_sizes = [1, 8, 16, 32, 64, 128, 256]
        benchmark = BatchSizeBenchmark(
            data_stream_factory=data_stream_factory,
            batch_sizes=batch_sizes,
            num_batches=args.num_batches,
            warmup_batches=5,
        )
        results = benchmark.run(pipeline_seed=args.seed)
        benchmark.print_results()

        # Save results
        output_path = os.path.join(args.output_dir, f"batch_size_benchmark_{timestamp}.json")
        with open(output_path, "w") as f:
            # Convert keys to strings for JSON serialization
            serializable_results = {str(k): v for k, v in results.items()}
            json.dump(serializable_results, f, indent=2)
        print(f"Batch size benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
