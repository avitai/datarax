"""Datarax file reporting monitoring example.

This example demonstrates how to use the Datarax monitoring system with a
file reporter to save metrics to a CSV file.
"""

import time

import jax.numpy as jnp

from datarax.monitoring.pipeline import MonitoredPipeline
from datarax.monitoring.reporters import ConsoleReporter, FileReporter
from datarax.samplers import RangeSampler


def main():
    """Run the example."""
    print("Datarax File Reporter Example")
    print("----------------------------")
    print("This example demonstrates using FileReporter to save metrics")
    print("to a CSV file")
    print()

    # Define metrics file - will be created in temp/monitoring by default
    metrics_file = "pipeline_metrics.csv"

    print(f"Metrics will be saved to: temp/monitoring/{metrics_file}")
    print()

    # Create a sampler that generates numbers from 0 to 99
    sampler = RangeSampler(start=0, stop=100)

    # Create a monitored pipeline
    pipeline = MonitoredPipeline(sampler)

    # Register both console and file reporters
    console_reporter = ConsoleReporter(report_interval=2.0)
    file_reporter = FileReporter(
        filename=metrics_file,
        report_interval=1.0,
    )

    pipeline.callbacks.register(console_reporter)
    pipeline.callbacks.register(file_reporter)

    # Transform raw numbers into dictionaries with features
    pipeline = pipeline.map(
        lambda x: {
            "value": x * x,
            "original": x,
            "is_even": x % 2 == 0,
            "sqrt": float(jnp.sqrt(jnp.array(x))),
        }
    )

    # Process elements and record custom metrics
    print("Processing data with metrics collection...")
    start_time = time.time()

    total_elements = 0
    for i, element in enumerate(pipeline.iterator()):
        # Each element is a dictionary with our computed features
        try:
            total_elements += 1

            # Record custom metrics if the pipeline has metrics collection
            if hasattr(pipeline, "metrics"):
                # Record element value
                pipeline.metrics.record_metric("value", element["value"], "Elements")

                # Record progress count
                pipeline.metrics.record_metric("processed_count", total_elements, "Progress")

                # Record statistics about even/odd numbers
                if element["is_even"]:
                    pipeline.metrics.record_metric("even_number_found", 1, "Statistics")
                else:
                    pipeline.metrics.record_metric("odd_number_found", 1, "Statistics")

            # Print progress periodically
            if i % 10 == 0:
                print(f"Processed {i} elements...")

            # Add a slight delay to simulate processing time
            time.sleep(0.02)
        except Exception as e:
            print(f"Error processing element {i}: {e}")

    elapsed = time.time() - start_time
    print(f"\nCompleted processing in {elapsed:.2f} seconds")
    print(f"Processed {total_elements} elements")
    output_path = f"temp/monitoring/{metrics_file}"
    print(f"Metrics saved to: {output_path}")

    # Close the file reporter to ensure data is written
    if hasattr(file_reporter, "close"):
        file_reporter.close()


if __name__ == "__main__":
    main()
