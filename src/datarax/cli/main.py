"""Datarax CLI main module.

This module provides the main entry point for the Datarax command-line interface.
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

import jax
from datarax import __version__
from datarax.dag import DAGExecutor


def validate_config(config_path: str) -> bool:
    """Validate a pipeline configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        True if valid, False otherwise.
    """
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        # Check required sections
        if "pipeline" not in config:
            print("Error: Missing [pipeline] section", file=sys.stderr)
            return False

        if "name" not in config["pipeline"]:
            print("Error: Missing pipeline name", file=sys.stderr)
            return False

        # Check for DAG format (required)
        if "dag" not in config and "sources" not in config:
            print(
                "Error: Missing pipeline definition (no [dag] or [sources] section)",
                file=sys.stderr,
            )
            return False

        return True
    except Exception as e:
        print(f"Error validating config: {e}", file=sys.stderr)
        return False


def run_pipeline(config_path: str, overrides: dict[str, str] | None = None) -> int:
    """Run a pipeline from configuration.

    Args:
        config_path: Path to the pipeline configuration file.
        overrides: Optional configuration overrides.

    Returns:
        Exit code (0 for success).
    """
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                # Parse nested keys like "pipeline.batch_size"
                keys = key.split(".")
                target = config
                for k in keys[:-1]:
                    if k not in target:
                        target[k] = {}
                    target = target[k]

                # Try to parse value as number
                try:
                    if "." in value:
                        target[keys[-1]] = float(value)
                    else:
                        target[keys[-1]] = int(value)
                except ValueError:
                    # Keep as string
                    target[keys[-1]] = value

        print(f"Loading pipeline: {config['pipeline']['name']}")

        # Create and run pipeline using DAGExecutor
        if "dag" in config:
            # Modern DAG-based pipeline
            executor = DAGExecutor.from_config(config["dag"])
            print("Starting pipeline execution...")

            # Run the pipeline
            for batch_num, batch in enumerate(executor):
                if batch_num % 100 == 0:
                    print(f"Processed batch {batch_num}")
                # In real implementation, we'd process the batch

        else:
            print("Non-DAG pipeline format not supported. Please use DAG format.")
            return 1

        print("Pipeline execution completed successfully")
        return 0

    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        return 1


def run_benchmark(dataset: str, **kwargs: Any) -> int:
    """Run benchmark on a dataset.

    Args:
        dataset: Dataset to benchmark.
        **kwargs: Additional benchmark parameters.

    Returns:
        Exit code.
    """
    try:
        from datarax.benchmarking import run_benchmark as _run_benchmark

        results = _run_benchmark(dataset, **kwargs)
        print(f"Benchmark results: {results}")
        return 0
    except Exception as e:
        print(f"Error running benchmark: {e}", file=sys.stderr)
        return 1


def list_components(component_type: str | None = None) -> dict[str, list]:
    """List available Datarax components.

    Args:
        component_type: Optional specific type to list.

    Returns:
        Dictionary of component types and names.
    """
    components = {
        "sources": ["memory", "tfds", "huggingface", "array_record"],
        "transforms": ["function", "key", "conditional", "parallel"],
        "augmenters": ["brightness", "contrast", "rotate", "noise", "dropout"],
        "samplers": ["sequential", "shuffle", "weighted"],
    }

    if component_type:
        return {component_type: components.get(component_type, [])}
    return components


def create_pipeline_template(output_path: str, template: str = "basic") -> bool:
    """Create a pipeline configuration template.

    Args:
        output_path: Path to write the template.
        template: Template type to create.

    Returns:
        True if successful.
    """
    templates = {
        "basic": """[pipeline]
name = "my_pipeline"

[[nodes]]
id = "source"
type = "DataSource"
class = "MemorySource"

[nodes.params]
num_samples = 1000
sample_shape = [28, 28, 1]

[[nodes]]
id = "batch"
type = "BatchNode"

[nodes.params]
batch_size = 32

[[edges]]
from = "source"
to = "batch"
""",
        "image_classification": """[pipeline]
name = "image_classifier"

[[nodes]]
id = "source"
type = "DataSource"
class = "TFDSEagerSource"

[nodes.params]
name = "mnist"
split = "train"

[[nodes]]
id = "operator"
type = "Operator"
class = "ElementOperator"

[nodes.params]
stochastic = false

[[nodes]]
id = "batch"
type = "BatchNode"

[nodes.params]
batch_size = 32
drop_remainder = true

[[edges]]
from = "source"
to = "operator"

[[edges]]
from = "operator"
to = "batch"
""",
    }

    try:
        template_content = templates.get(template, templates["basic"])
        Path(output_path).write_text(template_content)
        print(f"Created pipeline template at {output_path}")
        return True
    except Exception as e:
        print(f"Error creating template: {e}", file=sys.stderr)
        return False


def profile_pipeline(config_path: str, num_iterations: int = 100) -> dict[str, float]:
    """Profile a pipeline's performance.

    Args:
        config_path: Path to pipeline configuration.
        num_iterations: Number of iterations to profile.

    Returns:
        Performance metrics.
    """
    import time

    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        if "dag" in config:
            executor = DAGExecutor.from_config(config["dag"])

            # Warmup
            for _ in range(10):
                next(iter(executor))

            # Profile
            start_time = time.time()
            for i, batch in enumerate(executor):
                if i >= num_iterations:
                    break

            elapsed = time.time() - start_time
            throughput = num_iterations / elapsed
            latency = elapsed / num_iterations

            metrics = {
                "throughput": throughput,
                "latency": latency,
                "batches_per_second": throughput,
                "ms_per_batch": latency * 1000,
            }

            print("Performance Metrics:")
            print(f"  Throughput: {throughput:.2f} batches/sec")
            print(f"  Latency: {latency * 1000:.2f} ms/batch")

            return metrics

    except Exception as e:
        print(f"Error profiling pipeline: {e}", file=sys.stderr)
        return {}


def main(argv: list[str] | None = None) -> int:
    """Execute the Datarax CLI program.

    Args:
        argv: List of command-line arguments. If None, sys.argv is used.

    Returns:
        An exit code (0 for success, non-zero for error).
    """
    # Check environment variables
    os.environ.get("DATARAX_LOG_LEVEL", "INFO")
    device = os.environ.get("DATARAX_DEVICE", "auto")

    if device != "auto":
        # Set JAX device
        if device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        elif device.startswith("cuda"):
            jax.config.update("jax_platform_name", "gpu")

    parser = argparse.ArgumentParser(
        prog="datarax",
        description="Datarax: A high-performance data pipeline framework for JAX.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a pipeline defined in a configuration file")
    run_parser.add_argument(
        "--config-path",
        "-c",
        required=True,
        help="Path to the configuration file",
    )
    run_parser.add_argument(
        "--override",
        "-o",
        action="append",
        help="Override a configuration value (format: key=value)",
    )

    # Command: validate
    validate_parser = subparsers.add_parser("validate", help="Validate a pipeline configuration")
    validate_parser.add_argument(
        "--config-path",
        "-c",
        required=True,
        help="Path to the configuration file",
    )

    # Command: benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Dataset to benchmark (synthetic, mnist, etc.)",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=32,
        help="Batch size for benchmarking",
    )
    benchmark_parser.add_argument(
        "--num-iterations",
        "-n",
        type=int,
        default=100,
        help="Number of iterations",
    )

    # Command: list
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument(
        "--type",
        "-t",
        choices=["sources", "transforms", "augmenters", "samplers"],
        help="Component type to list",
    )

    # Command: create
    create_parser = subparsers.add_parser("create", help="Create a pipeline template")
    create_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output path for the template",
    )
    create_parser.add_argument(
        "--template",
        "-t",
        default="basic",
        choices=["basic", "image_classification"],
        help="Template type",
    )

    # Command: profile
    profile_parser = subparsers.add_parser("profile", help="Profile pipeline performance")
    profile_parser.add_argument(
        "--config-path",
        "-c",
        required=True,
        help="Path to the configuration file",
    )
    profile_parser.add_argument(
        "--num-iterations",
        "-n",
        type=int,
        default=100,
        help="Number of iterations to profile",
    )

    # Command: version
    subparsers.add_parser("version", help="Print the Datarax version")

    args = parser.parse_args(argv)

    if args.command == "run":
        if not os.path.exists(args.config_path):
            print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
            return 1

        # Parse overrides
        overrides = {}
        if args.override:
            for override in args.override:
                if "=" not in override:
                    print(
                        f"Error: Invalid override format: {override}. Expected format: key=value",
                        file=sys.stderr,
                    )
                    return 1
                key, value = override.split("=", 1)
                overrides[key] = value

        return run_pipeline(args.config_path, overrides=overrides)

    elif args.command == "validate":
        if not os.path.exists(args.config_path):
            print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
            return 1

        if validate_config(args.config_path):
            print("Configuration is valid")
            return 0
        else:
            return 1

    elif args.command == "benchmark":
        return run_benchmark(
            args.dataset,
            batch_size=args.batch_size,
            num_iterations=args.num_iterations,
        )

    elif args.command == "list":
        components = list_components(args.type)
        for comp_type, comp_list in components.items():
            print(f"\n{comp_type.capitalize()}:")
            for comp in comp_list:
                print(f"  - {comp}")
        return 0

    elif args.command == "create":
        if create_pipeline_template(args.output, args.template):
            return 0
        else:
            return 1

    elif args.command == "profile":
        if not os.path.exists(args.config_path):
            print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
            return 1

        metrics = profile_pipeline(args.config_path, args.num_iterations)
        return 0 if metrics else 1

    elif args.command == "version":
        print(f"Datarax version {__version__}")
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
