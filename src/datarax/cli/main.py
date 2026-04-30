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


_CONFIG_LOAD_ERRORS = (OSError, tomllib.TOMLDecodeError)
_PIPELINE_ERRORS = _CONFIG_LOAD_ERRORS + (KeyError, TypeError, ValueError, RuntimeError)


def _emit(message: str = "", *, error: bool = False) -> None:
    """Write CLI output without direct print calls."""
    stream = sys.stderr if error else sys.stdout
    stream.write(f"{message}\n")


def _load_config_from_toml(config_path: str) -> dict[str, Any]:
    """Load TOML configuration from disk."""
    with Path(config_path).open("rb") as f:
        return tomllib.load(f)


def _coerce_override_value(raw_value: str) -> Any:
    """Coerce override value into int/float when possible."""
    for parser in (int, float):
        try:
            return parser(raw_value)
        except ValueError:
            continue
    return raw_value


def _apply_overrides(config: dict[str, Any], overrides: dict[str, str]) -> None:
    """Apply dotted-key overrides to a nested config dictionary."""
    for dotted_key, raw_value in overrides.items():
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = _coerce_override_value(raw_value)


def is_config_valid(config_path: str) -> bool:
    """Validate a pipeline configuration file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        True if valid, False otherwise.
    """
    try:
        config = _load_config_from_toml(config_path)

        # Check required sections
        if "pipeline" not in config:
            _emit("Error: Missing [pipeline] section", error=True)
            return False

        if "name" not in config["pipeline"]:
            _emit("Error: Missing pipeline name", error=True)
            return False

        # Check for DAG format (required)
        if "dag" not in config and "sources" not in config:
            _emit(
                "Error: Missing pipeline definition (no [dag] or [sources] section)",
                error=True,
            )
            return False

        return True
    except _CONFIG_LOAD_ERRORS as e:
        _emit(f"Error validating config: {e}", error=True)
        return False


def run_pipeline(config_path: str, overrides: dict[str, str] | None = None) -> int:
    """Run a pipeline from configuration.

    Args:
        config_path: Path to the pipeline configuration file.
        overrides: Optional configuration overrides.

    Returns:
        Exit code (0 for success).
    """
    del overrides  # config-driven runner removed; overrides no-op
    _emit(
        f"Config-driven pipeline runner is no longer available. "
        f"Construct a Pipeline directly in Python — see "
        f"docs/user_guide/dag_construction.md. "
        f"(Requested config: {config_path})",
        error=True,
    )
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
        from calibrax.cli import main as calibrax_main

        data_dir = str(kwargs.get("data", "benchmark-data"))
        _emit(
            "The legacy 'datarax benchmark' command now delegates to calibrax.\n"
            f"Dataset argument '{dataset}' is ignored; showing summary for store: {data_dir}"
        )
        calibrax_main.main(
            args=["summary", "--data", data_dir],
            standalone_mode=False,
        )
        return 0
    except (ImportError, RuntimeError, ValueError, TypeError) as e:
        _emit(f"Error running benchmark: {e}", error=True)
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


def is_pipeline_template_written_to_path(output_path: str, template: str = "basic") -> bool:
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
        _emit(f"Created pipeline template at {output_path}")
        return True
    except OSError as e:
        _emit(f"Error creating template: {e}", error=True)
        return False


def profile_pipeline(config_path: str, num_iterations: int = 100) -> dict[str, float]:
    """Profile a pipeline's performance.

    Args:
        config_path: Path to pipeline configuration.
        num_iterations: Number of iterations to profile.

    Returns:
        Performance metrics.
    """
    del num_iterations  # config-driven profiler removed
    _emit(
        f"Config-driven pipeline profiler is no longer available. "
        f"Profile a Pipeline directly in Python — see "
        f"benchmarks/migrated_examples_comparison.py for an example. "
        f"(Requested config: {config_path})",
        error=True,
    )
    return {}


def _handle_run(args: argparse.Namespace) -> int:
    """Handle the 'run' subcommand."""
    if not Path(args.config_path).exists():
        _emit(f"Error: Config file not found: {args.config_path}", error=True)
        return 1

    overrides = {}
    if args.override:
        for override in args.override:
            if "=" not in override:
                _emit(
                    f"Error: Invalid override format: {override}. Expected format: key=value",
                    error=True,
                )
                return 1
            key, value = override.split("=", 1)
            overrides[key] = value

    return run_pipeline(args.config_path, overrides=overrides)


def _handle_validate(args: argparse.Namespace) -> int:
    """Handle the 'validate' subcommand."""
    if not Path(args.config_path).exists():
        _emit(f"Error: Config file not found: {args.config_path}", error=True)
        return 1

    if is_config_valid(args.config_path):
        _emit("Configuration is valid")
        return 0
    return 1


def _handle_benchmark(args: argparse.Namespace) -> int:
    """Handle the 'benchmark' subcommand."""
    return run_benchmark(
        args.dataset,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
    )


def _handle_list(args: argparse.Namespace) -> int:
    """Handle the 'list' subcommand."""
    components = list_components(args.type)
    for comp_type, comp_list in components.items():
        _emit()
        _emit(f"{comp_type.capitalize()}:")
        for comp in comp_list:
            _emit(f"  - {comp}")
    return 0


def _handle_create(args: argparse.Namespace) -> int:
    """Handle the 'create' subcommand."""
    return 0 if is_pipeline_template_written_to_path(args.output, args.template) else 1


def _handle_profile(args: argparse.Namespace) -> int:
    """Handle the 'profile' subcommand."""
    if not Path(args.config_path).exists():
        _emit(f"Error: Config file not found: {args.config_path}", error=True)
        return 1

    metrics = profile_pipeline(args.config_path, args.num_iterations)
    return 0 if metrics else 1


def _handle_version(args: argparse.Namespace) -> int:
    """Handle the 'version' subcommand."""
    del args  # unused
    _emit(f"Datarax version {__version__}")
    return 0


def _configure_device() -> None:
    """Configure JAX device from environment variables."""
    device = os.environ.get("DATARAX_DEVICE", "auto")
    if device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif device.startswith("cuda"):
        jax.config.update("jax_platform_name", "gpu")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="datarax",
        description="Datarax: A high-performance data pipeline framework for JAX.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a pipeline defined in a configuration file")
    run_parser.add_argument(
        "--config-path", "-c", required=True, help="Path to the configuration file"
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
        "--config-path", "-c", required=True, help="Path to the configuration file"
    )

    # Command: benchmark
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    benchmark_parser.add_argument(
        "--dataset", "-d", required=True, help="Dataset to benchmark (synthetic, mnist, etc.)"
    )
    benchmark_parser.add_argument(
        "--batch-size", "-b", type=int, default=32, help="Batch size for benchmarking"
    )
    benchmark_parser.add_argument(
        "--num-iterations", "-n", type=int, default=100, help="Number of iterations"
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
    create_parser.add_argument("--output", "-o", required=True, help="Output path for the template")
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
        "--config-path", "-c", required=True, help="Path to the configuration file"
    )
    profile_parser.add_argument(
        "--num-iterations", "-n", type=int, default=100, help="Number of iterations to profile"
    )

    # Command: version
    subparsers.add_parser("version", help="Print the Datarax version")

    return parser


# Command dispatch table
_HANDLERS: dict[str, Any] = {
    "run": _handle_run,
    "validate": _handle_validate,
    "benchmark": _handle_benchmark,
    "list": _handle_list,
    "create": _handle_create,
    "profile": _handle_profile,
    "version": _handle_version,
}


def main(argv: list[str] | None = None) -> int:
    """Execute the Datarax CLI program.

    Args:
        argv: List of command-line arguments. If None, sys.argv is used.

    Returns:
        An exit code (0 for success, non-zero for error).
    """
    _configure_device()

    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = _HANDLERS.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
