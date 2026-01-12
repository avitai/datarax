#!/usr/bin/env python

"""Example demonstrating Datarax's TOML configuration system.

This script shows how to load and use TOML configuration files for Datarax
pipelines, including environment variable overrides and schema validation.
"""

import argparse
import os
from pathlib import Path

import datarax.config as config


def main():
    """Run the configuration example."""
    parser = argparse.ArgumentParser(description="Demonstrate Datarax TOML configuration system")
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline_example.toml",
        help="Path to the TOML configuration file",
    )
    args = parser.parse_args()

    # Get the absolute path to the configuration file
    config_path = Path(__file__).parent / args.config

    print(f"Loading configuration from: {config_path}")
    print("-" * 80)

    # Load the configuration file
    cfg = config.load_toml(config_path)

    print("Original configuration:")
    print_config(cfg)
    print("-" * 80)

    # Set some environment variables for demonstration
    os.environ["DATARAX_BATCH_SIZE"] = "64"
    os.environ["DATARAX_SOURCES__IMAGES__PATH"] = "/data/custom/images"

    # Apply environment variable overrides
    cfg_with_env = config.apply_environment_overrides(cfg)

    print("Configuration with environment overrides:")
    print_config(cfg_with_env)
    print("-" * 80)

    # Define a simple schema for validation
    from datarax.config.schema import PipelineSchema

    try:
        # Validate the configuration against the pipeline schema
        validated_cfg = PipelineSchema.create(cfg_with_env)

        print("Validated configuration:")
        print_config(validated_cfg)
        print("-" * 80)

        print("Validation successful!")
    except Exception as e:
        print(f"Validation error: {e}")


def print_config(cfg, prefix=""):
    """Print a nested configuration dictionary in a readable format.

    Args:
        cfg: The configuration dictionary to print
        prefix: Prefix for nested keys
    """
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, prefix + "  ")
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                print(f"{prefix}{key}:")
                for i, item in enumerate(value):
                    print(f"{prefix}  [{i}]:")
                    print_config(item, prefix + "    ")
            else:
                print(f"{prefix}{key} = {value}")
        else:
            print(f"{prefix}{key} = {value}")


if __name__ == "__main__":
    main()
