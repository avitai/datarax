#!/usr/bin/env python
"""
Distributed test runner for Datarax on Vertex AI / Kubernetes.

This script initializes the JAX distributed system and then runs the test suite.
It is designed to be the entrypoint for distributed containers.
"""

import os
import sys
import json
import logging
import subprocess
import jax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_cloud_env():
    """Load variables from .env.cloud if present."""
    env_path = ".env.cloud"
    if os.path.exists(env_path):
        logger.info(f"Loading environment from {env_path}")
        try:
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, _, value = line.partition("=")
                    if key and value:
                        os.environ.setdefault(key.strip(), value.strip())
        except Exception as e:
            logger.warning(f"Failed to load {env_path}: {e}")


def setup_distributed_environment():
    """Initialize JAX distributed system based on environment variables."""
    load_cloud_env()

    # 1. Native JAX auto-detection (e.g. SLURM)
    # If JAX can auto-detect, jax.distributed.initialize() with no args is best.
    # But for custom cloud jobs, we often need to be explicit.

    coordinator_address = os.environ.get("MASTER_ADDR")
    num_processes = os.environ.get("WORLD_SIZE")
    process_id = os.environ.get("RANK")

    # Vertex AI CLUSTER_SPEC parsing (if applicable)
    if "CLUSTER_SPEC" in os.environ and not coordinator_address:
        logger.info("Parsing CLUSTER_SPEC for Vertex AI...")
        try:
            cluster_spec = json.loads(os.environ["CLUSTER_SPEC"])
            # Assuming 'workerpool0' is the coordinator (master)
            # This logic depends on exact Vertex AI topology
            master_node = cluster_spec["task"][0]["workerpool0"][0]
            coordinator_address = (
                master_node.split(":")[0] + ":" + str(os.environ.get("MASTER_PORT", "1234"))
            )
            # Rank and world size usually set in TF_CONFIG or similar,
            # but standard env vars might be missing in raw Vertex jobs
        except Exception as e:
            logger.warning(f"Failed to parse CLUSTER_SPEC: {e}")

    # Fallback/Defaults
    if coordinator_address is None:
        logger.warning("No distributed coordinator address found. Running in local mode.")
        return

    logger.info(
        f"Initializing JAX Distributed: coordinator={coordinator_address}, "
        f"num_processes={num_processes}, process_id={process_id}"
    )

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=int(num_processes) if num_processes else None,
        process_id=int(process_id) if process_id else None,
    )

    logger.info(
        f"JAX Initialized: process_index={jax.process_index()}, "
        f"process_count={jax.process_count()}, "
        f"local_devices={jax.local_devices()}"
    )


def main():
    setup_distributed_environment()

    # Run pytest
    # Pass all arguments from script to pytest
    pytest_args = sys.argv[1:]
    if not pytest_args:
        # Default to running all tests if no args
        pytest_args = ["tests/"]

    logger.info(f"Running pytest with args: {pytest_args}")

    try:
        # Using subprocess to invoke pytest ensures clean separation
        # and allows pytest to handle its own exit codes
        subprocess.run(["python", "-m", "pytest", *pytest_args], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Tests failed!")
        sys.exit(e.returncode)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
