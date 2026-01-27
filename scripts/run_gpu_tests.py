#!/usr/bin/env python
"""Script to run previously disabled GPU tests.

This script tests various Datarax components with GPU now that CUDA support is
available.
"""

import os
import subprocess  # nosec B404 - needed for running tests
import sys

import jax


def check_gpu_availability():
    """Check if GPU is available and print device information."""
    gpu_devices = jax.devices("gpu")
    print(f"\nGPU devices: {gpu_devices}")
    if not gpu_devices:
        print("No GPU devices detected. Cannot run GPU tests.")
        return False

    print(f"JAX version: {jax.__version__}")
    print(f"CUDA available: {len(gpu_devices) > 0}")
    print(f"Using device: {jax.devices()[0].platform.upper()}")
    return True


def run_examples():
    """Run GPU-relevant examples from the examples directory."""
    # Find examples in the advanced directory that may benefit from GPU
    example_dirs = [
        "docs/examples/advanced/performance",
        "docs/examples/advanced/distributed",
    ]

    examples = []
    for example_dir in example_dirs:
        if os.path.exists(example_dir):
            for f in os.listdir(example_dir):
                if f.endswith(".py") and not f.startswith("_"):
                    examples.append(os.path.join(example_dir, f))

    if not examples:
        print("\nNo GPU-relevant examples found to run.")
        return

    print("\n" + "=" * 50)
    print("Running examples on GPU")
    print("=" * 50)

    # Set environment to ensure CUDA is used
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cuda"

    for example in examples:
        print(f"\nRunning: {example}")
        try:
            subprocess.run([sys.executable, example], check=True, env=env)  # nosec B603
            print(f"✅ {example} completed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ {example} failed")
        except FileNotFoundError:
            print(f"⚠️ {example} not found, skipping")


def run_gpu_tests():
    """Run pytest with GPU tests enabled."""
    print("\n" + "=" * 50)
    print("Running GPU unit tests")
    print("=" * 50)

    try:
        # Set environment to ensure CUDA is used
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cuda"

        # Run tests with device=gpu to enable GPU execution
        cmd = [sys.executable, "-m", "pytest", "--device=gpu", "-v"]

        # Add specific test files that exist and are relevant for GPU
        test_files = []

        # Hardware optimization tests
        if os.path.exists("tests/utils/test_hardware_optimizations.py"):
            test_files.append("tests/utils/test_hardware_optimizations.py")

        # Distributed tests benefit from GPU
        if os.path.exists("tests/distributed"):
            test_files.append("tests/distributed/")

        # Sharding tests
        if os.path.exists("tests/sharding"):
            test_files.append("tests/sharding/")

        # Benchmarks with GPU markers
        if os.path.exists("tests/benchmarks/test_batch_alignment.py"):
            test_files.append("tests/benchmarks/test_batch_alignment.py")

        if test_files:
            cmd.extend(test_files)
        else:
            # If no specific files, run all tests
            cmd.append("tests/")

        result = subprocess.run(cmd, check=False, env=env)  # nosec B603

        if result.returncode == 0:
            print("✅ GPU tests completed successfully")
        else:
            print("❌ Some GPU tests failed")
            sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"❌ GPU tests failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if not check_gpu_availability():
        sys.exit(1)

    run_examples()
    run_gpu_tests()

    print("\n" + "=" * 50)
    print("GPU testing complete")
    print("=" * 50)
