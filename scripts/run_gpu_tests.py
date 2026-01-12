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
    """Run examples that were previously disabled due to GPU issues."""
    examples = [
        "examples/rng_test_example.py",
        "examples/advanced_optimized_transforms.py",
    ]

    print("\n" + "=" * 50)
    print("Running previously disabled examples on GPU")
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


def run_gpu_tests():
    """Run pytest with GPU tests enabled."""
    print("\n" + "=" * 50)
    print("Running GPU unit tests")
    print("=" * 50)

    try:
        # Try to install pytest first in case it's not available
        subprocess.run(["pip", "install", "pytest"], check=False)  # nosec B603 B607

        # Set environment to ensure CUDA is used
        env = os.environ.copy()
        env["JAX_PLATFORMS"] = "cuda"

        # Run all tests with device=gpu to enable GPU tests
        # Instead of limiting to specific tests with the gpu marker
        cmd = ["python", "-m", "pytest", "--device=gpu", "-v"]

        # Add specific test files that exist and are relevant for GPU
        test_files = []
        if os.path.exists("tests/transforms/test_optimized_rng_module.py"):
            # This file was previously failing on GPU
            test_files.append("tests/transforms/test_optimized_rng_module.py")

        if os.path.exists("tests/utils/test_hardware_optimizations.py"):
            # This file contains hardware optimization tests (CPU/GPU)
            test_files.append("tests/utils/test_hardware_optimizations.py")

        if test_files:
            cmd.extend(test_files)

        result = subprocess.run(cmd, check=True, env=env)  # nosec B603

        if result.returncode == 0:
            print("✅ GPU tests completed successfully")
        else:
            print("❌ Some GPU tests failed")
    except subprocess.CalledProcessError:
        print("❌ Some GPU tests failed")


if __name__ == "__main__":
    if not check_gpu_availability():
        sys.exit(1)

    run_examples()
    run_gpu_tests()

    print("\n" + "=" * 50)
    print("GPU testing complete")
    print("=" * 50)
