#!/usr/bin/env python
"""Check GPU availability and configuration for JAX.

Exit codes:
    0 - GPU available
    1 - No GPU available or error
"""

import sys

import jax


def main() -> int:
    """Check GPU availability and return exit code."""
    print("JAX version:", jax.__version__)
    print("Default backend:", jax.default_backend())
    print("\nAll devices:")
    for device in jax.devices():
        print(f"  - {device}")

    print("\nGPU devices:")
    try:
        gpu_devices = jax.devices("gpu")
        if gpu_devices:
            for device in gpu_devices:
                print(f"  - {device}")
            print(f"\n✅ CUDA available: {len(gpu_devices)} GPU(s) detected")
            return 0
        else:
            print("  (none)")
            print("\n❌ No GPU devices found")
            return 1
    except RuntimeError as e:
        print(f"  Error querying GPU devices: {e}")
        print("\n❌ GPU not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
