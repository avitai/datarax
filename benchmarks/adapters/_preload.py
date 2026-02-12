"""Pre-import fixups run before any benchmark adapter is imported.

1. Deep Lake before TensorFlow — avoids fatal OpenSSL conflict.
   Deep Lake's native extension (aws-c-cal) performs a strict OpenSSL version
   assertion that fails if TensorFlow's bundled BoringSSL is loaded first.
   Reference: https://github.com/awslabs/aws-c-cal/blob/main/source/unix/openssl_platform_init.c

2. TensorFlow GPU memory growth — prevents TF from pre-allocating all VRAM,
   which starves JAX/DALI and can cause SIGSEGV on multi-framework processes.

3. JAX CUDA backend init — JAX's XLA runtime must create its CUDA context
   before DALI or TF touch the GPU. On some NVIDIA drivers (580+), creating
   an XLA context after DALI's context causes a segfault in libcuda.so.

This module is safe to import multiple times (Python caches the import).

Used by:
- tests/conftest.py (before TF config at test collection time)
- benchmarks/adapters/__init__.py (before peer adapter imports)
"""

import os

# --- Ray isolation (must be set before `import ray`) ---
# SkyPilot sets RAY_ADDRESS to its orchestration cluster.  If the benchmark
# connects to that cluster, Ray workers inherit SkyPilot's uv runtime_env
# which lacks benchmark deps, causing silent hangs (Ray #59639).
# Clearing RAY_ADDRESS forces Ray to start a fresh local cluster.
os.environ.pop("RAY_ADDRESS", None)

# Ray 2.53's uv runtime_env hook detects uv-managed venvs and tries to
# recreate them for workers -- but the new venv lacks ray itself, causing
# workers to crash silently and from_numpy()/iter_batches() to hang.
# See: https://github.com/ray-project/ray/issues/59639
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

# --- TF GPU config (must be set before `import tensorflow`) ---
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Deep Lake before TF ---
try:
    import deeplake  # noqa: F401
except ImportError:
    pass

# --- JAX early CUDA init ---
# Trigger XLA's CUDA context creation before any peer framework can.
# `import jax` alone is lazy; calling jax.devices() forces backend init.
try:
    import jax

    jax.devices()
except Exception:
    pass
