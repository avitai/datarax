# =============================================================================
# Datarax — Development & GPU Runtime Image
# =============================================================================
# For users running datarax pipelines, tests, and examples in a container.
# Does NOT include the `benchmark` extra (adds ~10GB of competing frameworks).
# For benchmarking images, see benchmarks/docker/Dockerfile.{cpu,gpu,tpu}.
#
# Build:  docker build -t datarax:latest .
# Run:    docker run --rm --gpus all datarax:latest python -c "import datarax, jax; print(jax.devices())"
# Test:   docker run --rm -e JAX_PLATFORMS=cpu datarax:latest python -m pytest tests/ -x -q
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# JAX runtime defaults — prevent full GPU memory preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv — single-layer binary copy from official OCI image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Layer 1: Dependencies (cached unless pyproject.toml or uv.lock change) ---
COPY pyproject.toml uv.lock README.md LICENSE ./

# datarax requires Python 3.12+; Ubuntu 22.04 ships 3.10, so uv provides the interpreter.
RUN uv venv --python 3.12 /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install main + dev/cuda12/test/data extras (not benchmark, not docs)
RUN uv pip install -e ".[cuda12,data,dev,test]"

# --- Layer 2: Source code (changes frequently, invalidates only this layer) ---
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples
COPY benchmarks ./benchmarks
COPY conftest.py ./conftest.py

# Reinstall datarax in editable mode now that source is present
RUN uv pip install -e ".[cuda12,data,dev,test]"

# Verify JAX can import (allow failure on CPU-only build hosts)
RUN python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')" || true

# Default command — overridable at runtime
CMD ["python", "scripts/distributed_test_runner.py"]
