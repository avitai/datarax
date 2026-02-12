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

# System dependencies + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv — single-layer binary copy from official OCI image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Layer 1: Dependencies (cached unless pyproject.toml or uv.lock change) ---
COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install main + dev/gpu/test/data extras (not benchmark, not docs)
RUN uv pip install -e ".[dev,gpu,test,data]"

# --- Layer 2: benchkit path dependency ---
COPY tools/benchkit ./tools/benchkit
RUN uv pip install -e tools/benchkit

# --- Layer 3: Source code (changes frequently, invalidates only this layer) ---
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples
COPY benchmarks ./benchmarks
COPY conftest.py ./conftest.py

# Reinstall datarax in editable mode now that source is present
RUN uv pip install -e ".[dev,gpu,test,data]"

# Verify JAX can import (allow failure on CPU-only build hosts)
RUN python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')" || true

# Default command — overridable at runtime
CMD ["python", "scripts/distributed_test_runner.py"]
