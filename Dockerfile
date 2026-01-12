FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml uv.lock README.md LICENSE ./

# Create virtual environment and install dependencies
# We use --system to install directly into the container environment or manage venv explicitly
RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies (dev, gpu, test extras)
RUN uv pip install -e ".[dev,gpu,test]"

# Copy source code
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples

# Verify installation
RUN python -c "import jax; print(f'JAX version: {jax.__version__}, Devices: {jax.devices()}')" || true

# Set entrypoint
ENTRYPOINT ["python", "scripts/distributed_test_runner.py"]
