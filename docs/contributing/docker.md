# Docker

Datarax provides Docker images for two audiences:

1. **Development & Runtime** — run pipelines, tests, and examples in a GPU-enabled container
2. **Benchmarking** — run competitive benchmarks across CPU, GPU, and TPU platforms

## Images Overview

| Image | Dockerfile | Base | Extras | Size |
|-------|-----------|------|--------|------|
| `datarax:latest` | `Dockerfile` | `nvidia/cuda:12.4.1-cudnn-runtime` | `dev,gpu,test,data` | ~4GB |
| `datarax-bench:cpu` | `benchmarks/docker/Dockerfile.cpu` | `python:3.11-slim` | `benchmark` | ~8GB |
| `datarax-bench:gpu` | `benchmarks/docker/Dockerfile.gpu` | `nvidia/cuda:12.4.1-cudnn-runtime` | `benchmark,gpu` | ~12GB |
| `datarax-bench:tpu` | `benchmarks/docker/Dockerfile.tpu` | `python:3.11-slim` | `benchmark` + `jax[tpu]` | ~8GB |

!!! note
    The root image intentionally excludes the `benchmark` extra, which adds PyTorch, NVIDIA DALI, Ray, MosaicML, and other competing frameworks (~10GB). Use the benchmark-specific images for competitive benchmarking.

## Building Images

### Development Image

```bash
docker build -t datarax:latest .
```

### Benchmark Images

```bash
# CPU benchmarks
docker build -f benchmarks/docker/Dockerfile.cpu -t datarax-bench:cpu .

# GPU benchmarks (requires NVIDIA GPU for runtime, not for build)
docker build -f benchmarks/docker/Dockerfile.gpu -t datarax-bench:gpu .

# TPU benchmarks (runs on GCE TPU VMs)
docker build -f benchmarks/docker/Dockerfile.tpu -t datarax-bench:tpu .
```

!!! tip
    The `.dockerignore` file excludes `.venv/`, `.git/`, design docs, and other non-runtime files, keeping the build context under 500MB.

## Running Containers

### Development

```bash
# Interactive Python with GPU
docker run --rm -it --gpus all datarax:latest python

# Run tests on CPU
docker run --rm -e JAX_PLATFORMS=cpu datarax:latest \
    python -m pytest tests/ -x --timeout=60 -m "not gpu and not slow" -q

# Run tests with GPU
docker run --rm --gpus all datarax:latest \
    python -m pytest tests/ -x --timeout=120 -q

# Run a specific example
docker run --rm --gpus all datarax:latest \
    python examples/core/02_pipeline_tutorial.py

# CLI tools
docker run --rm datarax:latest datarax --help
docker run --rm datarax:latest datarax-bench --help
```

### Benchmarking

```bash
# CPU benchmarks with simulated 4 devices
docker run --rm datarax-bench:cpu

# GPU benchmarks on all available GPUs
docker run --rm --gpus all datarax-bench:gpu

# Save results to host
docker run --rm --gpus all -v $(pwd)/results:/app/results \
    datarax-bench:gpu python -m benchmarks.runners.full_runner \
    --platform gpu --output /app/results/
```

### GPU Passthrough

GPU access requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
# All GPUs
docker run --rm --gpus all datarax:latest python -c "import jax; print(jax.devices())"

# Specific GPU
docker run --rm --gpus '"device=0"' datarax:latest python -c "import jax; print(jax.devices())"
```

## Environment Variables

Key variables for controlling JAX behavior inside containers:

| Variable | Default | Description |
|----------|---------|-------------|
| `JAX_PLATFORMS` | (auto) | Force platform: `cpu`, `cuda`, `tpu` |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `false` | Disable full GPU memory grab at startup |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.75` | Fraction of GPU memory JAX may use |
| `XLA_FLAGS` | (none) | XLA compiler flags (e.g., simulated devices) |

## Use Cases

### Vertex AI

The root image works directly with Google Cloud Vertex AI custom training:

```bash
# Tag and push
docker tag datarax:latest gcr.io/PROJECT_ID/datarax:latest
docker push gcr.io/PROJECT_ID/datarax:latest

# Submit training job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=datarax-test \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_A100,accelerator-count=1,container-image-uri=gcr.io/PROJECT_ID/datarax:latest
```

### SkyPilot

SkyPilot GPU benchmarks use the config at `benchmarks/sky/gpu-benchmark.yaml`:

```bash
sky launch benchmarks/sky/gpu-benchmark.yaml
```

This installs `.[benchmark,gpu]` on the provisioned VM directly (no Docker needed — SkyPilot manages the environment).

## Image Tagging Convention

| Tag | Description |
|-----|-------------|
| `datarax:latest` | Development/runtime image (GPU-enabled) |
| `datarax-bench:cpu` | Benchmark image for CPU |
| `datarax-bench:gpu` | Benchmark image for GPU |
| `datarax-bench:tpu` | Benchmark image for TPU |
