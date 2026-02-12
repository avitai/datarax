# Datarax: A Data Pipeline Framework for JAX

[![CI](https://github.com/avitai/datarax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/ci.yml)
[![Test Coverage](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml)
[![codecov](https://codecov.io/gh/avitai/datarax/branch/main/graph/badge.svg)](https://codecov.io/gh/avitai/datarax)
[![Build](https://github.com/avitai/datarax/actions/workflows/build-verification.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/build-verification.yml)
[![Summary](https://github.com/avitai/datarax/actions/workflows/summary.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/summary.yml)

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

---

> **Early Development - API Unstable**
>
> Datarax is in early development and undergoing rapid iteration.
> Breaking changes are expected. Pin to specific commits if stability is required.
> We recommend waiting for a stable release (v1.0) before using Datarax in production.

---

**Datarax** (*Data + Array/JAX*) is a high-performance, extensible data pipeline framework specifically engineered for JAX-based machine learning workflows. It leverages JAX's JIT compilation, automatic differentiation, and hardware acceleration to build efficient, scalable data loading, preprocessing, and augmentation pipelines on CPUs, GPUs, and TPUs.

## Key Features

- **JAX-Native Design:** All core components built on JAX's functional paradigm with Flax NNX module system for state management
- **High Performance:** JIT-compiled pipelines via XLA, with built-in profiling and roofline analysis
- **DAG Execution Engine:** Graph-based pipeline construction with branching, parallel execution, caching, and rebatching nodes
- **Scalability:** Multi-device and multi-host data distribution with device mesh sharding
- **Determinism:** Reproducible pipelines by default using Grain's Feistel cipher shuffling (O(1) memory)
- **Extensibility:** Custom data sources, operators, and augmentation strategies via composable NNX modules
- **Benchmarking Suite:** Comparative benchmarks against 12+ frameworks (Grain, tf.data, PyTorch DataLoader, DALI, Ray Data, and more)
- **Ecosystem Integration:** Works with Flax, Optax, Orbax, HuggingFace Datasets, and TensorFlow Datasets

## Why Datarax?

Datarax's differentiable pipeline architecture enables optimization paradigms that are impossible with traditional data loaders. Here are three real-world examples:

### Learned Augmentation Policy (10,000x Faster Search)
Traditional augmentation search (AutoAugment) requires 15,000 GPU-hours of RL. With datarax's differentiable operators, [DADA-style gradient-based search](examples/advanced/differentiable/01_dada_learned_augmentation_guide.py) achieves the same accuracy in **~0.1 GPU-hours** — because gradients flow through the augmentation pipeline.

### Task-Optimized Image Processing (+30% Detection Accuracy)
Camera ISPs are tuned for human perception, not AI tasks. Datarax's DAG executor lets you [build a differentiable ISP pipeline](examples/advanced/differentiable/02_learned_isp_guide.py) where detection loss backpropagates through every processing stage, automatically optimizing for **what the model actually needs**.

### Cross-Domain Extensibility (Audio Synthesis in 3 Operators)
Datarax isn't just for images. By implementing [3 custom operators for DDSP audio synthesis](examples/advanced/differentiable/03_ddsp_audio_synthesis_guide.py), you get a complete differentiable audio pipeline — with **100x less training data** than neural audio models — proving the framework extends to any domain.

> **Learn more**: [Differentiable Pipeline Examples](docs/examples/advanced/differentiable/)

## Installation

```bash
# Basic installation
pip install datarax

# With data loading support (HuggingFace, TFDS, audio/image libs)
pip install datarax[data]

# With GPU support (CUDA 12)
pip install datarax[gpu]

# Full development installation
pip install datarax[all]
```

### macOS / Apple Silicon

```bash
# macOS CPU mode (recommended)
pip install datarax[all-cpu]
JAX_PLATFORMS=cpu python your_script.py

# Metal GPU acceleration (experimental, M1/M2/M3+)
pip install jax-metal
JAX_PLATFORMS=metal python your_script.py
```

> **Note:** Metal GPU acceleration is community-tested. CI runs on macOS with CPU only.

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.dag.nodes import OperatorNode
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.typing import Element


def normalize(element: Element, key: jax.Array | None = None) -> Element:
    return element.update_data({"image": element.data["image"] / 255.0})


def augment(element: Element, key: jax.Array) -> Element:
    key1, _ = jax.random.split(key)
    flip = jax.random.bernoulli(key1, 0.5)
    new_image = jax.lax.cond(
        flip, lambda img: jnp.flip(img, axis=1), lambda img: img,
        element.data["image"],
    )
    return element.update_data({"image": new_image})


# Create in-memory data source
data = {
    "image": np.random.randint(0, 255, (1000, 28, 28, 1)).astype(np.float32),
    "label": np.random.randint(0, 10, (1000,)).astype(np.int32),
}
source = MemorySource(MemorySourceConfig(), data=data, rngs=nnx.Rngs(0))

# Build pipeline with DAG-based API
normalizer = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=normalize, rngs=nnx.Rngs(0),
)
augmenter = ElementOperator(
    ElementOperatorConfig(stochastic=True, stream_name="augmentations"),
    fn=augment, rngs=nnx.Rngs(42),
)

pipeline = (
    from_source(source, batch_size=32)
    >> OperatorNode(normalizer)
    >> OperatorNode(augmenter)
)

# Process batches
for i, batch in enumerate(pipeline):
    if i >= 3:
        break
    print(f"Batch {i}: images {batch['image'].shape}, labels {batch['label'].shape}")
```

### Advanced: Branching and Parallel DAGs

```python
from datarax.dag.nodes import OperatorNode, Merge, Branch

# Define additional operators
def invert(element: Element, key=None) -> Element:
    return element.update_data({"image": 1.0 - element.data["image"]})

inverter = ElementOperator(
    ElementOperatorConfig(stochastic=False), fn=invert, rngs=nnx.Rngs(0),
)

def is_high_contrast(element):
    return jnp.var(element.data["image"]) > 0.1

# Build a complex DAG:
# 1. Source -> Batching
# 2. Parallel: normalizer AND inverter (| creates a Parallel node)
# 3. Merge: average the two branches
# 4. Branch: conditional path based on image variance
complex_pipeline = (
    from_source(source, batch_size=32)
    >> (OperatorNode(normalizer) | OperatorNode(inverter))
    >> Merge("mean")
    >> Branch(
           condition=is_high_contrast,
           true_path=OperatorNode(augmenter),
           false_path=OperatorNode(normalizer),
       )
)
```

## Architecture

```
src/datarax/
  core/         # Base modules: DataSourceModule, OperatorModule, Element, Batcher, Sampler, Sharder
  dag/          # DAG executor and node system (source, operator, batch, cache, control flow)
  sources/      # MemorySource, TFDS (eager/streaming), HuggingFace (eager/streaming), ArrayRecord, MixedSource
  operators/    # ElementOperator, MapOperator, CompositeOperator, modality-specific (image, text)
    strategies/ # Sequential, Parallel, Branching, Ensemble, Merging execution strategies
  samplers/     # Sequential, Shuffle (Feistel cipher), Range, EpochAware samplers
  sharding/     # ArraySharder, JaxProcessSharder for multi-device distribution
  distributed/  # DeviceMesh, DataParallel for multi-host training
  batching/     # DefaultBatcher with buffer state management
  checkpoint/   # NNXCheckpointHandler with Orbax integration
  monitoring/   # Pipeline monitor, DAG monitor, reporters
  performance/  # Roofline analysis, XLA optimization utilities
  benchmarking/ # Profiler, comparative engine, regression guard, resource monitor
  control/      # Prefetcher for asynchronous data loading
  memory/       # Shared memory manager for multi-process data sharing
  config/       # TOML-based configuration system with schema validation
  cli/          # datarax and datarax-bench CLI entry points
  utils/        # PyTree utilities, external integration helpers
```

## Benchmarking

Datarax includes a benchmarking suite for competitive comparison against 12 data loading frameworks across 25 scenarios spanning vision, NLP, tabular, multimodal, I/O, distributed, and pipeline complexity workloads.

```bash
# Install benchmark dependencies (adds PyTorch, DALI, Ray, etc.)
pip install datarax[benchmark]

# Run benchmarks locally
datarax-bench run --platform cpu --profile ci_cpu --repetitions 5

# Run on cloud (SkyPilot)
sky launch benchmarks/sky/gpu-benchmark.yaml --env WANDB_API_KEY=$WANDB_API_KEY
```

Benchmark results are exported to W&B with charts, gap analysis, stability reports, and raw result artifacts. See [Benchmarking Guide](docs/benchmarks/index.md) for methodology and cloud deployment.

## Development Setup

Datarax uses `uv` as its package manager:

```bash
# Clone and setup
git clone https://github.com/avitai/datarax.git
cd datarax
pip install uv

# Automatic setup
./setup.sh && source activate.sh

# Or manual install
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# CPU-only (most stable)
JAX_PLATFORMS=cpu python -m pytest

# Specific module
JAX_PLATFORMS=cpu python -m pytest tests/sources/test_memory_source.py
```

### Docker

```bash
# Build and run
docker build -t datarax:latest .
docker run --rm --gpus all datarax:latest python -c "import datarax, jax; print(jax.devices())"

# Benchmark images
docker build -f benchmarks/docker/Dockerfile.gpu -t datarax-bench:gpu .
```

See [Docker Guide](docs/contributing/docker.md) for full details.

## Documentation

- [Installation Guide](docs/getting_started/installation.md)
- [Quick Start](docs/getting_started/quick_start.md)
- [Core Concepts](docs/getting_started/core_concepts.md)
- [User Guide](docs/user_guide/)
- [API Reference](docs/api_reference/index.md)
- [Examples](docs/examples/overview.md)
- [Benchmarking](docs/benchmarks/index.md)
- [Contributing](docs/contributing/contributing_guide.md)
- [Docker](docs/contributing/docker.md)

## License

Datarax is licensed under the [MIT License](LICENSE).
