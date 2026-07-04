# Development Examples & Utilities

> **Looking for tutorials?** See the [Documentation Examples](https://datarax.readthedocs.io/examples/overview/) or browse `docs/examples/` directly.

This directory contains **development examples and utilities** that are not part of the main documentation. These are more advanced scripts, testing utilities, and framework comparisons.

## Directory Structure

```
examples/
├── core/                         # Core pipeline tutorials and quickrefs
├── advanced/                     # Advanced topics (augmentation, checkpointing,
│                                 #   dag, differentiable, distributed,
│                                 #   multi_source, performance, sampling, training)
├── integration/                  # Source integrations (arrayrecord, huggingface, tfds)
├── comparison/                   # Framework comparison scripts
├── config/                       # Configuration system examples
├── utils/                        # Shared utilities for examples
├── _templates/                   # Example templates
├── custom_benchmark.py           # Benchmarking utilities
├── hf_model_training_example.py  # Full HF model training workflow
├── hf_datasets_test.py           # HuggingFace datasets testing
├── run_all_examples_on_gpu.sh    # Run every example on GPU
└── run_gpu_examples.sh           # Run the selected GPU example set
```

## Running Examples

```bash
# Activate the development environment
source ./activate.sh

# Run any example
python examples/custom_benchmark.py
python examples/hf_model_training_example.py
python examples/config/config_example.py --config pipeline_example.toml
```

> **Note:** `config/config_example.py` reads a TOML file via `--config`. The path
> is resolved relative to `examples/config/`, and it must point at an existing
> TOML file (there is no default file shipped in the directory).

## Contents

### Core (`core/`)
Core pipeline tutorials and quickrefs covering pipeline construction, operators,
and dataset-specific walkthroughs (CIFAR-10, MNIST, Fashion-MNIST).

### Advanced (`advanced/`)
Deeper guides grouped by topic: augmentation, checkpointing, DAG construction,
differentiable pipelines, distributed sharding, multi-source interleaving,
performance optimization, sampling, and end-to-end training.

### Integration (`integration/`)
Source integration examples for ArrayRecord, HuggingFace Datasets, and
TensorFlow Datasets. See `integration/README.md` for details.

### Benchmarking (`custom_benchmark.py`)
Measures pipeline throughput using calibrax's `TimingCollector` and `rank_table`.
Running the script executes three benchmarks in sequence:
- `run_pipeline_benchmark()` — times a basic single-stage pipeline and reports
  wall-clock time, batches/sec, and elements/sec.
- `run_comparison_benchmark()` — times a basic pipeline against an augmented one
  (random flip + a simulated heavy transform) and ranks them by throughput.
- `run_batch_size_benchmark()` — sweeps batch sizes (8, 16, 32, 64, 128) and
  reports elements/sec and wall-clock time for each.

### HuggingFace Training (`hf_model_training_example.py`)
Complete model training workflow with HuggingFace Datasets:
- End-to-end training pipeline
- Dataset loading and preprocessing
- Training loop integration

### Configuration (`config/`)
Configuration system examples demonstrating:
- TOML configuration loading (`load_toml_from_path`)
- Environment variable overrides (`apply_environment_overrides`)
- Schema validation with `PipelineSchema`

### Comparison (`comparison/`)
Framework comparison scripts demonstrating the architectural differences between
Datarax's NNX-based stateful approach and Google Grain's stateless framework.
See `comparison/README.md` for detailed documentation.

### Utilities (`utils/`)
Shared utilities used by multiple examples:
- Sample data generation (`sample_data.py`)

## For Documentation Examples

The official tutorial examples are in `docs/examples/` and include:
- **Core tutorials**: Simple pipeline, operators tutorial
- **Integration guides**: HuggingFace, TensorFlow Datasets
- **Advanced topics**: Distributed processing, checkpointing, performance optimization

These are the examples rendered in the documentation website.
