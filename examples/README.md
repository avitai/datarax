# Development Examples & Utilities

> **Looking for tutorials?** See the [Documentation Examples](https://datarax.readthedocs.io/examples/overview/) or browse `docs/examples/` directly.

This directory contains **development examples and utilities** that are not part of the main documentation. These are more advanced scripts, testing utilities, and framework comparisons.

## Directory Structure

```
examples/
├── comparison/              # Framework comparison scripts
├── config/                  # Configuration system examples
├── monitoring/              # Advanced monitoring examples
├── utils/                   # Shared utilities for examples
├── _templates/              # Example templates
├── custom_benchmark.py      # Benchmarking utilities
├── hf_model_training_example.py  # Full HF model training workflow
└── hf_datasets_test.py      # HuggingFace datasets testing
```

## Running Examples

```bash
# Activate the development environment
source ./activate.sh

# Run any example
python examples/custom_benchmark.py
python examples/hf_model_training_example.py
python examples/config/config_example.py
```

## Contents

### Benchmarking (`custom_benchmark.py`)
Advanced benchmarking utilities for measuring pipeline performance:
- `PipelineBenchmark` for single pipeline performance measurement
- `BatchSizeBenchmark` for comparing different batch sizes
- Performance profiling and comparison tools

### HuggingFace Training (`hf_model_training_example.py`)
Complete model training workflow with HuggingFace Datasets:
- End-to-end training pipeline
- Dataset loading and preprocessing
- Training loop integration

### Configuration (`config/`)
Configuration system examples demonstrating:
- YAML/TOML configuration loading
- Registry-based component instantiation
- Environment variable handling

### Monitoring (`monitoring/`)
Advanced monitoring examples:
- `file_reporter_example.py` - File-based metrics reporting

### Comparison (`comparison/`)
Framework comparison scripts demonstrating the architectural differences between
Datarax's NNX-based stateful approach and Google Grain's stateless framework.
See `comparison/README.md` for detailed documentation.

### Utilities (`utils/`)
Shared utilities used by multiple examples:
- Sample data generation
- Common helper functions

## For Documentation Examples

The official tutorial examples are in `docs/examples/` and include:
- **Core tutorials**: Simple pipeline, operators tutorial
- **Integration guides**: HuggingFace, TensorFlow Datasets
- **Advanced topics**: Distributed processing, checkpointing, monitoring

These are the examples rendered in the documentation website.
