# Examples Overview

Datarax provides a comprehensive set of examples organized by complexity and topic.
Each example follows a consistent structure with learning goals, prerequisites, and
expected outcomes.

## Quick Start

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Simple Pipeline**

    ---

    Build your first data pipeline in 5 minutes

    [:octicons-arrow-right-24: Quick Reference](core/simple-pipeline.md)

-   :material-database:{ .lg .middle } **HuggingFace Integration**

    ---

    Load datasets from HuggingFace Hub

    [:octicons-arrow-right-24: Quick Reference](integration/huggingface/hf-quickref.md)

</div>

## Example Categories

### Core Pipeline

Essential examples for understanding Datarax fundamentals.

| Example | Level | Description |
|---------|-------|-------------|
| [Simple Pipeline](core/simple-pipeline.md) | Beginner | Basic pipeline with memory source and operators |
| [Pipeline Tutorial](core/pipeline-tutorial.md) | Intermediate | Comprehensive guide to operators and composition |
| [Operators Tutorial](core/operators-tutorial.md) | Intermediate | Deep dive into operator types and patterns |
| [CIFAR-10 Quick Reference](core/cifar10-quickref.md) | Beginner | CIFAR-10 dataset loading and preprocessing |
| [Augmentation Quick Reference](core/augmentation-quickref.md) | Beginner | Image augmentation techniques |
| [MNIST Tutorial](core/mnist-tutorial.md) | Intermediate | Complete MNIST training pipeline |
| [Fashion Augmentation](core/fashion-augmentation-tutorial.md) | Intermediate | Fashion-MNIST with advanced augmentation |
| [Composition Strategies](core/composition-strategies-tutorial.md) | Intermediate | All 11 operator composition patterns |
| [Advanced Operators](core/advanced-operators-tutorial.md) | Intermediate | Probabilistic, selector, and patch dropout operators |

### Integration

Connect Datarax with external data sources and libraries.

| Example | Level | Description |
|---------|-------|-------------|
| [HuggingFace Quick Reference](integration/huggingface/hf-quickref.md) | Beginner | Load datasets from HuggingFace Hub |
| [HuggingFace Tutorial](integration/huggingface/hf-tutorial.md) | Intermediate | Advanced HF usage and training pipelines |
| [IMDB Example](integration/huggingface/imdb-quickref.md) | Beginner | Text classification with IMDB dataset |
| [TFDS Quick Reference](integration/tfds/tfds-quickref.md) | Beginner | Load datasets from TensorFlow Datasets |
| [ArrayRecord Quick Reference](integration/arrayrecord/arrayrecord-quickref.md) | Intermediate | Google's ArrayRecord format integration |

### Advanced

Production-ready patterns and optimization techniques.

| Example | Level | Description |
|---------|-------|-------------|
| [MixUp & CutMix Tutorial](advanced/augmentation/mixup-cutmix-tutorial.md) | Intermediate | Batch-level mixing augmentations |
| [Checkpoint Quick Reference](advanced/checkpointing/checkpoint-quickref.md) | Intermediate | Save and restore pipeline state |
| [Resumable Training Guide](advanced/checkpointing/resumable-training-guide.md) | Advanced | Full checkpointing workflow |
| [DAG Fundamentals Guide](advanced/dag/dag-fundamentals-guide.md) | Advanced | Deep dive into DAG pipeline architecture |
| [Sharding Quick Reference](advanced/distributed/sharding-quickref.md) | Intermediate | Multi-device data distribution |
| [Sharding Guide](advanced/distributed/sharding-guide.md) | Advanced | Advanced distributed training patterns |
| [Monitoring Quick Reference](advanced/monitoring/monitoring-quickref.md) | Beginner | Pipeline metrics and reporting |
| [Interleaved Tutorial](advanced/multi_source/interleaved-tutorial.md) | Intermediate | Multiple data source mixing |
| [Optimization Guide](advanced/performance/optimization-guide.md) | Advanced | Performance tuning and profiling |
| [Sampling Tutorial](advanced/sampling/sampling-tutorial.md) | Intermediate | Sequential, shuffle, range, and epoch-aware samplers |
| [End-to-End CIFAR-10](advanced/training/e2e-cifar10-guide.md) | Advanced | Complete training pipeline with all features |

## Documentation Tiers

Datarax examples follow a three-tier documentation pattern:

### Tier 1: Quick Reference (~5-10 min)

- Minimal code, maximum clarity
- Single focused concept
- Copy-paste ready snippets
- Ideal for: Getting started, quick lookups

### Tier 2: Tutorial (~30-60 min)

- Step-by-step instruction
- Multiple related concepts
- Hands-on practice exercises
- Ideal for: Learning new features

### Tier 3: Advanced Guide (~60+ min)

- Deep dive into internals
- Performance optimization
- Production considerations
- Ideal for: Expert users, complex use cases

## Feature Coverage

The examples cover all major Datarax features:

| Feature Area | Examples | Coverage |
|--------------|----------|----------|
| Data Sources | Memory, HuggingFace, TFDS, ArrayRecord | Complete |
| Operators | Element, Batch, Probabilistic, Selector, Patch Dropout | Complete |
| Composition | All 11 strategies (Sequential, Parallel, Ensemble, Branching) | Complete |
| Samplers | Sequential, Shuffle, Range, EpochAware | Complete |
| DAG Pipeline | Nodes, Caching, Control Flow, Composition Operators | Complete |
| Distributed | Sharding, Multi-device | Complete |
| Checkpointing | State save/restore, Resumable training | Complete |
| Monitoring | Metrics, Reporters, Callbacks | Complete |

## Running Examples

All examples are available as both Python scripts and Jupyter notebooks.

### As Python Scripts

```bash
# Activate environment
source activate.sh

# Run any example
python examples/core/01_simple_pipeline.py
```

### As Jupyter Notebooks

```bash
# Start Jupyter
uv run jupyter lab

# Navigate to examples/ directory
```

### Generating Notebooks from Scripts

```bash
# Convert a single file
python scripts/jupytext_converter.py py-to-nb examples/core/01_simple_pipeline.py

# Batch convert directory
python scripts/jupytext_converter.py batch-py-to-nb examples/core/
```

## Prerequisites

Before running examples, ensure you have:

1. **Datarax installed**: `uv pip install datarax`
2. **JAX configured**: GPU support recommended for performance
3. **Environment activated**: `source activate.sh`

For external data sources:

- **HuggingFace**: `uv pip install "datarax[data]"`
- **TFDS**: `uv pip install "datarax[data]"`
- **ArrayRecord**: `uv pip install "datarax[data]" array-record`

## For Contributors

Want to add your own examples? We welcome contributions!

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **Documentation Design Guide**

    ---

    Comprehensive standards for creating educational examples and tutorials

    [:octicons-arrow-right-24: Read the Guide](../contributing/example_documentation_design.md)

-   :material-file-document-outline:{ .lg .middle } **Example Template**

    ---

    Start from our template with proper structure and formatting

    [:octicons-arrow-right-24: View Template](https://github.com/avitai/datarax/blob/main/examples/_templates/example_template.py)

</div>

### Quick Start for Contributors

1. Read the [Example Documentation Design Guide](../contributing/example_documentation_design.md)
2. Copy the template from `examples/_templates/example_template.py`
3. Follow the 7-part structure and quality checklist
4. Submit a PR with `.py`, generated `.ipynb`, and `.md` documentation files

## Next Steps

- [API Reference](../core/index.md) - Detailed API documentation
- [Contributing Guide](../contributing/contributing_guide.md) - General contribution guidelines
