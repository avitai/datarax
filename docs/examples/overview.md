# Examples Overview

Datarax provides a comprehensive set of examples organized by complexity and topic.
Each example follows a consistent structure with learning goals, prerequisites, and
expected outcomes.

## Quick Start

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Simple Pipeline**

    ---

    Build your first data pipeline in 5 minutes

    [:octicons-arrow-right-24: Quick Reference](core/01_simple_pipeline.ipynb)

-   :material-database:{ .lg .middle } **HuggingFace Integration**

    ---

    Load datasets from HuggingFace Hub

    [:octicons-arrow-right-24: Quick Reference](integration/huggingface/01_hf_quickref.ipynb)

</div>

## Example Categories

### Core Pipeline

Essential examples for understanding Datarax fundamentals.

| Example | Level | Description |
|---------|-------|-------------|
| [Simple Pipeline](core/01_simple_pipeline.ipynb) | Beginner | Basic pipeline with memory source and operators |
| [Pipeline Tutorial](core/02_pipeline_tutorial.ipynb) | Intermediate | Comprehensive guide to operators and composition |
| [Operators Tutorial](core/03_operators_tutorial.ipynb) | Intermediate | Deep dive into operator types and patterns |

### Integration

Connect Datarax with external data sources and libraries.

| Example | Level | Description |
|---------|-------|-------------|
| [HuggingFace Quick Reference](integration/huggingface/01_hf_quickref.ipynb) | Beginner | Load datasets from HuggingFace Hub |
| [HuggingFace Tutorial](integration/huggingface/02_hf_tutorial.ipynb) | Intermediate | Advanced HF usage and training pipelines |
| [TFDS Quick Reference](integration/tfds/01_tfds_quickref.ipynb) | Beginner | Load datasets from TensorFlow Datasets |

### Advanced

Production-ready patterns and optimization techniques.

| Example | Level | Description |
|---------|-------|-------------|
| [Sharding Quick Reference](advanced/distributed/01_sharding_quickref.ipynb) | Intermediate | Multi-device data distribution |
| [Checkpointing Quick Reference](advanced/checkpointing/01_checkpoint_quickref.ipynb) | Intermediate | Save and restore pipeline state |
| [Monitoring Quick Reference](advanced/monitoring/01_monitoring_quickref.ipynb) | Beginner | Pipeline metrics and reporting |

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

## Next Steps

- [API Reference](../core/index.md) - Detailed API documentation
- [Contributing Guide](../contributing/contributing_guide.md) - Add your own examples
