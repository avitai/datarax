# Command Line Interface

Command-line tools for running and benchmarking Datarax pipelines without writing Python code.

## Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `datarax run` | Execute pipeline | `datarax run config.yaml` |
| `datarax benchmark` | Measure performance | `datarax benchmark pipeline.yaml` |

`★ Insight ─────────────────────────────────────`

- CLI is useful for quick experiments and CI/CD
- Config files define pipelines declaratively
- Benchmark command includes warmup automatically
- Use Python API for complex workflows

`─────────────────────────────────────────────────`

## Quick Start

```bash
# Run a pipeline from config
datarax run config.yaml

# Benchmark pipeline performance
datarax benchmark config.yaml --batches 100

# Get help
datarax --help
```

## Modules

- [main](main.md) - Main CLI entry point and commands
- [benchmark](benchmark.md) - CLI benchmark tool for performance testing

## Config File Format

```yaml
# config.yaml
source:
  type: hf
  name: mnist
  split: train

pipeline:
  batch_size: 32
  transforms:
    - type: normalize
    - type: augment
      probability: 0.5

output:
  format: tfrecord
  path: ./output
```

## Benchmark Output

```bash
$ datarax benchmark config.yaml --batches 100

Pipeline Benchmark Results
==========================
Warmup batches: 10
Measured batches: 100

Throughput: 1234.56 samples/sec
Latency (mean): 25.6 ms
Latency (p99): 32.1 ms
```

## See Also

- [Config](../config/index.md) - Configuration system
- [Benchmarking](../benchmarking/index.md) - Programmatic benchmarking
- [Installation](../getting_started/installation.md) - Installing CLI
