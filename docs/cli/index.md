# Command Line Interface

Command-line helpers for validating configs, scaffolding templates, and
inspecting a Datarax installation. Pipelines themselves are built in Python
(see the [DAG Construction Guide](../user_guide/dag_construction.md)).

## Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `datarax validate` | Validate a pipeline config | `datarax validate -c pipeline.toml` |
| `datarax create` | Scaffold a config template | `datarax create -o pipeline.toml -t basic` |
| `datarax list` | List available components | `datarax list --type sources` |
| `datarax benchmark` | Show a calibrax store summary | `datarax benchmark --dataset synthetic` |
| `datarax version` | Print the installed version | `datarax version` |

!!! note "Key points"

    - Config files are TOML and are validated, not executed, by the CLI
    - `datarax benchmark` delegates to [calibrax](https://github.com/avitai/calibrax)
      and prints a store summary; for comparative benchmarks use
      `uv run python -m benchmarks.cli run`
    - Build and run pipelines with the Python API

## Quick Start

```bash
# Scaffold a template, then validate it
datarax create --output pipeline.toml --template basic
datarax validate --config-path pipeline.toml

# List the registered source components
datarax list --type sources

# Print the installed version
datarax version
```

## Modules

- [main](main.md) - Main CLI entry point and commands
- [benchmark](benchmark.md) - `python -m datarax.cli.benchmark` timing tool

## Config File Format

`datarax create` writes a TOML template describing the pipeline as a list of
nodes and edges:

```toml
[pipeline]
name = "my_pipeline"

[[nodes]]
id = "source"
type = "DataSource"
class = "MemorySource"

[nodes.params]
num_samples = 1000
sample_shape = [28, 28, 1]

[[nodes]]
id = "batch"
type = "BatchNode"

[nodes.params]
batch_size = 32

[[edges]]
from = "source"
to = "batch"
```

`datarax validate` accepts this `[[nodes]]` layout as well as configs that
declare a `[dag]` or `[sources]` section.

## See Also

- [Config](../config/index.md) - Configuration system
- [Benchmarking](../benchmarking/index.md) - Programmatic benchmarking
- [Installation](../getting_started/installation.md) - Installing the CLI
