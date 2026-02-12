# API Reference

Complete auto-generated API documentation for all Datarax modules. Each page is built from source docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

## Module Groups

| Section | Description |
|---------|-------------|
| [Core Components](../core/index.md) | Data structures, operators, samplers, and module base classes |
| [Operators](../operators/index.md) | Data transformation operators (image, composition strategies) |
| [Sources](../sources/index.md) | Data source implementations (HuggingFace, TFDS, ArrayRecord, Memory) |
| [DAG](../dag/index.md) | Directed Acyclic Graph pipeline execution engine |
| [Distributed](../distributed/index.md) | Multi-device sharding, data parallelism, device mesh |
| [Benchmarking](../benchmarking/index.md) | Timing, profiling, regression detection, monitoring |
| [Monitoring](../monitoring/index.md) | Runtime metrics collection, callbacks, reporters |
| [Checkpoint](../checkpoint/index.md) | Pipeline state saving and restoration |
| [Config](../config/index.md) | Configuration loading, schema, environment, registry |
| [Samplers](../samplers/index.md) | Data sampling strategies (sequential, shuffle, range, epoch-aware) |
| [Utilities](../utils/index.md) | PRNG, pytree utilities, external helpers |
| [CLI](../cli/index.md) | Command-line interface entry points |

## How It Works

API pages use the `:::` directive to auto-generate documentation from Python docstrings:

```markdown
::: datarax.core.operator
```

This pulls class signatures, method documentation, type annotations, and inheritance hierarchies directly from the source code. Google-style docstrings are parsed automatically.
