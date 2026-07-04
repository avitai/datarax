# Config

Configuration management and validation system. Load configurations from
TOML files, apply environment-variable overrides, and validate against
schemas.

## Components

| Component | Purpose | Formats |
|-----------|---------|---------|
| **Loaders** | Load from files | TOML (with includes) |
| **Environment** | Env var overrides | `DATARAX_*` prefix |
| **Schema** | Validation | Type checking, constraints |
| **Registry** | Component lookup | Config-driven instantiation |

!!! note "Key points"

    - Configs are authored in TOML; loaders support include directives
    - Environment overrides are applied explicitly via `apply_environment_overrides`
    - Schemas catch errors at load time, not runtime
    - Registry enables dynamic component creation

## Quick Start

```python
from datarax.config import load_toml_from_path

# Load from a TOML file
config = load_toml_from_path("config.toml")

# Access nested values
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
```

## Modules

- [loaders](loaders.md) - TOML configuration file loaders (with includes)
- [environment](environment.md) - Environment variable overrides
- [schema](schema.md) - Configuration schema definitions and validation
- [registry](registry.md) - Component registry for config-driven instantiation

## Schema Validation

Define a schema by subclassing `ConfigSchema` with `SchemaField` class
attributes, then call the classmethod `validate`:

```python
from datarax.config import ConfigSchema
from datarax.config.schema import SchemaField


class TrainingSchema(ConfigSchema):
    batch_size: SchemaField = SchemaField(int, required=True)
    learning_rate: SchemaField = SchemaField(float, required=False, default=1e-3)


# Validates and applies defaults; raises ValidationError on error
config = TrainingSchema.validate(raw_config)
```

## Environment Variables

Environment overrides are opt-in — apply them explicitly to a loaded
config. Nested keys use a `__` separator under the `DATARAX_` prefix:

```python
from datarax.config import apply_environment_overrides

# DATARAX_TRAINING__BATCH_SIZE=64 -> config["training"]["batch_size"] = 64
config = apply_environment_overrides(config)
```

## See Also

- [Core Config](../core/config.md) - Module configuration classes
- [Installation](../getting_started/installation.md) - Environment setup
