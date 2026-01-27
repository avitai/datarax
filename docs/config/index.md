# Config

Configuration management and validation system. Load configurations from files, environment variables, and use schemas for validation.

## Components

| Component | Purpose | Formats |
|-----------|---------|---------|
| **Loaders** | Load from files | YAML, JSON, TOML |
| **Environment** | Env var handling | `DATARAX_*` prefix |
| **Schema** | Validation | Type checking, constraints |
| **Registry** | Component lookup | Config-driven instantiation |

`★ Insight ─────────────────────────────────────`

- Use YAML for human-readable configs
- Environment variables override file configs
- Schemas catch errors at load time, not runtime
- Registry enables dynamic component creation

`─────────────────────────────────────────────────`

## Quick Start

```python
from datarax.config import load_config, ConfigSchema

# Load from YAML file
config = load_config("config.yaml")

# Access nested values
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
```

## Modules

- [loaders](loaders.md) - Configuration file loaders (YAML, JSON, TOML)
- [environment](environment.md) - Environment variable handling
- [schema](schema.md) - Configuration schema definitions and validation
- [registry](registry.md) - Component registry for config-driven instantiation

## Schema Validation

```python
from datarax.config import ConfigSchema

schema = ConfigSchema({
    "batch_size": {"type": "int", "min": 1},
    "learning_rate": {"type": "float", "min": 0},
    "model": {"type": "str", "choices": ["small", "medium", "large"]},
})

# Validates and raises on error
config = schema.validate(raw_config)
```

## Environment Variables

```python
from datarax.config import get_env_config

# Reads DATARAX_* environment variables
env_config = get_env_config()
# DATARAX_BATCH_SIZE=64 -> {"batch_size": 64}
```

## See Also

- [Core Config](../core/config.md) - Module configuration classes
- [Installation](../getting_started/installation.md) - Environment setup
