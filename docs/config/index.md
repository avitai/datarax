# Config

Configuration in Datarax is **parameters, composed**: small TOML files that include and
override one another, environment overrides for values they already have, and a schema you
define to check the result. The pipeline itself is built in Python from the validated values,
so every source, operator and stage stays a typed, inspectable object.

## Components

| Component | Purpose |
|-----------|---------|
| **Loaders** | Read TOML; a file's `include` list merges other files under it |
| **Environment** | `DATARAX_CONFIG__*` variables replace values the configuration already has |
| **Schema** | A `ConfigSchema` subclass checks types, required fields and defaults |
| **Registry** | Look components up by name (`register_component`) |

!!! note "Key points"

    - Split a configuration by concern (data, augmentation, training) and compose the pieces
      with `include`; a file's own values take precedence over what it includes
    - An environment override is read as the type of the value it replaces; a name the
      configuration does not have is refused, not added
    - A schema's fields include its base classes', and a field can hold a nested schema, so
      schemas compose the way the files do
    - Build the `Pipeline` in Python from the validated values

## Composing configuration files

```toml
# data.toml
[data]
batch_size = 128
image_size = [32, 32]
```

```toml
# augment.toml
[augment]
crop_padding = 4
flip = true
```

```toml
# experiment.toml
include = ["data.toml", "augment.toml"]

[data]
batch_size = 256   # overrides data.toml
```

```python
from datarax.config import apply_environment_overrides, load_config_from_path_with_includes

config = load_config_from_path_with_includes("experiment.toml")
# {'data': {'batch_size': 256, 'image_size': [32, 32]},
#  'augment': {'crop_padding': 4, 'flip': True}}

# DATARAX_CONFIG__DATA__BATCH_SIZE=64 -> config["data"]["batch_size"] == 64 (an int)
config = apply_environment_overrides(config)
```

`apply_environment_overrides` returns a copy and reads the process environment unless you pass
`environ=` a mapping of the variables you mean.

## Schema validation

```python
from datarax.config import ConfigSchema, SchemaField


class Data(ConfigSchema):
    batch_size: SchemaField = SchemaField(int)
    image_size: SchemaField = SchemaField(list)


class Augment(ConfigSchema):
    crop_padding: SchemaField = SchemaField(int, required=False, default=0)
    flip: SchemaField = SchemaField(bool, required=False, default=False)


class Experiment(ConfigSchema):
    data: SchemaField = SchemaField(Data)
    augment: SchemaField = SchemaField(Augment, required=False, default={})


validated = Experiment.validate(config)  # raises ValidationError naming the field
```

A nested failure names its path (`Field 'data': Required field 'batch_size' is missing`), an
absent optional nested schema gets its own defaults, an integer where a float is expected
becomes a float, and a boolean is never an integer.

## Modules

- [loaders](loaders.md) - TOML loading, includes and merging
- [environment](environment.md) - Environment variable overrides
- [schema](schema.md) - Configuration schemas and validation
- [registry](registry.md) - Component registry

## See Also

- [Core Config](../core/config.md) - Module configuration classes
- [Installation](../getting_started/installation.md) - Environment setup
