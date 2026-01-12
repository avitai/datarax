"""Component registry for Datarax.

This module provides a registry system for Datarax components, allowing
components to be registered and instantiated dynamically from configuration.
Enhanced with Flax NNX module support for proper RNG and Variable handling.
"""

import inspect
from typing import Any, Callable, Type

import flax.nnx as nnx

from datarax.core.batcher import BatcherModule
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule
from datarax.utils.prng import create_rngs


# Type alias for component constructors
ComponentConstructor = Callable[..., Any]

# Registry of component constructors by type name
# This will be populated by the @register_component decorator
_COMPONENT_REGISTRY: dict[str, ComponentConstructor] = {}

# Registry of component types
_COMPONENT_TYPES: dict[str, type] = {
    "source": DataSourceModule,
    "operator": OperatorModule,
    "sampler": SamplerModule,
    "batcher": BatcherModule,
    "sharder": SharderModule,
}


def _is_nnx_module(cls: Type) -> bool:
    """Check if a class is an NNX module.

    Args:
        cls: Class to check.

    Returns:
        True if the class inherits from nnx.Module.
    """
    return inspect.isclass(cls) and issubclass(cls, nnx.Module)


def _get_constructor_signature(constructor: ComponentConstructor) -> inspect.Signature:
    """Get the signature of a constructor.

    Args:
        constructor: The constructor function or class.

    Returns:
        The signature of the constructor.
    """
    if inspect.isclass(constructor):
        return inspect.signature(constructor.__init__)
    else:
        return inspect.signature(constructor)


def _prepare_rngs_for_nnx(config: dict[str, Any]) -> nnx.Rngs:
    """Prepare RNG configuration for NNX modules.

    Args:
        config: Configuration dictionary potentially containing RNG specs.

    Returns:
        An nnx.Rngs object suitable for NNX modules.
    """
    rngs_config = config.get("rngs", {})

    if isinstance(rngs_config, dict):
        if rngs_config:
            # Create Rngs from seed specifications
            import jax

            rng_dict = {}
            for stream_name, seed_value in rngs_config.items():
                if isinstance(seed_value, int):
                    rng_dict[stream_name] = jax.random.key(seed_value)
                else:
                    rng_dict[stream_name] = seed_value
            return nnx.Rngs(rng_dict)
        else:
            # Use default RNGs if empty dict
            seed = config.get("seed", None)
            return create_rngs(seed=seed)
    elif isinstance(rngs_config, nnx.Rngs):
        return rngs_config
    else:
        # Fallback to default
        seed = config.get("seed", None)
        return create_rngs(seed=seed)


def _prepare_config_for_nnx(
    constructor: ComponentConstructor, config: dict[str, Any]
) -> dict[str, Any]:
    """Prepare configuration for NNX module instantiation.

    Args:
        constructor: The NNX module constructor.
        config: Original configuration dictionary.

    Returns:
        Modified configuration suitable for NNX module instantiation.
    """
    # Create a copy to avoid modifying the original
    prepared_config = dict(config)

    # Get constructor signature
    sig = _get_constructor_signature(constructor)

    # Check if constructor expects rngs parameter
    has_rngs_param = False
    for param_name, param in sig.parameters.items():
        if param_name == "rngs" or (
            param.annotation == nnx.Rngs
            or param.annotation == (nnx.Rngs | None)
            or str(param.annotation).endswith("nnx.Rngs]")
        ):
            has_rngs_param = True
            break

    # Prepare rngs if needed
    if has_rngs_param:
        prepared_config["rngs"] = _prepare_rngs_for_nnx(config)

    # Remove NNX-specific configuration keys that shouldn't be passed to constructor
    nnx_keys_to_remove = {"variables", "load_state_from", "rngs"}
    for key in nnx_keys_to_remove:
        if key in prepared_config and key not in sig.parameters:
            prepared_config.pop(key)

    return prepared_config


def _initialize_nnx_variables(instance: nnx.Module, config: dict[str, Any]) -> None:
    """Initialize NNX variables from configuration.

    Args:
        instance: The NNX module instance.
        config: Configuration containing variable specifications.
    """
    variables_config = config.get("variables", {})
    if not variables_config:
        return

    # Apply variable values to the instance
    for var_path, var_value in variables_config.items():
        # Navigate through nested attributes using dot notation
        obj = instance
        path_parts = var_path.split(".")

        for part in path_parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise ValueError(f"Variable path '{var_path}' not found in module")

        final_attr = path_parts[-1]
        if hasattr(obj, final_attr):
            attr = getattr(obj, final_attr)
            if isinstance(attr, nnx.Variable):
                attr.set_value(var_value)
            else:
                setattr(obj, final_attr, var_value)
        else:
            raise ValueError(f"Variable '{final_attr}' not found in module")


def _load_state_from_file(instance: nnx.Module, config: dict[str, Any]) -> None:
    """Load state from file if specified in configuration.

    Args:
        instance: The NNX module instance.
        config: Configuration potentially containing load_state_from path.
    """
    load_path = config.get("load_state_from")
    if load_path:
        # Implementation would depend on the checkpoint format
        # This is a placeholder for now
        raise NotImplementedError("State loading from file not yet implemented")


def register_component(
    component_type: str, name: str | None = None
) -> Callable[[Type[Any] | Callable[..., Any]], Type[Any] | Callable[..., Any]]:
    """Register a component constructor with the registry.

    This decorator can be used to register a component constructor with
    the component registry, allowing it to be instantiated from configuration.
    Enhanced to handle both regular classes and NNX modules.

    Args:
        component_type: The type of component being registered (e.g., "source",
            "transformer", "augmenter").
        name: Optional custom name for the component. If not provided, the
            class name will be used.

    Returns:
        A decorator function that registers the component constructor.

    Examples:
        ```python
        @register_component("source")
        class MyDataSource(DataSourceModule):
            def __init__(self, path: str, *, rngs: nnx.Rngs | None = None):
                super().__init__(rngs=rngs)
                self.path = path
        ```
    """

    def decorator(cls_or_fn: Type[Any] | Callable[..., Any]):
        # Use provided name or class/function name
        component_name = name or cls_or_fn.__name__

        # Register the component constructor
        key = f"{component_type}.{component_name}"
        _COMPONENT_REGISTRY[key] = cls_or_fn

        # Return the original class or function
        return cls_or_fn

    return decorator


def get_component_constructor(component_type: str, name: str) -> ComponentConstructor:
    """Get a component constructor by type and name.

    Args:
        component_type: The type of component (e.g., "source", "transformer").
        name: The name of the component.

    Returns:
        The component constructor function or class.

    Raises:
        KeyError: If the component is not registered.
    """
    component_key = f"{component_type}.{name}"
    if component_key not in _COMPONENT_REGISTRY:
        raise KeyError(f"Component not registered: {component_key}")

    return _COMPONENT_REGISTRY[component_key]


def is_component_registered(component_type: str, name: str) -> bool:
    """Check if a component is registered.

    Args:
        component_type: The type of component (e.g., "source", "transformer").
        name: The name of the component.

    Returns:
        True if the component is registered, False otherwise.
    """
    component_key = f"{component_type}.{name}"
    return component_key in _COMPONENT_REGISTRY


def list_registered_components(
    component_type: str | None = None,
) -> dict[str, ComponentConstructor]:
    """List all registered components of a given type.

    Args:
        component_type: Optional type of components to list. If None, list all
            registered components.

    Returns:
        Dictionary mapping component names to constructors.
    """
    if component_type is None:
        return dict(_COMPONENT_REGISTRY)

    prefix = f"{component_type}."
    return {
        key.replace(prefix, ""): constructor
        for key, constructor in _COMPONENT_REGISTRY.items()
        if key.startswith(prefix)
    }


def create_component_from_config(component_type: str, name: str, config: dict[str, Any]) -> Any:
    """Create a component instance from configuration.

    Enhanced to properly handle NNX modules with RNG and Variable initialization.

    Args:
        component_type: The type of component (e.g., "source", "transformer").
        name: The name of the component.
        config: Configuration dictionary for the component.

    Returns:
        The instantiated component.

    Raises:
        KeyError: If the component is not registered.
        TypeError: If the configuration is invalid.
    """
    constructor = get_component_constructor(component_type, name)

    try:
        # Check if this is an NNX module
        if _is_nnx_module(constructor):
            # Prepare configuration for NNX module
            prepared_config = _prepare_config_for_nnx(constructor, config)

            # Create the instance
            instance = constructor(**prepared_config)

            # Initialize variables if specified
            _initialize_nnx_variables(instance, config)

            # Load state from file if specified
            _load_state_from_file(instance, config)

            return instance
        else:
            # Regular component creation
            return constructor(**config)

    except Exception as e:
        msg = f"Failed to create component {component_type}.{name}: {e}"
        raise TypeError(msg) from e


def get_component_info(component_type: str, name: str) -> dict[str, Any]:
    """Get detailed information about a registered component.

    Args:
        component_type: The type of component.
        name: The name of the component.

    Returns:
        Dictionary containing component information including whether it's an NNX module,
        required parameters, and RNG requirements.
    """
    constructor = get_component_constructor(component_type, name)

    is_nnx = _is_nnx_module(constructor)
    sig = _get_constructor_signature(constructor)

    # Extract parameter information
    params_info = {}
    requires_rngs = False

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_info = {
            "annotation": str(param.annotation)
            if param.annotation != inspect.Parameter.empty
            else None,
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "kind": str(param.kind),
        }

        if param_name == "rngs" or (
            param.annotation == nnx.Rngs
            or param.annotation == (nnx.Rngs | None)
            or str(param.annotation).endswith("nnx.Rngs]")
        ):
            requires_rngs = True

        params_info[param_name] = param_info

    return {
        "constructor": constructor,
        "is_nnx_module": is_nnx,
        "requires_rngs": requires_rngs,
        "parameters": params_info,
        "docstring": constructor.__doc__,
    }
