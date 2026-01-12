"""Configuration dataclasses for Datarax modules.

This module provides typed configuration classes for all Datarax modules:
- DataraxModuleConfig: Base configuration for all modules
- OperatorConfig: Configuration for parametric operators
- StructuralConfig: Configuration for structural processors (runtime immutable)

All configs use dataclass with __post_init__ validation for fail-fast configuration errors.
"""

from dataclasses import dataclass
from typing import Any, Callable

from flax import nnx
from jaxtyping import PyTree


class FrozenInstanceError(Exception):
    """Raised when attempting to modify a frozen config instance."""

    pass


def validate_stochastic_config(stochastic: bool, stream_name: str | None) -> None:
    """Validate stochastic configuration rules.

    Args:
        stochastic: Whether the module uses randomness
        stream_name: RNG stream name (required if stochastic=True)

    Raises:
        ValueError: If validation rules are violated
    """
    # Rule 1: Stochastic requires stream_name
    if stochastic and stream_name is None:
        raise ValueError("Stochastic modules require stream_name for RNG management")

    # Rule 2: Deterministic forbids stream_name
    if not stochastic and stream_name is not None:
        raise ValueError(
            "Deterministic modules should not specify stream_name. "
            "Remove stream_name or set stochastic=True"
        )


@dataclass
class DataraxModuleConfig:
    """Base configuration for all Datarax modules.

    All module configs are dataclasses with __post_init__ validation.
    Configuration is validated at construction, before being passed to module.

    Child classes inherit from this and add their specific configuration.

    Attributes:
        cacheable: Whether to enable caching for this module
        batch_stats_fn: Function or module to compute batch statistics dynamically
        precomputed_stats: Static precomputed statistics

    Validation Rules:
        - batch_stats_fn and precomputed_stats are mutually exclusive
    """

    # Common configuration (inherited by all modules)
    cacheable: bool = False
    batch_stats_fn: Callable | nnx.Module | None = None
    precomputed_stats: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate configuration after initialization.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Mutual exclusivity validation
        if self.batch_stats_fn is not None and self.precomputed_stats is not None:
            raise ValueError(
                "Cannot specify both batch_stats_fn and precomputed_stats. "
                "Choose dynamic computation or static values, not both."
            )


@dataclass
class OperatorConfig(DataraxModuleConfig):
    """Configuration for OperatorModule (mutable, learnable).

    Inherits from DataraxModuleConfig:
    - cacheable: bool
    - batch_stats_fn: Callable | nnx.Module | None
    - precomputed_stats: dict[str, Any] | None

    Adds operator-specific configuration:
    - stochastic: bool
    - stream_name: str | None

    Validation Rules:
        - Inherits mutual exclusivity of statistics from parent
        - Stochastic operators require stream_name for RNG management
        - Deterministic operators should not specify stream_name

    Attributes:
        stochastic: Whether this operator uses randomness
        stream_name: RNG stream name (required if stochastic=True)
    """

    # Operator-specific configuration
    stochastic: bool = False
    stream_name: str | None = None

    def __post_init__(self):
        """Validate configuration after initialization.

        Validates both base config (via super) and operator-specific rules.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Call parent validation (mutual exclusivity of statistics)
        super().__post_init__()

        # Validate stochastic configuration rules
        validate_stochastic_config(self.stochastic, self.stream_name)


@dataclass
class MapOperatorConfig(OperatorConfig):
    """Configuration for MapOperator - unified deterministic/stochastic operator.

    Inherits from OperatorConfig:
    - cacheable: bool
    - batch_stats_fn: Callable | nnx.Module | None
    - precomputed_stats: dict[str, Any] | None
    - stochastic: bool (currently must be False - stochastic mode not yet implemented)
    - stream_name: str | None

    Adds MapOperator-specific configuration:
    - subtree: PyTree | None - Nested dict matching element.data structure
      If None, user fn is applied to full element (full-tree mode)
      If specified, only the specified subtree is affected (subtree mode)

    Validation Rules:
        - Inherits all validation from OperatorConfig
        - Currently enforces stochastic=False (NotImplementedError if True)

    Attributes:
        subtree: Optional PyTree mask specifying which parts of data to transform.
                 Structure must match element.data. Use None as leaf to indicate
                 field should be transformed. Example: {"image": None, "mask": None}

    Examples:
        Full-tree mode:

        ```python
        from datarax.core.config import MapOperatorConfig

        config = MapOperatorConfig(subtree=None, stochastic=False)
        ```

        Subtree mode (single field):

        ```python
        config = MapOperatorConfig(subtree={"image": None}, stochastic=False)
        ```

        Subtree mode (multiple fields):

        ```python
        config = MapOperatorConfig(
            subtree={"image": None, "mask": None},
            stochastic=False
        )
        ```
    """

    subtree: PyTree | None = None

    # Note: No additional validation in __post_init__ needed
    # Parent OperatorConfig handles stochastic validation
    # MapOperator.__init__ enforces stochastic=False


@dataclass
class ElementOperatorConfig(OperatorConfig):
    """Configuration for ElementOperator - element-level transformation operator.

    Inherits from OperatorConfig:
    - cacheable: bool
    - batch_stats_fn: Callable | nnx.Module | None
    - precomputed_stats: dict[str, Any] | None
    - stochastic: bool
    - stream_name: str | None

    ElementOperator applies user-provided functions to entire Element structures
    (data + state + metadata), enabling coordinated transformations across
    multiple fields and access to element state.

    Validation Rules:
        - Inherits all validation from OperatorConfig

    Examples:
        Deterministic element transformation:

        ```python
        from datarax.core.config import ElementOperatorConfig

        config = ElementOperatorConfig(stochastic=False)
        ```

        Stochastic element augmentation:

        ```python
        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        ```
    """

    # No additional fields - inherits all from OperatorConfig
    # User function is passed to ElementOperator, not stored in config
    pass


@dataclass
class BatchMixOperatorConfig(OperatorConfig):
    """Configuration for BatchMixOperator - unified MixUp and CutMix batch augmentation.

    Inherits from OperatorConfig:
    - cacheable: bool
    - batch_stats_fn: Callable | nnx.Module | None
    - precomputed_stats: dict[str, Any] | None
    - stochastic: bool (always True for BatchMixOperator)
    - stream_name: str | None

    BatchMixOperator performs batch-level sample mixing that cannot be
    decomposed into element-level operations. It mixes samples across
    the batch, either through linear interpolation (MixUp) or patch
    cutting/pasting (CutMix).

    Attributes:
        mode: Mixing mode - "mixup" or "cutmix"
        alpha: Beta distribution parameter for mixing ratio (default: 1.0)
        data_field: Field name containing data to mix (default: "image" for cutmix)
        label_field: Field name containing labels to mix (default: "label")

    Validation Rules:
        - mode must be "mixup" or "cutmix"
        - alpha must be positive
        - Always stochastic (forced to True)

    Examples:
        MixUp augmentation:

        ```python
        from datarax.core.config import BatchMixOperatorConfig

        config = BatchMixOperatorConfig(mode="mixup", alpha=0.4)
        ```

        CutMix augmentation:

        ```python
        config = BatchMixOperatorConfig(mode="cutmix", alpha=1.0)
        ```
    """

    mode: str = "mixup"
    alpha: float = 1.0
    data_field: str = "image"
    label_field: str = "label"
    # Override defaults: BatchMixOperator is always stochastic
    stochastic: bool = True
    stream_name: str | None = "batch_mix"

    def __post_init__(self):
        """Validate configuration after initialization.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Validate mode
        if self.mode not in ("mixup", "cutmix"):
            raise ValueError(f"mode must be 'mixup' or 'cutmix', got '{self.mode}'")

        # Validate alpha
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")

        # Force stochastic (batch mixing always requires randomness)
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "batch_mix")

        # Call parent validation
        super().__post_init__()


@dataclass
class StructuralConfig(DataraxModuleConfig):
    """Configuration for StructuralModule (runtime immutable, compile-time constants).

    Inherits from DataraxModuleConfig:
    - cacheable: bool
    - batch_stats_fn: Callable | nnx.Module | None
    - precomputed_stats: dict[str, Any] | None

    Adds structural-specific configuration:
    - stochastic: bool
    - stream_name: str | None

    Note: This config enforces runtime immutability through __setattr__ override.
    After __post_init__ completes, the instance is frozen and cannot be modified.
    All configuration must be known at module construction time for JIT compilation.

    Validation Rules:
        - Inherits mutual exclusivity of statistics from parent
        - Stochastic structural modules require stream_name for RNG management
        - Deterministic structural modules should not specify stream_name

    Attributes:
        stochastic: Whether this structural module uses randomness (e.g., sampling)
        stream_name: RNG stream name (required if stochastic=True)
    """

    # Structural-specific configuration
    stochastic: bool = False
    stream_name: str | None = None

    def __post_init__(self):
        """Validate configuration and freeze instance.

        Validates both base config (via super) and structural-specific rules,
        then marks the instance as frozen to prevent further modifications.

        Raises:
            ValueError: If configuration is invalid.
        """
        # Call parent validation (mutual exclusivity of statistics)
        super().__post_init__()

        # Validate stochastic configuration rules
        validate_stochastic_config(self.stochastic, self.stream_name)

        # Mark as frozen after validation (must use object.__setattr__)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to enforce immutability after initialization.

        Args:
            name: Attribute name
            value: Attribute value

        Raises:
            FrozenInstanceError: If attempting to modify frozen instance
        """
        # Check if instance is frozen (after __post_init__ completes)
        if hasattr(self, "_frozen") and object.__getattribute__(self, "_frozen"):
            raise FrozenInstanceError(
                f"Cannot modify frozen StructuralConfig field '{name}'. "
                f"StructuralConfig instances are immutable after construction."
            )
        # Allow normal attribute setting during initialization
        object.__setattr__(self, name, value)
