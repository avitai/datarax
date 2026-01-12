"""Operator implementations for Datarax.

This module provides concrete operator implementations:
- MapOperator: Unified operator for full-tree and subtree transformations

MapOperator supports two modes:
1. Full-tree mode (subtree=None): Applies user function to entire element data
2. Subtree mode (subtree specified): Applies user function to data, but only
   affects specified subtree using JAX tree masking

These operators implement the unified OperatorModule API.

Feature Support Status:
- MapOperator: Current implementation (deterministic mode only)
- Stochastic mode: To be added in future update after use case analysis
"""

from datarax.operators.map_operator import MapOperator
from datarax.operators.composite_operator import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import (
    SelectorOperator,
    SelectorOperatorConfig,
)
from datarax.operators.element_operator import ElementOperator
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.core.config import (
    BatchMixOperatorConfig,
    ElementOperatorConfig,
    MapOperatorConfig,
)

__all__ = [
    # Core operators
    "MapOperator",
    "MapOperatorConfig",
    "ElementOperator",
    "ElementOperatorConfig",
    # Composition
    "CompositeOperatorModule",
    "CompositeOperatorConfig",
    "CompositionStrategy",
    # Wrappers
    "ProbabilisticOperator",
    "ProbabilisticOperatorConfig",
    "SelectorOperator",
    "SelectorOperatorConfig",
    # Batch-level
    "BatchMixOperator",
    "BatchMixOperatorConfig",
]
