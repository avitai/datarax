"""Operator implementations for Datarax.

This module provides concrete operator implementations:

- MapOperator: Unified operator for full-tree and subtree transformations
- ElementOperator: Element-level transformation wrapper
- CompositeOperatorModule: Compose multiple operators with 11 strategies
- ProbabilisticOperator: Apply operators with configurable probability
- SelectorOperator: Route inputs to one of N operators
- BatchMixOperator: Batch-level mixing (e.g., CutMix, MixUp)

CompositeOperatorModule's WEIGHTED_PARALLEL replaces the fields named in ``mix_fields``
with a weighted sum of its operators' outputs, under three weight modes: static weights
(a linear combination), learnable weights (``softmax`` over ``nnx.Param`` logits), and
dynamic external weights via ``weight_key`` for differentiable pipelines (e.g.,
Gumbel-Softmax policies).
"""

from datarax.core.config import (
    BatchMixOperatorConfig,
    ElementOperatorConfig,
    MapOperatorConfig,
)
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.element_operator import ElementOperator
from datarax.operators.map_operator import MapOperator
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import (
    SelectorOperator,
    SelectorOperatorConfig,
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
