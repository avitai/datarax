"""Datarax core components.

This module provides core modules and pipeline implementation for Datarax.
"""

from datarax.core.batcher import BatcherModule
from datarax.core.data_source import DataSourceModule
from datarax.core.module import DataraxModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule

from datarax.core.config import (
    DataraxModuleConfig,
    OperatorConfig,
    StructuralConfig,
    ElementOperatorConfig,
    BatchMixOperatorConfig,
)
from datarax.core.operator import OperatorModule
from datarax.core.structural import StructuralModule

# Import typing exports
from datarax.typing import Batch, Element

# Import utility functions from pytree_utils
from datarax.utils.pytree_utils import (
    add_batch_dimension,
    apply_to_batch_dimension,
    concatenate_batches,
    get_batch_size,
    get_pytree_structure_info,
    is_single_element,
    remove_batch_dimension,
    split_batch,
    validate_batch_consistency,
)


__all__ = [
    # ===== Type aliases =====
    "Batch",
    "Element",
    # ===== Base Modules =====
    "DataraxModule",
    # ===== Unified Architecture =====
    "DataraxModuleConfig",
    "OperatorConfig",
    "StructuralConfig",
    "ElementOperatorConfig",
    "BatchMixOperatorConfig",
    "OperatorModule",
    "StructuralModule",
    # ===== Data Source Modules =====
    "DataSourceModule",
    # ===== Sampler Modules =====
    "SamplerModule",
    # ===== Batcher Modules =====
    "BatcherModule",
    # ===== Sharder Modules =====
    "SharderModule",
    # ===== Utility Functions =====
    "is_single_element",
    "add_batch_dimension",
    "remove_batch_dimension",
    "get_batch_size",
    "split_batch",
    "concatenate_batches",
    "apply_to_batch_dimension",
    "validate_batch_consistency",
    "get_pytree_structure_info",
]
