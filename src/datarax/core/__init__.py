"""Datarax core components.

This module provides core modules and pipeline implementation for Datarax.
"""

from datarax.core.batcher import BatcherModule
from datarax.core.config import (
    BatchMixOperatorConfig,
    DataraxModuleConfig,
    ElementOperatorConfig,
    OperatorConfig,
    SamplerConfig,
    StructuralConfig,
)
from datarax.core.data_source import DataSourceModule

# The record and batch types
from datarax.core.element_batch import Batch, Element
from datarax.core.module import DataraxModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.structural import StructuralModule
from datarax.core.temporal import TimeSeriesSpec


__all__ = [
    # ===== Record and batch types =====
    "Batch",
    "Element",
    # ===== Base Modules =====
    "DataraxModule",
    # ===== Unified Architecture =====
    "DataraxModuleConfig",
    "OperatorConfig",
    "SamplerConfig",
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
    # ===== Time-series contracts =====
    "TimeSeriesSpec",
]
