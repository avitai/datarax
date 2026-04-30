"""Datarax core components.

This module provides core modules and pipeline implementation for Datarax.
"""

from datarax.core.batcher import BatcherModule
from datarax.core.config import (
    BatchMixOperatorConfig,
    DataraxModuleConfig,
    ElementOperatorConfig,
    OperatorConfig,
    StructuralConfig,
)
from datarax.core.data_source import DataSourceModule
from datarax.core.module import DataraxModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule
from datarax.core.structural import StructuralModule
from datarax.core.temporal import TimeSeriesSpec

# Import typing exports
from datarax.typing import Batch, Element


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
    # ===== Time-series contracts =====
    "TimeSeriesSpec",
]
