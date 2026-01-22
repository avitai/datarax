"""Datarax data source components.

This module provides data source components for loading data.

Note: TFDSSource is lazily imported to avoid TensorFlow import hang on macOS ARM64.
Use explicit import if needed: `from datarax.sources.tfds_source import TFDSSource`
"""

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.array_record_source import ArrayRecordSourceModule

# Lazy imports for modules with heavy dependencies (TensorFlow, HuggingFace)
# This prevents import hangs on macOS ARM64 and speeds up import time
_lazy_imports = {
    "TFDSSource": "datarax.sources.tfds_source",
    "TfdsDataSourceConfig": "datarax.sources.tfds_source",
    "HFSource": "datarax.sources.hf_source",
    "HfDataSourceConfig": "datarax.sources.hf_source",
}


def __getattr__(name: str):
    """Lazy import for TFDSSource and HFSource to avoid heavy dependency loading."""
    if name in _lazy_imports:
        import importlib
        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy imports in dir() for discoverability."""
    return list(__all__)


__all__ = [
    "MemorySource",
    "MemorySourceConfig",
    "TFDSSource",
    "TfdsDataSourceConfig",
    "HFSource",
    "HfDataSourceConfig",
    "ArrayRecordSourceModule",
]
