"""Shared configuration primitives for source backends."""

import logging
from dataclasses import dataclass

from datarax.core.config import StructuralConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceConfigBase(StructuralConfig):
    """Common fields shared by eager and streaming source configs."""

    name: str | None = None
    split: str | None = None
    data_dir: str | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None
