"""A pipeline places each process's share of its batches; meshes and other arrays are substrax's.

``Pipeline(..., sharding=...)`` reads and places the rows of every global batch a process's devices
hold (``datarax.pipeline.placement``). Any other batch is placed by
``substrax.spmd.place_batch_on_shards``, logical-to-mesh axis rules are ``substrax.mesh.MeshRules``
with ``partition_spec_for_names``, a per-shard transform is ``nnx.shard_map`` and a partitioned
parameter ``nnx.with_partitioning``. No other datarax module splits data between processes, so
the pipeline's placement is the one per-process share.
"""

from __future__ import annotations

import pkgutil

import pytest

import datarax
import datarax.core
from datarax.config import registry


pytestmark = pytest.mark.contract


def _module_names() -> list[str]:
    return [module.name for module in pkgutil.walk_packages(datarax.__path__, "datarax.")]


def test_the_pipeline_placement_is_the_one_sharding_module() -> None:
    names = _module_names()

    assert "datarax.pipeline.placement" in names  # the walk reaches the pipeline's modules
    assert [name for name in names if "shard" in name.lower()] == []


def test_core_defines_no_sharder_component() -> None:
    core_modules = {module.name for module in pkgutil.iter_modules(datarax.core.__path__)}

    assert "sharder" not in core_modules
    assert "SharderModule" not in datarax.__all__
    assert "SharderModule" not in datarax.core.__all__
    assert "sharder" not in registry._COMPONENT_TYPES
