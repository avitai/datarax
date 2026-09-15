"""Datarax provides process-level sharding; device placement and axis rules come from substrax.

Batch placement is ``substrax.spmd.place_batch_on_shards``, logical-to-mesh axis rules are
``substrax.mesh.MeshRules`` with ``partition_spec_for_names``, a per-shard transform is
``nnx.shard_map`` and a partitioned parameter is ``nnx.with_partitioning``. Datarax's own module
is the process sharder, which slices data for the current JAX process.
"""

from __future__ import annotations

import pkgutil

import pytest

import datarax
import datarax.core
import datarax.sharding
from datarax.config import registry
from datarax.core.module import DataraxModule
from datarax.sharding import JaxProcessSharderModule


pytestmark = pytest.mark.contract


def test_sharding_package_exports_only_the_process_sharder() -> None:
    assert datarax.sharding.__all__ == ["JaxProcessSharderConfig", "JaxProcessSharderModule"]


def test_sharding_package_holds_only_the_process_sharder_module() -> None:
    modules = sorted(module.name for module in pkgutil.iter_modules(datarax.sharding.__path__))

    assert modules == ["jax_process_sharder"]


def test_core_defines_no_sharder_component() -> None:
    core_modules = {module.name for module in pkgutil.iter_modules(datarax.core.__path__)}

    assert "sharder" not in core_modules
    assert "SharderModule" not in datarax.__all__
    assert "SharderModule" not in datarax.core.__all__
    assert "sharder" not in registry._COMPONENT_TYPES


def test_process_sharder_is_a_datarax_module() -> None:
    sharder = JaxProcessSharderModule()

    assert isinstance(sharder, DataraxModule)
    assert sharder.shard_data(list(range(8))) == list(range(8))
