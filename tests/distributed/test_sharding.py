"""Tests for NNX SPMD sharding utilities.

Tests MeshRules, factory functions, and sharding helpers that follow
upstream Flax NNX FSDP patterns.
"""

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from datarax.distributed.sharding import (
    MeshRules,
    apply_sharding_rules,
    create_named_sharding,
    data_parallel_rules,
    fsdp_rules,
)


class TestMeshRules:
    """Tests for the MeshRules dataclass."""

    def test_creation_with_defaults(self):
        """Test default MeshRules has all None axes."""
        rules = MeshRules()
        assert rules.data is None
        assert rules.embed is None
        assert rules.mlp is None
        assert rules.heads is None

    def test_creation_with_values(self):
        """Test MeshRules with explicit axis assignments."""
        rules = MeshRules(data="dp", embed="mp")
        assert rules.data == "dp"
        assert rules.embed == "mp"
        assert rules.mlp is None

    def test_call_returns_axis_tuple(self):
        """Test that calling MeshRules returns axis mappings for given keys."""
        rules = MeshRules(data="dp", embed="mp", mlp="mp")
        result = rules("data", "embed", "mlp")
        assert result == ("dp", "mp", "mp")

    def test_call_with_unmapped_keys(self):
        """Test that unmapped keys return None."""
        rules = MeshRules(data="dp")
        result = rules("data", "heads")
        assert result == ("dp", None)

    def test_call_single_key(self):
        """Test calling with a single key."""
        rules = MeshRules(data="dp")
        result = rules("data")
        assert result == ("dp",)

    def test_immutability(self):
        """Test that MeshRules is frozen."""
        rules = MeshRules(data="dp")
        with pytest.raises(AttributeError):
            rules.data = "mp"  # type: ignore[misc]


class TestFactoryFunctions:
    """Tests for MeshRules factory functions."""

    def test_data_parallel_rules(self):
        """Test data_parallel_rules maps data axis only."""
        rules = data_parallel_rules(data_axis="data")
        assert rules.data == "data"
        assert rules.embed is None
        assert rules.mlp is None
        assert rules.heads is None

    def test_data_parallel_rules_custom_axis(self):
        """Test data_parallel_rules with custom axis name."""
        rules = data_parallel_rules(data_axis="batch")
        assert rules.data == "batch"

    def test_fsdp_rules(self):
        """Test fsdp_rules maps data and model axes."""
        rules = fsdp_rules(data_axis="data", model_axis="model")
        assert rules.data == "data"
        assert rules.embed == "model"
        assert rules.mlp == "model"
        assert rules.heads == "model"

    def test_fsdp_rules_custom_axes(self):
        """Test fsdp_rules with custom axis names."""
        rules = fsdp_rules(data_axis="dp", model_axis="mp")
        assert rules.data == "dp"
        assert rules.embed == "mp"


class TestCreateNamedSharding:
    """Tests for create_named_sharding function."""

    def test_single_axis(self):
        """Test creating NamedSharding with a single axis."""
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
        sharding = create_named_sharding(mesh, "data")
        assert isinstance(sharding, NamedSharding)
        assert sharding.spec == PartitionSpec("data")

    def test_no_axes_replicated(self):
        """Test that no axis names produces a replicated sharding."""
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
        sharding = create_named_sharding(mesh)
        assert sharding.spec == PartitionSpec()

    def test_multiple_axes(self):
        """Test creating NamedSharding with multiple axes."""
        mesh = Mesh(
            np.array(jax.devices()[:1]).reshape(1, 1),
            axis_names=("data", "model"),
        )
        sharding = create_named_sharding(mesh, "data", "model")
        assert sharding.spec == PartitionSpec("data", "model")

    def test_none_axes_mixed(self):
        """Test creating NamedSharding with None for replicated dimensions."""
        mesh = Mesh(np.array(jax.devices()[:1]), axis_names=("data",))
        sharding = create_named_sharding(mesh, "data", None)
        assert sharding.spec == PartitionSpec("data", None)


class TestApplyShardingRules:
    """Tests for apply_sharding_rules function."""

    def test_basic_application(self):
        """Test applying MeshRules to create a PartitionSpec."""
        rules = MeshRules(data="dp", embed="mp")
        spec = apply_sharding_rules(rules, "data", "embed")
        assert spec == PartitionSpec("dp", "mp")

    def test_unmapped_dimensions(self):
        """Test that unmapped dimensions produce None in PartitionSpec."""
        rules = MeshRules(data="dp")
        spec = apply_sharding_rules(rules, "data", "heads")
        assert spec == PartitionSpec("dp", None)

    def test_empty_rules(self):
        """Test applying empty rules produces all-None PartitionSpec."""
        rules = MeshRules()
        spec = apply_sharding_rules(rules, "data", "embed")
        assert spec == PartitionSpec(None, None)
