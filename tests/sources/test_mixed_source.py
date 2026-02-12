"""Tests for MixDataSourcesNode — weighted mixing of multiple data sources.

TDD RED phase: These tests define the expected behavior of MixDataSourcesConfig
and MixDataSourcesNode before implementation.
"""

import pytest
import flax.nnx as nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def source_a():
    """Create a MemorySource with 10 elements labeled 'a'."""
    data = [{"value": i, "label": "a"} for i in range(10)]
    config = MemorySourceConfig()
    return MemorySource(config, data)


@pytest.fixture
def source_b():
    """Create a MemorySource with 10 elements labeled 'b'."""
    data = [{"value": i + 100, "label": "b"} for i in range(10)]
    config = MemorySourceConfig()
    return MemorySource(config, data)


@pytest.fixture
def source_small():
    """Create a small MemorySource with 3 elements labeled 'small'."""
    data = [{"value": i + 200, "label": "small"} for i in range(3)]
    config = MemorySourceConfig()
    return MemorySource(config, data)


@pytest.fixture
def large_source_a():
    """Create a large MemorySource with 800 elements labeled 'a'.

    Sized proportional to 0.8 weight so it doesn't exhaust prematurely
    when paired with large_source_b (200) and weights [0.8, 0.2].
    """
    data = [{"value": i, "label": "a"} for i in range(800)]
    config = MemorySourceConfig()
    return MemorySource(config, data)


@pytest.fixture
def large_source_b():
    """Create a large MemorySource with 200 elements labeled 'b'.

    Sized proportional to 0.2 weight for weighted sampling tests.
    """
    data = [{"value": i + 1000, "label": "b"} for i in range(200)]
    config = MemorySourceConfig()
    return MemorySource(config, data)


# ============================================================================
# TestMixDataSourcesConfig
# ============================================================================


class TestMixDataSourcesConfig:
    """Test MixDataSourcesConfig validation and normalization."""

    def test_valid_config(self):
        """Valid config with 2 sources and equal weights."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        assert config.num_sources == 2
        assert config.weights == pytest.approx((0.5, 0.5))

    def test_weights_must_match_sources_count(self):
        """Raises ValueError if len(weights) != num_sources."""
        with pytest.raises(ValueError, match="weights"):
            MixDataSourcesConfig(num_sources=3, weights=(0.5, 0.5))

    def test_weights_must_be_positive(self):
        """Raises ValueError for negative weights."""
        with pytest.raises(ValueError, match="positive"):
            MixDataSourcesConfig(num_sources=2, weights=(-0.5, 1.5))

    def test_weights_are_normalized(self):
        """Unnormalized weights [1.0, 3.0] are stored as [0.25, 0.75]."""
        config = MixDataSourcesConfig(num_sources=2, weights=(1.0, 3.0))
        assert config.weights == pytest.approx((0.25, 0.75))

    def test_empty_sources_raises(self):
        """Raises ValueError for num_sources=0."""
        with pytest.raises(ValueError):
            MixDataSourcesConfig(num_sources=0, weights=())

    def test_config_is_stochastic(self):
        """Mixing requires RNG, so config.stochastic must be True."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        assert config.stochastic is True
        assert config.stream_name == "mix"


# ============================================================================
# TestMixDataSourcesIteration
# ============================================================================


class TestMixDataSourcesIteration:
    """Test iteration behavior of MixDataSourcesNode."""

    def test_len_is_sum_of_sources(self, source_a, source_b):
        """len(mixed) == sum(len(s) for s in sources)."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))
        assert len(mixed) == len(source_a) + len(source_b)  # 10 + 10 = 20

    def test_iteration_yields_correct_count(self, source_a, source_b):
        """Iterating produces exactly len(mixed) elements."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))
        elements = list(mixed)
        assert len(elements) == 20

    def test_elements_come_from_sources(self, source_a, source_b):
        """Each yielded element must match one of the source elements."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))

        # Collect all possible elements from both sources
        valid_labels = {"a", "b"}

        for elem in mixed:
            assert elem["label"] in valid_labels

    def test_weighted_sampling_distribution(self, large_source_a, large_source_b):
        """With weights [0.8, 0.2], ~80% of elements should come from source_a.

        Sources are sized proportional to weights (800 + 200 = 1000 total)
        so neither source exhausts prematurely, allowing the RNG distribution
        to be accurately measured.
        """
        config = MixDataSourcesConfig(num_sources=2, weights=(0.8, 0.2))
        mixed = MixDataSourcesNode(config, [large_source_a, large_source_b], rngs=nnx.Rngs(42))
        elements = list(mixed)

        count_a = sum(1 for e in elements if e["label"] == "a")
        fraction_a = count_a / len(elements)

        # With 1000 samples, weights [0.8, 0.2], and proportionally-sized
        # sources, expect ~80% from source_a. Allow 5% tolerance.
        assert 0.70 < fraction_a < 0.90, f"Expected ~80% from source_a, got {fraction_a:.1%}"

    def test_epoch_increments(self, source_a, source_b):
        """Each full iteration increments epoch by 1."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))

        assert mixed.epoch.get_value() == 0

        # First iteration
        _ = list(mixed)
        assert mixed.epoch.get_value() == 1

        # Second iteration
        _ = list(mixed)
        assert mixed.epoch.get_value() == 2

    def test_deterministic_with_seed(self, source_a, source_b):
        """Same seed produces same element order across separate instances."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))

        mixed1 = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(99))
        elements1 = list(mixed1)

        # Reset sources for second run
        source_a.reset()
        source_b.reset()

        mixed2 = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(99))
        elements2 = list(mixed2)

        # Same seed → same sequence of source selections
        labels1 = [e["label"] for e in elements1]
        labels2 = [e["label"] for e in elements2]
        assert labels1 == labels2


# ============================================================================
# TestMixDataSourcesEdgeCases
# ============================================================================


class TestMixDataSourcesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_source_passthrough(self, source_a):
        """With one source and weight [1.0], behaves like iterating the source."""
        config = MixDataSourcesConfig(num_sources=1, weights=(1.0,))
        mixed = MixDataSourcesNode(config, [source_a], rngs=nnx.Rngs(42))

        elements = list(mixed)
        assert len(elements) == len(source_a)
        # All elements come from source_a
        assert all(e["label"] == "a" for e in elements)

    def test_source_exhaustion(self, source_a, source_small):
        """When one source is shorter, continues sampling from the other.

        source_a has 10 elements, source_small has 3. Total = 13.
        Even if the RNG keeps trying source_small after its 3 elements are used,
        the implementation falls back to source_a.
        """
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_small], rngs=nnx.Rngs(42))

        elements = list(mixed)
        assert len(elements) == 13  # 10 + 3

    def test_getitem_returns_none(self, source_a, source_b):
        """Random access is not supported — __getitem__ returns None."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))
        assert mixed[0] is None

    def test_reset_resets_all_sources(self, source_a, source_b):
        """reset() propagates to child sources and resets internal state."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        mixed = MixDataSourcesNode(config, [source_a, source_b], rngs=nnx.Rngs(42))

        # Iterate once to advance state
        _ = list(mixed)
        assert mixed.epoch.get_value() == 1

        # Reset
        mixed.reset()
        assert mixed.epoch.get_value() == 0
        assert mixed.index.get_value() == 0

    def test_constructor_validates_num_sources(self, source_a):
        """Constructor raises if len(sources) != config.num_sources."""
        config = MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5))
        with pytest.raises(ValueError, match="num_sources"):
            MixDataSourcesNode(config, [source_a], rngs=nnx.Rngs(42))
