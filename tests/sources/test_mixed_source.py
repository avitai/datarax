"""``MixDataSourcesNode``: Grain's mix over indexed children, in one 64-bit index space.

Position ``k`` of an epoch belongs to the child Grain's ``MapDataset.mix`` selects for ``k``, at
the child position Grain reads there; an epoch is Grain's length, the most positions that serve
each child record at most once. A mixed record's index is its child's offset (the lengths of the
children before it) plus its index within that child, in two words. Grain's selection is
transcribed in uint32 words so it runs traced and on the host, and these tests hold it to
Grain's public API position by position. The mix holds no state of its own.
"""

from __future__ import annotations

import inspect
import math
import sys

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import PADDING_INDEX
from datarax.core.index_words import from_words, MAX_RECORDS, to_words
from datarax.core.prng import per_record_keys
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.mixing import grain_mix, grain_mix_indices, offsets_of, Sized
from tests.test_common.streams import non_array_state_leaves, RecordStream


def _mix(children: list[DataSourceModule], weights: tuple[float, ...]) -> MixDataSourcesNode:
    return MixDataSourcesNode(MixDataSourcesConfig(weights=weights), children)


def _memory(values: np.ndarray) -> MemorySource:
    return MemorySource(MemorySourceConfig(), {"x": values.astype(np.float32)})


def _sized(lengths: list[int], weights: tuple[float, ...]) -> MixDataSourcesNode:
    return _mix([Sized(length) for length in lengths], weights)


def _named(mix: MixDataSourcesNode, start: int, size: int, key: jax.Array | None = None) -> list:
    return [int(v) for v in from_words(mix.record_indices_at(start, size, key))]


class TestTheConfig:
    def test_weights_are_normalised(self) -> None:
        assert MixDataSourcesConfig(weights=(1.0, 3.0)).weights == pytest.approx((0.25, 0.75))

    @pytest.mark.parametrize("weights", [(0.5, 0.0), (-0.5, 1.5), ()], ids=["zero", "neg", "none"])
    def test_weights_that_are_not_all_positive_are_refused(
        self, weights: tuple[float, ...]
    ) -> None:
        """Grain mixes only positive proportions: a zero weight would never serve its child."""
        with pytest.raises(ValueError, match="positive"):
            MixDataSourcesConfig(weights=weights)

    def test_the_child_count_is_the_weight_count(self) -> None:
        assert "num_sources" not in inspect.signature(MixDataSourcesConfig).parameters
        with pytest.raises(ValueError, match="2 weights for 3 sources"):
            _mix([_memory(np.arange(4)) for _ in range(3)], (0.5, 0.5))

    def test_nothing_in_a_mix_is_random(self) -> None:
        config = MixDataSourcesConfig(weights=(0.5, 0.5))
        assert config.stochastic is False
        assert config.stream_name is None
        assert "rngs" not in inspect.signature(MixDataSourcesNode.__init__).parameters


class TestTheChildren:
    """A mix names its children's records by stable position: only unsharded indexed children."""

    @pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
    def test_a_stream_child_is_refused_naming_its_kind(self, kind: RecordIdentity) -> None:
        stream = RecordStream({"x": np.arange(4, dtype=np.float32)}, kind=kind)
        with pytest.raises(TypeError, match=f"child 1.*{kind.name}"):
            _mix([_memory(np.arange(4)), stream], (0.5, 0.5))

    def test_a_worker_sharded_child_is_refused(self) -> None:
        sharded = MemorySource(
            MemorySourceConfig(num_workers=2, shard_id=0), {"x": np.arange(4, dtype=np.float32)}
        )
        with pytest.raises(ValueError, match="child 0.*num_workers=2"):
            _mix([sharded, _memory(np.arange(4))], (0.5, 0.5))

    def test_a_child_without_records_is_refused(self) -> None:
        with pytest.raises(ValueError, match="child 1.*no records"):
            _mix([_memory(np.arange(4)), _memory(np.arange(0))], (0.5, 0.5))

    def test_a_child_whose_spec_cannot_be_read_is_refused_naming_it(self) -> None:
        class Unspecified(Sized):
            def element_spec(self) -> None:
                raise NotImplementedError("no spec")

        with pytest.raises(ValueError, match="child 1 .*Unspecified.*element_spec"):
            _mix([Sized(4), Unspecified(4)], (0.5, 0.5))

    def test_children_with_different_records_are_refused_naming_the_field(self) -> None:
        wide = MemorySource(MemorySourceConfig(), {"x": np.zeros((4, 3), np.float32)})
        with pytest.raises(ValueError, match=r"\['x'\].*\(3,\).*\(\)"):
            _mix([_memory(np.arange(4)), wide], (0.5, 0.5))

    def test_a_mix_is_a_child_like_any_indexed_source(self) -> None:
        inner = _mix([_memory(np.arange(4)), _memory(100 + np.arange(4))], (0.5, 0.5))
        outer = _mix([inner, _memory(200 + np.arange(8))], (0.5, 0.5))
        served = np.asarray(outer.get_records(outer.record_indices_at(0, len(outer)))["x"])
        expected = [0, 200, 100, 201, 1, 202, 101, 203, 2, 204, 102, 205, 3, 206, 103, 207]
        np.testing.assert_array_equal(served, np.asarray(expected, np.float32))


# Weight sets and Grain's period ``S`` for each: the smallest weight scaled to 100, the others
# scaled alike and truncated (Grain's ``_float_to_int_proportions``).
_WEIGHT_SETS = [
    ((0.5, 0.5), 200),
    ((0.3, 0.7), 333),
    ((0.25, 0.25, 0.5), 400),
    ((1.0, 2.0, 3.5), 650),
    ((0.001, 0.999), 100_000),
    ((1e-4, 1 - 1e-4), 1_000_000),
]


def _lengths_for(weights: tuple[float, ...], periods: int, period: int) -> list[int]:
    """Child lengths long enough that an epoch spans ``periods`` of Grain's periods."""
    total = sum(weights)
    return [math.ceil(periods * period * w / total) + 3 for w in weights]


class TestGrainsSelection:
    """The mix serves, position by position, what Grain's public ``MapDataset.mix`` serves."""

    @pytest.mark.parametrize(("weights", "period"), _WEIGHT_SETS)
    def test_every_position_of_five_periods_equals_grains(
        self, weights: tuple[float, ...], period: int
    ) -> None:
        lengths = _lengths_for(weights, 5, period)
        mix = _sized(lengths, weights)
        positions = 5 * period
        assert len(mix) >= positions
        expected = grain_mix_indices(lengths, mix.weights, range(positions))
        assert _named(mix, 0, positions) == expected

    def test_positions_up_to_two_to_the_63_equal_grains(self) -> None:
        weights = (0.3, 0.7)
        lengths = [1 << 61, (1 << 62) + (1 << 61)]
        mix = _sized(lengths, weights)
        last = len(mix) - 1
        assert last > (1 << 62)
        positions = [(1 << 31) - 3, (1 << 32) + 7, (1 << 53) + 1, (1 << 62) + 5, last]
        expected = grain_mix_indices(lengths, mix.weights, positions)
        assert [_named(mix, p, 1)[0] for p in positions] == expected

    @pytest.mark.parametrize(
        ("lengths", "weights", "length"),
        [
            ([9, 14], (0.5, 0.5), 18),
            ([1000, 10], (0.5, 0.5), 20),
            ([9, 14], (0.3, 0.7), 20),
            ([5, 9, 14], (0.2, 0.3, 0.5), 25),
            ([1 << 52, 3 << 50], (0.5, 0.5), 3 << 51),
        ],
    )
    def test_an_epoch_is_grains_length(
        self, lengths: list[int], weights: tuple[float, ...], length: int | None
    ) -> None:
        """Grain's length: the most positions that serve each child record at most once."""
        mix = _sized(lengths, weights)
        assert len(mix) == len(grain_mix(lengths, mix.weights))
        if length is not None:
            assert len(mix) == length

    def test_the_length_past_two_to_the_53_is_exact(self) -> None:
        """Grain computes the length in float64; the mix keeps its rule in integers."""
        lengths = [(1 << 53) + 1, (1 << 53) + 1]
        mix = _sized(lengths, (0.5, 0.5))
        assert len(mix) == (1 << 54) + 2  # each child's records twice over, at 1:1
        assert len(grain_mix(lengths, mix.weights)) == 1 << 54

    @pytest.mark.parametrize(("weights", "period"), _WEIGHT_SETS[:4])
    def test_an_epoch_serves_grains_counts_each_record_at_most_once(
        self, weights: tuple[float, ...], period: int
    ) -> None:
        lengths = _lengths_for(weights, 3, period)
        mix = _sized(lengths, weights)
        named = np.asarray(_named(mix, 0, len(mix)), np.uint64)
        expected = np.asarray(grain_mix_indices(lengths, mix.weights, range(len(mix))), np.uint64)
        np.testing.assert_array_equal(named, expected)
        assert len(np.unique(named)) == len(named)
        owners = np.searchsorted(np.asarray(offsets_of(lengths)), named, side="right") - 1
        assert (np.bincount(owners, minlength=len(lengths)) <= np.asarray(lengths)).all()

    def test_the_traced_naming_equals_the_host_naming(self) -> None:
        lengths = [1 << 30, 1 << 30]
        mix = _sized(lengths, (0.3, 0.7))
        names = jax.jit(lambda start: mix.record_indices_at(start, 64, None))
        for start in (0, 12345, 1 << 30, len(mix) - 64, len(mix) - 10):
            np.testing.assert_array_equal(
                names(jnp.int32(start)), mix.record_indices_at(start, 64, None)
            )

    def test_the_naming_program_does_not_grow_with_the_period(self) -> None:
        """No table of Grain's period is a constant of the program: S = 200 and S = 1e6 alike."""

        def program_text(weights: tuple[float, ...]) -> str:
            mix = _sized([1 << 24, 1 << 24], weights)
            names = jax.jit(lambda start: mix.record_indices_at(start, 256, None))
            return names.lower(jnp.int32(0)).as_text()

        small, large = program_text((0.5, 0.5)), program_text((1e-4, 1 - 1e-4))
        assert len(large) < 1.01 * len(small)


class TestOneIndexSpace:
    """Every child's records have one 64-bit index: the child's offset plus its own index."""

    _LENGTHS = [(1 << 31) - 10, (1 << 31) + 5, (1 << 32) + 5]

    def test_indices_past_two_to_the_31_and_32_are_the_offset_plus_the_child_index(self) -> None:
        weights = (1 / 3, 1 / 3, 1 / 3)
        mix = _sized(self._LENGTHS, weights)
        positions = [0, 1, 2, 3 * 2**30 + 1, len(mix) - 3, len(mix) - 2, len(mix) - 1]
        named = [_named(mix, p, 1)[0] for p in positions]
        assert named == grain_mix_indices(self._LENGTHS, mix.weights, positions)
        assert max(named) > 1 << 32
        assert len(set(named)) == len(named)
        assert all(index < sum(self._LENGTHS) for index in named)

    def test_one_child_index_in_three_children_gives_three_keys(self) -> None:
        indices = jnp.asarray(to_words([offset + 5 for offset in offsets_of(self._LENGTHS)]))
        assert not (np.asarray(indices) == PADDING_INDEX).all(axis=1).any()
        keys = jax.random.key_data(
            per_record_keys(
                jax.random.key(0), indices, jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.int32)
            )
        )
        assert len({tuple(row) for row in np.asarray(keys).tolist()}) == 3

    def test_a_space_reaching_the_padding_index_is_refused(self) -> None:
        third = sys.maxsize
        assert 3 * third >= MAX_RECORDS
        with pytest.raises(ValueError, match="padding index"):
            _sized([third, third, third], (1 / 3, 1 / 3, 1 / 3))

    def test_an_epoch_past_python_s_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="sys.maxsize"):
            _sized([1 << 62, 1 << 62], (0.5, 0.5))


class TestNoState:
    """A mix is its children and its static proportions: no Variable, no key, no counter."""

    @staticmethod
    def _pair(weights: tuple[float, ...] = (0.5, 0.5)) -> MixDataSourcesNode:
        return _mix([_memory(np.arange(9)), _memory(100 + np.arange(14))], weights)

    def test_no_variable_holds_a_python_value(self) -> None:
        assert non_array_state_leaves(self._pair()) == []

    def test_a_tree_mode_split_and_merge_names_the_same_records(self) -> None:
        mix = self._pair()
        graphdef, state = nnx.split(mix, graph=False)
        merged = nnx.merge(graphdef, state)
        key = jax.random.key(3)
        np.testing.assert_array_equal(
            merged.record_indices_at(4, 9, key), mix.record_indices_at(4, 9, key)
        )

    def test_identically_built_mixes_share_one_graphdef_and_one_trace(self) -> None:
        first, second = self._pair(), self._pair()
        assert nnx.graphdef(first) == nnx.graphdef(second)
        assert hash(nnx.graphdef(first)) == hash(nnx.graphdef(second))
        names = nnx.jit(lambda mix, start: mix.record_indices_at(start, 6, None))
        three, five = jnp.int32(3), jnp.int32(5)
        with expect_compiles(1):
            jax.block_until_ready(names(first, three))
        with expect_compiles(0):
            jax.block_until_ready(names(second, five))

    def test_mixes_differing_only_in_weights_compile_apart_and_serve_their_own(self) -> None:
        even, skewed = self._pair((0.5, 0.5)), self._pair((0.3, 0.7))
        assert nnx.graphdef(even) != nnx.graphdef(skewed)
        names = nnx.jit(lambda mix, start: mix.record_indices_at(start, 6, None))
        zero = jnp.int32(0)
        jax.block_until_ready(names(even, zero))
        with expect_compiles(1):
            served = names(skewed, zero)
        np.testing.assert_array_equal(served, skewed.record_indices_at(0, 6, None))
        assert not np.array_equal(served, even.record_indices_at(0, 6, None))

    def test_a_pipeline_sharing_its_rngs_splits_in_tree_mode(self) -> None:
        def pipeline() -> Pipeline:
            return Pipeline(
                source=self._pair(), stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=True
            )

        graphdef, state = nnx.split(pipeline(), graph=False)
        merged, reference = nnx.merge(graphdef, state), pipeline()
        for _ in range(3):
            np.testing.assert_array_equal(merged.step().indices, reference.step().indices)

    def test_the_repr_names_the_children_the_weights_and_the_length(self) -> None:
        text = repr(self._pair((0.25, 0.75)))
        assert "MixDataSourcesNode" in text
        assert text.count("MemorySource") == 2
        assert "0.25" in text and "0.75" in text
        assert f"length={len(self._pair((0.25, 0.75)))}" in text


def test_grain_is_the_published_reference() -> None:
    """The oracle above is Grain's public API, not a copy of the arithmetic under test."""
    assert grain_mix([5, 3], (0.5, 0.5))[1] == 5
    assert isinstance(grain_mix([5, 3], (0.5, 0.5)), grain.MapDataset)
