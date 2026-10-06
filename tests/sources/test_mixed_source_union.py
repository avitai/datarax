"""A mix's records carry the union of its children's fields; a field some child lacks is ``Maybe``.

Design D8: "Every field of a pipeline's schema exists in every record at its static shape". A
field every child has stays a plain array. A field some child lacks is ``Maybe(value, present)``
in every batch of the mix: rows of a child that has it carry its value with ``present`` True (or
the child's own ``present`` when the child's field is already a ``Maybe``), rows of a child that
lacks it carry zeros with ``present`` False. Children are compared by their declared (device)
specs: a field whose shape, device dtype or nesting differs is refused naming both. Host columns
of one field stored in different dtypes of one kind (an int64 label beside an int32 one, both
int32 on the device) join at NumPy's lossless promotion, the same in every batch.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core import Maybe
from datarax.core.config import StructuralConfig
from datarax.core.index_words import to_words
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.pipeline import Pipeline
from datarax.sources.eager_source import EagerSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.mixing import child_columns, four_presence_cases, Sized


_N = 6


def _memory(columns: dict[str, Any]) -> MemorySource:
    return MemorySource(MemorySourceConfig(), columns)


def _mix(children: list[Any]) -> MixDataSourcesNode:
    weights = tuple(1.0 / len(children) for _ in children)
    return MixDataSourcesNode(MixDataSourcesConfig(weights=weights), children)


H, W, C, T = 4, 4, 3, 6


def _batch_of(mix: MixDataSourcesNode, positions: list[int]) -> Any:
    return mix.get_batch(np.asarray(mix.record_indices_at(0, len(mix)))[positions])


class TestTheFourPresenceCases:
    def test_a_field_some_child_lacks_is_maybe_and_one_all_have_is_plain(self) -> None:
        mix = four_presence_cases()
        batch = mix.get_batch(np.asarray(mix.record_indices_at(0, 8)))
        image, text = batch["image"], batch["text"]
        assert isinstance(image, Maybe) and isinstance(text, Maybe)
        assert isinstance(batch["label"], np.ndarray)
        owners = batch["label"]
        np.testing.assert_array_equal(owners, [0, 1, 2, 3, 0, 1, 2, 3])
        np.testing.assert_array_equal(image.present, np.isin(owners, [0, 1]))
        np.testing.assert_array_equal(text.present, np.isin(owners, [0, 2]))
        for row, owner in enumerate(owners):
            columns = mix.sources[owner].data
            local = row // 4
            if "image" in columns:
                np.testing.assert_array_equal(image.value[row], columns["image"][local])
            else:
                assert not image.value[row].any()
            if "text" in columns:
                np.testing.assert_array_equal(text.value[row], columns["text"][local])
            else:
                assert not text.value[row].any()

    def test_the_spec_is_the_union(self) -> None:
        spec = four_presence_cases().element_spec()
        assert spec["label"] == jax.ShapeDtypeStruct((), jnp.int32)
        assert spec["image"] == Maybe(
            jax.ShapeDtypeStruct((H, W, C), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_)
        )
        assert spec["text"] == Maybe(
            jax.ShapeDtypeStruct((T,), jnp.float32), jax.ShapeDtypeStruct((), jnp.bool_)
        )

    def test_every_presence_pattern_has_one_structure_and_the_spec(self) -> None:
        mix = four_presence_cases()
        structures = set()
        for child in range(4):
            batch = _batch_of(mix, [child, child + 4])  # rows of one child only
            structures.add(jax.tree.structure(batch))
            held = device_spec(jax.tree.map(array_to_spec_strip_leading, batch.data))
            assert held == mix.element_spec()
        assert len(structures) == 1

    def test_a_child_field_that_is_already_maybe_keeps_its_presence(self) -> None:
        class Columns(EagerSource):
            def __init__(self, columns: dict[str, Any]) -> None:
                super().__init__(StructuralConfig())
                self._store(columns)

        present = np.array([True, False, True, False, True, False])
        value = np.where(present[:, None], 1.0, 0.0).astype(np.float32) * np.ones((_N, T))
        text = Maybe(value.astype(np.float32), present)
        mix = _mix(
            [
                Columns({"text": text, "label": np.zeros(_N, np.int32)}),
                _memory(child_columns(1, ("text", "label"))),
            ]
        )
        batch = mix.get_batch(np.asarray(mix.record_indices_at(0, 12)))
        interleaved = np.stack([present, np.ones(_N, bool)], axis=1).ravel()
        np.testing.assert_array_equal(batch["text"].present, interleaved)

    def test_a_nested_mix_s_maybe_passes_through(self) -> None:
        inner = four_presence_cases()
        outer = _mix([inner, _memory(child_columns(9, ("image", "text", "label")))])
        assert outer.element_spec() == inner.element_spec()
        batch = outer.get_batch(np.asarray(outer.record_indices_at(0, 8)))
        np.testing.assert_array_equal(
            batch["image"].present, [True, True, True, True, False, True, False, True]
        )

    def test_a_mixed_field_refuses_arithmetic(self) -> None:
        batch = _batch_of(four_presence_cases(), [0, 1])
        with pytest.raises(TypeError, match="value_or"):
            np.asarray(batch["image"])
        with pytest.raises(TypeError):
            _ = batch["text"] * 2


class TestConflictsAndHostDtypes:
    def test_children_with_different_fields_are_mixed(self) -> None:
        """The union replaces the refusal of children whose fields differ."""
        assert len(four_presence_cases()) == 4 * _N

    @pytest.mark.parametrize(
        ("other", "match"),
        [
            ({"image": np.zeros((_N, 2, 2, C), np.float32)}, r"image.*\(4, 4, 3\).*\(2, 2, 3\)"),
            ({"image": np.zeros((_N, H, W, C), np.int32)}, r"image.*float32.*int32"),
            ({"label": np.zeros(_N, np.float32)}, r"label.*int32.*float32"),
        ],
        ids=["shape", "dtype", "float-beside-int"],
    )
    def test_a_field_that_differs_between_children_is_refused_naming_both(
        self, other: dict[str, Any], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _mix([_memory(child_columns(0, ("image", "label"))), _memory(other)])

    def test_a_field_that_holds_values_in_one_child_and_fields_in_another_is_refused(
        self,
    ) -> None:
        class Specified(Sized):
            def __init__(self, spec: Any) -> None:
                super().__init__(4)
                self.spec = spec

            def element_spec(self) -> Any:
                return self.spec

        image = jax.ShapeDtypeStruct((H, W, C), jnp.float32)
        with pytest.raises(ValueError, match=r"\['image'\].*\['image'\]\['rgb'\]"):
            _mix([Specified({"image": image}), Specified({"image": {"rgb": image}})])

    def test_host_labels_of_one_kind_join_at_the_lossless_promotion(self) -> None:
        """An int64 host label beside an int32 one: both int32 on the device, int64 on the host."""
        wide = _memory({"label": np.arange(_N, dtype=np.int64)})
        narrow = _memory({"label": np.arange(_N, dtype=np.int32)})
        mix = _mix([wide, narrow])
        assert mix.element_spec()["label"] == jax.ShapeDtypeStruct((), jnp.int32)
        for positions in ([0, 2], [1, 3], [0, 1]):
            batch = _batch_of(mix, positions)
            assert batch["label"].dtype == np.int64
            held = device_spec(jax.tree.map(array_to_spec_strip_leading, batch.data))
            assert held == mix.element_spec()


def test_a_pipeline_over_a_union_mix_serves_every_record_once() -> None:
    mix = four_presence_cases()
    pipe = Pipeline(source=mix, stages=[], batch_size=4, rngs=nnx.Rngs(0))
    assert sum(batch.batch_size for batch in pipe) == len(mix)


def test_the_words_named_are_the_words_read() -> None:
    mix = four_presence_cases()
    words = to_words(np.asarray([3, 22, 7], np.uint64))
    np.testing.assert_array_equal(mix.get_batch(words).indices, words)


def test_one_dag_compile_serves_every_presence_pattern() -> None:
    """A field's presence is data (``Maybe``), so batches holding different children's records
    run through one compiled DAG call."""

    class TouchesEveryField(nnx.Module):
        def __call__(self, batch: Any) -> Any:
            return batch.replace(data=jax.tree.map(lambda leaf: leaf, batch.data))

    mix = four_presence_cases()
    pipe = Pipeline(
        source=mix,
        stages=[TouchesEveryField()],
        batch_size=3,
        rngs=nnx.Rngs(0),
        drop_last=True,
        num_epochs=1,
    )
    batches = iter(pipe)
    jax.clear_caches()
    first = next(batches)
    with expect_compiles(0):
        rest = list(batches)
    patterns = {
        tuple(bool(present) for present in np.asarray(batch["text"].present))
        for batch in [first, *rest]
    }
    assert len(patterns) > 1, patterns
