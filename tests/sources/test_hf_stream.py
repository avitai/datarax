"""The HuggingFace stream: always HF's streaming read, ordered by the pipeline's seed (C5b).

``HFStreamingSource`` reads ``load_dataset(..., streaming=True)``; a map-style HuggingFace dataset
is ``HFEagerSource``'s. It names records by arrival (``ARRIVAL``): ordinals count on across
passes and never repeat, so it refuses lookups by record. With the pipeline's key a pass is HF's
own buffer shuffle seeded from that key, ``set_epoch(pass)`` giving each pass its own order;
without it the dataset's order. It reads batched NumPy columns, each array column in its
feature's dtype, and keeps text and objects as provenance beside the batch.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.pipeline import Pipeline
from datarax.sources import from_hf, HFEagerSource, HFStreamingConfig, HFStreamingSource


datasets = pytest.importorskip("datasets")

_N = 12


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Serve a local dataset in place of the Hub, recording every ``load_dataset`` call."""
    features = datasets.Features(
        {
            "pixels": datasets.Array3D(shape=(2, 2, 3), dtype="uint8"),
            "label": datasets.Value("int64"),
            "text": datasets.Value("string"),
        }
    )
    dataset = datasets.Dataset.from_dict(
        {
            "pixels": [np.full((2, 2, 3), i, np.uint8) for i in range(_N)],
            "label": list(range(_N)),
            "text": [f"text {i}" for i in range(_N)],
        },
        features=features,
    )
    calls: list[dict[str, Any]] = []

    def load_dataset(name: str, split: str | None = None, **kwargs: Any) -> Any:
        calls.append({"name": name, "split": split, **kwargs})
        if kwargs.get("streaming"):
            return dataset.to_iterable_dataset(num_shards=3)
        return dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return calls


def _stream(**kwargs: Any) -> HFStreamingSource:
    return HFStreamingSource(HFStreamingConfig(name="local", split="train", **kwargs))


def _pass(source: HFStreamingSource, size: int = 5, **kwargs: Any) -> list[Batch]:
    batches = []
    while (batch := source.get_batch(size, **kwargs)).batch_size:
        batches.append(batch)
    return batches


def _labels(batches: list[Batch]) -> list[int]:
    return [int(v) for b in batches for v in b["label"]]


def _names(batches: list[Batch]) -> list[int]:
    return [int(v) for b in batches for v in from_words(np.asarray(b.indices))]


class TestAlwaysAStream:
    def test_it_loads_with_hugging_face_streaming(self, loads: list[dict[str, Any]]) -> None:
        _stream()

        assert [call["streaming"] for call in loads] == [True]

    def test_the_config_has_no_streaming_or_shuffle_field(self) -> None:
        for field in ("streaming", "shuffle"):
            with pytest.raises(TypeError):
                HFStreamingConfig(name="local", split="train", **{field: True})  # type: ignore[arg-type]

    def test_from_hf_builds_the_eager_source_unless_asked_to_stream(
        self, loads: list[dict[str, Any]]
    ) -> None:
        assert isinstance(from_hf("local", "train"), HFEagerSource)
        assert isinstance(from_hf("local", "train", streaming=True), HFStreamingSource)
        with pytest.raises(TypeError):
            from_hf("local", "train", eager=False)  # type: ignore[call-arg]

    def test_it_has_no_length(self, loads: list[dict[str, Any]]) -> None:
        with pytest.raises(NotImplementedError):
            len(_stream())


class TestArrivalNames:
    def test_ordinals_count_on_across_passes(self, loads: list[dict[str, Any]]) -> None:
        source = _stream()

        first, second = _pass(source), _pass(source)

        assert source.record_identity is RecordIdentity.ARRIVAL
        assert _names(first) == list(range(_N))
        assert _names(second) == list(range(_N, 2 * _N))
        assert {int(e) for b in second for e in b.epochs} == {1}

    def test_lookups_by_record_are_refused_naming_the_kind(
        self, loads: list[dict[str, Any]]
    ) -> None:
        source = _stream()
        batch = source.get_batch(3)

        with pytest.raises(TypeError, match="ARRIVAL"):
            source.provenance(batch.indices)
        with pytest.raises(TypeError, match="ARRIVAL"):
            source.record_keys(batch)


class TestTheOrder:
    def test_without_a_key_each_pass_is_the_dataset_s_order(
        self, loads: list[dict[str, Any]]
    ) -> None:
        source = _stream()

        assert _labels(_pass(source)) == _labels(_pass(source)) == list(range(_N))

    def test_the_pipeline_s_seed_orders_each_pass_reproducibly(
        self, loads: list[dict[str, Any]]
    ) -> None:
        def passes(seed: int) -> list[list[int]]:
            source = _stream(shuffle_buffer_size=6)
            key = jax.random.key(seed)
            return [_labels(_pass(source, key=key)) for _ in range(3)]

        orders = passes(0)

        assert orders == passes(0)
        assert orders != passes(1)
        assert len({tuple(order) for order in orders}) == 3
        assert all(sorted(order) == list(range(_N)) for order in orders)

    def test_a_pipeline_passes_its_seed_and_serves_every_record_each_pass(
        self, loads: list[dict[str, Any]]
    ) -> None:
        def served(seed: int) -> list[int]:
            pipeline = Pipeline(
                source=_stream(shuffle_buffer_size=6),
                stages=[],
                batch_size=4,
                rngs=nnx.Rngs(seed),
                shuffle=True,
                num_epochs=2,
            )
            return [int(v) for batch in pipeline for v in np.asarray(batch["label"])]

        first = served(3)

        assert first == served(3)
        assert sorted(first[:_N]) == sorted(first[_N:]) == list(range(_N))
        assert first[:_N] != first[_N:]


class TestTheRead:
    def test_arrays_keep_their_feature_dtype_and_text_travels_beside(
        self, loads: list[dict[str, Any]]
    ) -> None:
        batch, provenance = _stream().get_batch(4, with_provenance=True)

        assert batch["pixels"].dtype == np.uint8
        assert batch["pixels"].shape == (4, 2, 2, 3)
        assert batch["label"].dtype == np.int64
        assert "text" not in batch
        assert [p["text"] for p in provenance] == [f"text {i}" for i in range(4)]
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))

    def test_key_filters_select_the_columns(self, loads: list[dict[str, Any]]) -> None:
        batch = _stream(include_keys={"label"}).get_batch(2)

        assert set(batch.data) == {"label"}

    def test_the_spec_is_the_first_record_s_array_part(self, loads: list[dict[str, Any]]) -> None:
        spec = _stream().element_spec()

        assert spec == {
            "pixels": jax.ShapeDtypeStruct((2, 2, 3), np.uint8),
            "label": jax.ShapeDtypeStruct((), np.int32),
        }
