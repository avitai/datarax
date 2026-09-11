"""Tests for ``local_files_only`` and ``element_spec`` on HuggingFace sources.

``datasets`` 4.x routes ``local_files_only`` through ``DownloadConfig`` rather
than as a top-level ``load_dataset`` kwarg. ``HFStreamingSource.element_spec``
is needed because the eager-source's generic spec (inherited from
``EagerSourceBase``) does not apply to iterator-backed streams.
"""

from __future__ import annotations

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.spec import validate_batch
from datarax.sources.hf_source import (
    HFEagerConfig,
    HFEagerSource,
    HFStreamingConfig,
    HFStreamingSource,
)


def _local_files_only_from_kwargs(kwargs: dict) -> bool:
    """Extract local_files_only from either direct kwarg or DownloadConfig."""
    if "local_files_only" in kwargs:
        return bool(kwargs["local_files_only"])
    download_config = kwargs.get("download_config")
    if download_config is None:
        return False
    return bool(getattr(download_config, "local_files_only", False))


def _serve(
    monkeypatch: pytest.MonkeyPatch,
    dataset: datasets.Dataset,
    captured: dict[str, object] | None = None,
) -> None:
    """Make ``datasets.load_dataset`` return ``dataset``, recording its keyword arguments."""

    def load_dataset(name, split=None, **kwargs):  # noqa: ARG001
        if captured is not None:
            captured.update(kwargs)
        return dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


@pytest.fixture
def numeric_dataset():
    """Numeric-only dataset suitable for both eager and streaming HF wrappers."""
    return datasets.Dataset.from_dict(
        {
            "label": list(range(8)),
            "feature": [np.random.randn(4).astype(np.float32) for _ in range(8)],
        }
    )


def test_hf_eager_source_passes_local_files_only_to_load_dataset(
    numeric_dataset, monkeypatch
) -> None:
    """``local_files_only=True`` flows to ``DownloadConfig.local_files_only``."""
    captured: dict[str, object] = {}
    _serve(monkeypatch, numeric_dataset, captured)

    config = HFEagerConfig(name="mock", split="train", local_files_only=True)
    HFEagerSource(config, rngs=nnx.Rngs(0))

    assert _local_files_only_from_kwargs(captured) is True


def test_hf_eager_source_default_local_files_only_is_false(numeric_dataset, monkeypatch) -> None:
    """Default ``local_files_only=False`` keeps download-on-demand behavior."""
    captured: dict[str, object] = {}
    _serve(monkeypatch, numeric_dataset, captured)

    config = HFEagerConfig(name="mock", split="train")
    HFEagerSource(config, rngs=nnx.Rngs(0))

    assert _local_files_only_from_kwargs(captured) is False


def test_hf_streaming_source_passes_local_files_only_to_load_dataset(
    numeric_dataset, monkeypatch
) -> None:
    """``local_files_only`` is honored on the streaming path via DownloadConfig."""
    captured: dict[str, object] = {}
    _serve(monkeypatch, numeric_dataset, captured)

    config = HFStreamingConfig(name="mock", split="train", local_files_only=True)
    HFStreamingSource(config, rngs=nnx.Rngs(0))

    assert _local_files_only_from_kwargs(captured) is True


def test_hf_streaming_source_element_spec_peeks_first_element(numeric_dataset, monkeypatch) -> None:
    """``HFStreamingSource.element_spec`` returns shape/dtype for one element.

    Streaming sources can't strip a leading dimension because they iterate one
    element at a time. The spec is derived by peeking the first element from
    the backend iterator and describing the record the source emits for it.
    """
    _serve(monkeypatch, numeric_dataset)

    config = HFStreamingConfig(name="mock", split="train")
    source = HFStreamingSource(config, rngs=nnx.Rngs(0))

    spec = source.element_spec()

    assert isinstance(spec, dict)
    assert set(spec.keys()) == {"label", "feature"}

    label_spec = spec["label"]
    feature_spec = spec["feature"]
    assert isinstance(label_spec, jax.ShapeDtypeStruct)
    assert isinstance(feature_spec, jax.ShapeDtypeStruct)
    # ``label`` is a Python int → scalar; ``feature`` is a length-4 float array.
    assert label_spec.shape == ()
    assert feature_spec.shape == (4,)
    assert feature_spec.dtype == jnp.float32


@pytest.mark.parametrize(
    "filters",
    [{"include_keys": {"feature"}}, {"exclude_keys": {"label"}}],
    ids=["include_keys", "exclude_keys"],
)
def test_hf_streaming_source_element_spec_honors_key_filters(
    numeric_dataset, monkeypatch, filters
) -> None:
    """The declared keys are exactly the keys the filtered records carry."""
    _serve(monkeypatch, numeric_dataset)
    source = HFStreamingSource(
        HFStreamingConfig(name="mock", split="train", **filters), rngs=nnx.Rngs(0)
    )

    spec = source.element_spec()

    assert set(spec) == {"feature"}
    assert set(source.get_batch(2)) == {"feature"}


def test_hf_streaming_source_element_spec_describes_its_converted_batches(monkeypatch) -> None:
    """Python floats reach batches as float32 arrays, and the spec declares float32."""
    dataset = datasets.Dataset.from_dict(
        {"score": [float(i) for i in range(6)], "label": list(range(6))}
    )
    _serve(monkeypatch, dataset)
    source = HFStreamingSource(HFStreamingConfig(name="mock", split="train"), rngs=nnx.Rngs(0))

    spec = source.element_spec()

    assert spec["score"] == jax.ShapeDtypeStruct((), jnp.float32)
    validate_batch(source.get_batch(3), spec, batch_size=3)
