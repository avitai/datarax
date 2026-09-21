"""Tests for ``local_files_only`` and ``element_spec`` on TFDS sources.

TFDS has no native ``local_files_only`` kwarg — the contract is enforced by
skipping ``builder.download_and_prepare()`` when ``local_files_only=True``.
The user is then responsible for ensuring the data is already prepared in
``data_dir``; if not, ``tfds.load`` will surface its own error.

These are unit tests over that kwarg, so they name a dataset that does not exist and patch
what would load one. A test that names a real dataset reads it wherever the machine happens
to have it prepared, and is a fast no-op everywhere else — the difference is invisible in a
CI job that prepares no datasets.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.sources.tfds_source import (
    TFDSEagerConfig,
    TFDSEagerSource,
    TFDSStreamingConfig,
    TFDSStreamingSource,
)


# Same skip pattern as test_tfds_source.py — TFDS imports are flaky on macOS.
TFDS_TEST_SKIP_EXCEPTIONS = (ImportError, ModuleNotFoundError, OSError, RuntimeError)


def _mock_eager_arrays() -> dict[str, jnp.ndarray]:
    """The arrays an eager source would have materialised.

    ``TFDSEagerSource.__init__`` loads the whole split, which patching the builder alone does
    not prevent: on a machine that happens to have the dataset prepared the constructor reads
    it, and these tests assert nothing about the data.
    """
    return {"image": jnp.zeros((4, 2, 2, 1)), "label": jnp.zeros((4,), dtype=jnp.int32)}


def _mock_streaming_builder():
    """Return a MagicMock shaped like a TFDS builder for streaming-source tests."""
    mock_builder = MagicMock()
    mock_builder.info.splits = {"train": MagicMock(num_examples=8)}
    mock_tf_dataset = MagicMock()
    mock_tf_dataset.prefetch.return_value = mock_tf_dataset
    mock_builder.as_dataset.return_value = mock_tf_dataset
    return mock_builder


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (TFDSEagerConfig(name="mock", split="train", local_files_only=True), True),
        (TFDSEagerConfig(name="mock", split="train"), False),
    ],
    ids=["requested", "default"],
)
def test_tfds_eager_source_passes_local_files_only_to_prepare_builder(
    config: TFDSEagerConfig, expected: bool
) -> None:
    """The flag flows into ``_prepare_tfds_builder`` as a kwarg, and defaults to False."""
    with (
        patch(
            "datarax.sources.tfds_source._prepare_tfds_builder", return_value=MagicMock()
        ) as mock_prepare,
        patch.object(
            TFDSEagerSource, "_load_all_from_backend_to_jax", return_value=_mock_eager_arrays()
        ),
    ):
        TFDSEagerSource(config, rngs=nnx.Rngs(0))

        assert mock_prepare.call_args.kwargs.get("local_files_only") is expected


def test_tfds_streaming_source_passes_local_files_only_to_prepare_builder() -> None:
    """Streaming source honors ``local_files_only=True`` the same way."""
    with patch(
        "datarax.sources.tfds_source._prepare_tfds_builder",
        return_value=_mock_streaming_builder(),
    ) as mock_prepare:
        config = TFDSStreamingConfig(name="mock", split="train", local_files_only=True)
        try:
            TFDSStreamingSource(config, rngs=nnx.Rngs(0))
        except TFDS_TEST_SKIP_EXCEPTIONS:
            pytest.skip("TFDS not importable in this environment")
        except Exception:  # noqa: BLE001
            pass

        kwargs = mock_prepare.call_args.kwargs
        assert kwargs.get("local_files_only") is True


def test_prepare_tfds_builder_skips_download_when_local_files_only() -> None:
    """The internal helper actually skips ``download_and_prepare`` when the flag is True.

    This is the load-bearing assertion: the kwarg flowing through configs and
    source ``__init__`` is meaningless if ``_prepare_tfds_builder`` itself
    doesn't honor it.
    """
    from datarax.sources.tfds_source import _prepare_tfds_builder  # noqa: PLC0415

    # Stub tfds.builder to return a mock without invoking the real TFDS module.
    # Using sys.modules manipulation keeps this test runnable without TFDS installed.
    builder = MagicMock()
    builder.download_and_prepare = MagicMock()

    with patch("tensorflow_datasets.builder", return_value=builder, create=True):
        try:
            _prepare_tfds_builder(
                name="mock",
                data_dir=None,
                try_gcs=False,
                download_and_prepare_kwargs=None,
                beam_num_workers=None,
                local_files_only=True,
            )
        except TFDS_TEST_SKIP_EXCEPTIONS:
            pytest.skip("TFDS not importable in this environment")

    builder.download_and_prepare.assert_not_called()
