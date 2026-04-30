"""Tests for ``local_files_only`` and ``element_spec`` on TFDS sources.

TFDS has no native ``local_files_only`` kwarg — the contract is enforced by
skipping ``builder.download_and_prepare()`` when ``local_files_only=True``.
The user is then responsible for ensuring the data is already prepared in
``data_dir``; if not, ``tfds.load`` will surface its own error.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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


def _mock_streaming_builder():
    """Return a MagicMock shaped like a TFDS builder for streaming-source tests."""
    mock_builder = MagicMock()
    mock_builder.info.splits = {"train": MagicMock(num_examples=8)}
    mock_tf_dataset = MagicMock()
    mock_tf_dataset.prefetch.return_value = mock_tf_dataset
    mock_builder.as_dataset.return_value = mock_tf_dataset
    return mock_builder


def test_tfds_eager_source_passes_local_files_only_to_prepare_builder() -> None:
    """``local_files_only=True`` flows into ``_prepare_tfds_builder`` as a kwarg."""
    with patch(
        "datarax.sources.tfds_source._prepare_tfds_builder", return_value=MagicMock()
    ) as mock_prepare:
        config = TFDSEagerConfig(name="mnist", split="train", local_files_only=True)
        try:
            TFDSEagerSource(config, rngs=nnx.Rngs(0))
        except TFDS_TEST_SKIP_EXCEPTIONS:
            pytest.skip("TFDS not importable in this environment")
        except Exception:  # noqa: BLE001 — we only assert on the kwarg below
            # Other errors past _prepare_tfds_builder are not relevant to this contract.
            pass

        kwargs = mock_prepare.call_args.kwargs
        assert kwargs.get("local_files_only") is True


def test_tfds_eager_source_default_local_files_only_is_false() -> None:
    """Default ``local_files_only=False`` preserves current download-on-demand behavior."""
    with patch(
        "datarax.sources.tfds_source._prepare_tfds_builder", return_value=MagicMock()
    ) as mock_prepare:
        config = TFDSEagerConfig(name="mnist", split="train")
        try:
            TFDSEagerSource(config, rngs=nnx.Rngs(0))
        except TFDS_TEST_SKIP_EXCEPTIONS:
            pytest.skip("TFDS not importable in this environment")
        except Exception:  # noqa: BLE001
            pass

        kwargs = mock_prepare.call_args.kwargs
        assert kwargs.get("local_files_only") is False


def test_tfds_streaming_source_passes_local_files_only_to_prepare_builder() -> None:
    """Streaming source honors ``local_files_only=True`` the same way."""
    with patch(
        "datarax.sources.tfds_source._prepare_tfds_builder",
        return_value=_mock_streaming_builder(),
    ) as mock_prepare:
        config = TFDSStreamingConfig(name="mnist", split="train", local_files_only=True)
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
