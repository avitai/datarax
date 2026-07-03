"""Shared data-generator factories for real-data scenario variants.

Each factory returns a lazy ``data_generator`` callable for a
:class:`~benchmarks.scenarios.base.ScenarioVariant`. A fresh
:class:`~benchmarks.fixtures.real_data.RealDataProvider` is constructed
per invocation with the scenario-wide default seed, so every adapter (and
every repetition) receives byte-identical raw numpy input.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from benchmarks.fixtures.real_data import RealDataProvider
from benchmarks.scenarios.base import DEFAULT_SEED


def cifar10_image_data(n: int, h: int = 32, w: int = 32) -> Callable[[], dict[str, Any]]:
    """Create a generator of n cifar10 images under the ``image`` key.

    Args:
        n: Number of images.
        h: Output height in pixels.
        w: Output width in pixels.

    Returns:
        Lazy generator yielding ``{"image": uint8 (n, h, w, 3)}``.
    """

    def generate() -> dict[str, Any]:
        provider = RealDataProvider(seed=DEFAULT_SEED)
        return {"image": provider.cifar10_images(n, h=h, w=w)}

    return generate


def wikitext_token_data(
    n: int, seq_len: int, vocab_size: int = 32000
) -> Callable[[], dict[str, Any]]:
    """Create a generator of n wikitext-103 sequences under ``tokens``.

    Args:
        n: Number of sequences.
        seq_len: Tokens per sequence.
        vocab_size: Hash-token vocabulary size.

    Returns:
        Lazy generator yielding ``{"tokens": int32 (n, seq_len)}``.
    """

    def generate() -> dict[str, Any]:
        provider = RealDataProvider(seed=DEFAULT_SEED)
        return {"tokens": provider.wikitext_tokens(n, seq_len, vocab_size=vocab_size)}

    return generate


def criteo_dense_data(n: int) -> Callable[[], dict[str, Any]]:
    """Create a generator of n Criteo dense feature rows under ``features``.

    Args:
        n: Number of rows.

    Returns:
        Lazy generator yielding ``{"features": float32 (n, 13)}``.
    """

    def generate() -> dict[str, Any]:
        provider = RealDataProvider(seed=DEFAULT_SEED)
        dense, _ = provider.criteo_features(n)
        return {"features": dense}

    return generate


def criteo_recommendation_data(n: int) -> Callable[[], dict[str, Any]]:
    """Create a generator of n Criteo dense + hashed-sparse rows.

    Mirrors the synthetic HTAB-1 layout: sparse ids are cast to float32
    and concatenated after the 13 dense columns.

    Args:
        n: Number of rows.

    Returns:
        Lazy generator yielding ``{"features": float32 (n, 39)}``.
    """

    def generate() -> dict[str, Any]:
        provider = RealDataProvider(seed=DEFAULT_SEED)
        dense, sparse = provider.criteo_features(n)
        return {"features": np.concatenate([dense, sparse.astype(np.float32)], axis=1)}

    return generate


def coco_pair_data(
    n: int,
    h: int,
    w: int,
    text_len: int = 77,
    vocab_size: int = 32000,
    image_dtype: str = "float32",
) -> Callable[[], dict[str, Any]]:
    """Create a generator of n COCO image/caption-token pairs.

    Args:
        n: Number of pairs.
        h: Output image height.
        w: Output image width.
        text_len: Tokens per caption.
        vocab_size: Hash-token vocabulary size.
        image_dtype: ``"float32"`` (pixels in [0, 1]) or ``"uint8"``.

    Returns:
        Lazy generator yielding ``{"image": ..., "tokens": int32 (n, text_len)}``.
    """

    def generate() -> dict[str, Any]:
        provider = RealDataProvider(seed=DEFAULT_SEED)
        images, tokens = provider.coco_image_text(
            n, h=h, w=w, text_len=text_len, vocab_size=vocab_size, image_dtype=image_dtype
        )
        return {"image": images, "tokens": tokens}

    return generate
