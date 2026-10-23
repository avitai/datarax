"""What identifies a TFDS split's per-batch decode, as a decoded copy of it records.

A copy of a split read per batch is checked against what the split's decode declares
(:class:`~datarax.core.data_source.SourceIdentified`): the dataset, version, config and
split, the SHA-256 of its ``features.json``, the features kept, and the versions of the libraries
that decode a record here. A decoder of another version decodes to other bytes in general (JPEG
is not bit-stable across libjpeg builds), so a library upgrade invalidates a copy. A ``Video``
feature has no copy: every decoded frame is far larger than its stored size.

Like :mod:`datarax.sources.tfds_source`, this module imports ``tensorflow_datasets`` at its top,
never inside a function (that module says why); ``datarax.sources`` reaches it only through that
module's lazy exports.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorflow_datasets as tfds
from google.protobuf.internal import api_implementation
from substrax.checkpoint.metadata import library_versions
from tensorflow_datasets.core.features import image_feature

from datarax.core.field_paths import KeyTuple, spelling


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitOrigin:
    """What a prepared split is, as a decoded copy of it records: never where its files are.

    Attributes:
        dataset: The dataset's name.
        version: Its version.
        config: Its builder config, or ``None``.
        split: The split, slices included.
        features: The SHA-256 of the dataset's ``features.json``, which the decode follows.
    """

    dataset: str
    version: str
    config: str | None
    split: str
    features: str


_DECODERS = ("tensorflow-datasets", "protobuf", "numpy", "pillow")
"""The distributions TFDS decodes a record with, besides OpenCV when it imports."""


def _decoder_versions() -> dict[str, str | None]:
    """The libraries TFDS decodes a record with here, and their versions.

    TFDS decodes images with OpenCV whenever it imports, else with Pillow; protobuf parses the
    example on the runtime the install selects (``upb``, ``cpp`` or ``python``).

    Returns:
        Each library's version, or ``None`` where it is not installed.
    """
    cv2 = image_feature.cv2
    versions: dict[str, str | None] = dict.fromkeys(_DECODERS)
    versions.update(library_versions(_DECODERS))  # a missing library stays None
    versions["protobuf"] = f"{versions['protobuf']} ({api_implementation.Type()})"
    return {**versions, "opencv": None if cv2 is None else str(cv2.__version__)}


def _refuse_videos(kept: Mapping[str, Any], keys: KeyTuple) -> None:
    """Refuse a ``Video`` anywhere among the kept features, a group's members included."""
    for name, feature in kept.items():
        here = (*keys, name)
        if isinstance(feature, Mapping):
            _refuse_videos(feature, here)
        elif isinstance(feature, tfds.features.Video):
            raise ValueError(
                f"feature {spelling(here)!r} is a Video, which decodes every frame to far more "
                "than its stored size: a split holding one has no decoded copy; read it per batch"
            )


def split_identity(  # noqa: DOC502 - _refuse_videos raises the ValueError
    origin: SplitOrigin, kept: Mapping[str, Any], filters: Mapping[str, Any]
) -> dict[str, Any]:
    """A split's per-batch decode, as ``source_identity()`` declares it.

    Args:
        origin: The split.
        kept: The features the decode keeps, by name.
        filters: The decode's key filters (``keys``, ``include_keys``, ``exclude_keys``).

    Returns:
        ``{"source": ..., "decoder": ...}``: the split, ``features.json``'s digest and the
        filters; the decoding libraries and their versions.

    Raises:
        ValueError: If a kept feature is a ``Video``, naming it.
    """
    _refuse_videos(kept, ())
    return {
        "source": {
            **dataclasses.asdict(origin),
            **{name: None if keys is None else sorted(keys) for name, keys in filters.items()},
        },
        "decoder": _decoder_versions(),
    }


__all__ = ["SplitOrigin", "split_identity"]
