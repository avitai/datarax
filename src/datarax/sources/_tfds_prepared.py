"""A TFDS split's prepared copy, and the features both TFDS sources keep of each record.

Neither TFDS source prepares a dataset, because preparing imports TensorFlow: a copy that is not
prepared, or not in the format a source reads, is refused with the error the source builds, naming
the call that prepares it (:func:`prepared_builder`, :func:`prepare_call`). A record keeps its
supervised features when ``as_supervised`` asks for them (:func:`supervised_keys`), then those the
key filters keep (:func:`kept_features`).

This module imports ``tensorflow_datasets`` at its top, as both sources' modules do.
``datarax.sources`` exports the TFDS sources' names lazily, so ``import datarax.sources`` imports no
TFDS: the import happens on the line that imports a TFDS source or ``from_tfds``. TFDS's first
import enters ``etils.epy.lazy_imports()``, which replaces ``builtins.__import__`` for the whole
process while it runs. A garbage collection in that window runs finalizers whose imports fail, so
the window must not open inside a constructor in the middle of a run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence, Set as AbstractSet
from typing import Any

import jax
import tensorflow_datasets as tfds

from datarax.sources.source_ops import filter_keys


def prepared_builder(  # noqa: DOC503 - the exception raised is the one ``refuse`` builds
    name: str, data_dir: str | None, file_format: str, refuse: Callable[[str], Exception]
) -> Any:
    """The builder of ``name`` as prepared in ``data_dir`` in ``file_format``, reading no record.

    Args:
        name: The TFDS dataset name, with its config when it has one.
        data_dir: The data directory; TFDS's default when ``None``.
        file_format: The format the copy must be prepared in (``"array_record"``, ``"tfrecord"``).
        refuse: Builds the error refusing a copy, from what was found instead.

    Returns:
        The TFDS builder.

    Raises:
        Exception: What ``refuse`` builds, if the copy is not prepared in ``file_format``.
    """
    try:
        builder = tfds.builder(name, data_dir=data_dir)
    except tfds.core.DatasetNotFoundError as error:
        raise refuse("it is not prepared there") from error
    if not builder.is_prepared():
        raise refuse(f"{builder.data_dir} is not prepared")
    formats = builder.info.available_file_formats()
    if tfds.core.FileFormat(file_format) not in formats:
        found = ", ".join(sorted(fmt.value for fmt in formats))
        raise refuse(f"{builder.data_dir} holds it prepared as {found}")
    return builder


def prepare_call(name: str, data_dir: str | None, file_format: str) -> str:
    return (
        f"tfds.builder({name!r}, data_dir={data_dir!r}, file_format={file_format!r})"
        ".download_and_prepare()"
    )


def where(data_dir: str | None) -> str:
    return data_dir if data_dir is not None else "TFDS's default data directory"


def kept_features(
    record: dict[str, Any],
    keys: Sequence[str] | None,
    include_keys: AbstractSet[str] | None,
    exclude_keys: AbstractSet[str] | None,
) -> dict[str, Any]:
    """The features of ``record`` a source keeps: the supervised ones if asked, then filtered."""
    if keys is not None:
        record = {key: record[key] for key in keys}
    return filter_keys(record, include_keys, exclude_keys)


def supervised_keys(info: Any, dataset: str) -> list[str]:
    """The features ``as_supervised`` keeps: those the dataset declares supervised.

    Args:
        info: The dataset's ``DatasetInfo``.
        dataset: The dataset's name, for the refusal.

    Returns:
        The supervised features' names.

    Raises:
        ValueError: If the dataset declares no supervised keys.
    """
    declared = info.supervised_keys
    if declared is None:
        raise ValueError(f"{dataset} declares no supervised keys, which as_supervised keeps")
    return [str(key) for key in jax.tree.leaves(declared)]
