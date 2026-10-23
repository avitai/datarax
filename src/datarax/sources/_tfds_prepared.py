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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import tensorflow_datasets as tfds

from datarax.core.field_paths import KeyTuple
from datarax.sources.source_ops import KeyFilter


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


def decoded_infos(feature: Any, keys: KeyTuple = ()) -> Iterator[tuple[KeyTuple, Any]]:
    """Each array a record decoded from ``feature`` holds, by its key tuple, with its tensor info.

    TFDS describes what a feature decodes to (``get_tensor_info``): a dict of infos for a
    ``FeaturesDict`` and for a ``Sequence`` of one, whose infos carry the sequence's length
    first, and one info for every other feature. A mapping of features is walked by name.

    Args:
        feature: A TFDS feature, or a mapping of names to features (or to such mappings).
        keys: ``feature``'s key tuple in the record, ``()`` at its root.

    Yields:
        ``(keys, info)`` per decoded array, the info's ``shape`` and ``np_dtype`` as stored.
    """
    if isinstance(feature, Mapping):
        for name, child in feature.items():
            yield from decoded_infos(child, (*keys, name))
    else:
        yield from _tensor_infos(feature.get_tensor_info(), keys)


def _tensor_infos(tree: Any, keys: KeyTuple) -> Iterator[tuple[KeyTuple, Any]]:
    """The infos of one feature's ``get_tensor_info`` tree, by key tuple."""
    if isinstance(tree, tfds.features.TensorInfo):
        yield keys, tree
        return
    for name, child in tree.items():
        yield from _tensor_infos(child, (*keys, name))


def decoded_names(features: Any) -> dict[str, Any]:
    """The fields a record decoded from ``features`` holds, nested as the record nests them.

    The record's structure is TFDS's own (:func:`decoded_infos`): a ``FeaturesDict`` and a
    ``Sequence`` of one decode to groups, every other feature to one field.

    Args:
        features: A ``FeaturesDict``, or a mapping of names to features.

    Returns:
        The names, a group as a dict and a field as ``None``.
    """
    names: dict[str, Any] = {}
    for keys, _ in decoded_infos(features):
        *groups, name = keys
        node = names
        for group in groups:
            node = node.setdefault(group, {})
        node[name] = None
    return names


def feature_tree(features: Any) -> dict[str, Any]:
    """``features`` as nested dicts down to whole features, for a filter to select from.

    Only a ``FeaturesDict`` opens: a ``Sequence`` stays one feature, since TFDS decodes it whole,
    and a filter naming a field inside it keeps it for the decode and selects from the record.

    Args:
        features: A ``FeaturesDict``, or a mapping of names to features.

    Returns:
        The features by name, a ``FeaturesDict`` as a dict.
    """
    return {
        name: feature_tree(feature) if isinstance(feature, tfds.features.FeaturesDict) else feature
        for name, feature in features.items()
    }


def kept_features(
    record: dict[str, Any], keys: Sequence[str] | None, key_filter: KeyFilter
) -> dict[str, Any]:
    """The features of ``record`` a source keeps: the supervised ones if asked, then filtered.

    Args:
        record: A decoded record, or the dataset's features as :func:`feature_tree` gives them.
        keys: The supervised features kept, or ``None`` for every feature.
        key_filter: The source's filter, resolved when it was built.

    Returns:
        The kept features.
    """
    if keys is not None:
        record = {key: record[key] for key in keys}
    return key_filter.keep(record)


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
