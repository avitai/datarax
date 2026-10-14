"""The TFDS sources' modules, one responsibility each, and where each name lives.

``_tfrecord`` reads TFRecord shard files (frames, CRCs, the offset index) and ``_stream_order``
computes a stream pass's record order; neither imports TensorFlow Datasets. ``_tfds_prepared``
finds a split's prepared copy and the features both sources keep; ``_tfds_identity`` names what a
split's per-batch decode is, as a decoded copy of it records. ``tfds_streaming_source`` holds
the streaming source and ``tfds_source`` the eager one and ``from_tfds``. A name lives in exactly
one of them, so a test that patches a name patches the one the code reads. The modules import in
any order, which a cycle among them would break.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
from types import ModuleType

import pytest
from substrax.testing import run_python

from datarax.sources import (
    _stream_order,
    _tfds_identity,
    _tfds_prepared,
    _tfrecord,
    tfds_source,
    tfds_streaming_source,
)
from tests.jax_test_environment import forwarded_jax_environment


pytestmark = pytest.mark.tfds

_MODULES = (
    "datarax.sources._tfrecord",
    "datarax.sources._stream_order",
    "datarax.sources._tfds_prepared",
    "datarax.sources._tfds_identity",
    "datarax.sources.tfds_streaming_source",
    "datarax.sources.tfds_source",
)
_CHILD_SECONDS = 300.0
_REPO_ROOT = Path(__file__).resolve().parents[2]

_HOMES: dict[ModuleType, tuple[str, ...]] = {
    _tfrecord: (
        "TFRECORD_HEADER",
        "TFRECORD_CRC",
        "CRC_MASK_DELTA",
        "LARGEST_OFFSET",
        "LARGEST_SHARD",
        "Shard",
        "ShardIndex",
        "SHARED_DIRECTORY",
        "share_index",
        "mapped_index",
        "shared_shard_index",
        "remove",
        "pread",
        "masked_crc",
        "DamagedRecordError",
        "record_index",
        "payload",
        "split_shard",
    ),
    _stream_order: (
        "RecordId",
        "shard_ids",
        "interleaved",
        "buffer_shuffled",
        "CYCLE_LENGTH",
        "BLOCK_LENGTH",
    ),
    _tfds_prepared: (
        "prepared_builder",
        "prepare_call",
        "where",
        "kept_features",
        "supervised_keys",
    ),
    _tfds_identity: (
        "SplitOrigin",
        "split_identity",
        "_decoder_versions",
    ),
    tfds_streaming_source: (
        "TFDSStreamingConfig",
        "_Frame",
        "_not_streamable",
        "_features",
        "_KeptFeatures",
        "_StreamRead",
        "_ordered_ids",
        "_decoded_batch",
        "_PassUnits",
        "TFDSStreamDataset",
        "_Unit",
        "_TFDSStreamIterator",
        "TFDSStreamingSource",
    ),
    tfds_source: (
        "_tfrecord_only",
        "TFDSEagerConfig",
        "_not_prepared",
        "_array_record_builder",
        "_ExampleDecoder",
        "_per_batch_split",
        "_split_read",
        "TFDSEagerSource",
        "from_tfds",
    ),
}
_PLACED = [(module, name) for module, names in _HOMES.items() for name in names]


@pytest.mark.parametrize(
    ("module", "name"), _PLACED, ids=[f"{m.__name__.rsplit('.', 1)[1]}.{n}" for m, n in _PLACED]
)
def test_each_name_is_defined_in_its_module(module: ModuleType, name: str) -> None:
    assert name in vars(module)
    value = vars(module)[name]
    if isinstance(value, type) or callable(value):
        assert value.__module__ == module.__name__


def _used_by_its_own_code(module: ModuleType) -> set[str]:
    """The names ``module``'s statements other than its imports read."""
    statements = [
        node
        for node in ast.parse(inspect.getsource(module)).body
        if not isinstance(node, ast.Import | ast.ImportFrom)
    ]
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }


_ELSEWHERE = [
    (module, name)
    for module, names in _HOMES.items()
    if module is not tfds_source
    for name in names
]


@pytest.mark.parametrize(
    ("module", "name"),
    _ELSEWHERE,
    ids=[f"{m.__name__.rsplit('.', 1)[1]}.{n}" for m, n in _ELSEWHERE],
)
def test_tfds_source_binds_another_module_s_name_only_to_use_it(
    module: ModuleType, name: str
) -> None:
    """No name stays in ``tfds_source`` for its old path: one bound there is one its code reads.

    A patch of a name then lands where the code calling it reads it, never on a leftover alias.
    """
    if name in vars(tfds_source):
        assert name in _used_by_its_own_code(tfds_source)
        assert vars(tfds_source)[name] is vars(module)[name]


_PUBLIC = {
    tfds_source: {
        "TFDSEagerConfig",
        "TFDSEagerSource",
        "from_tfds",
    },
    tfds_streaming_source: {"TFDSStreamingConfig", "TFDSStreamingSource", "TFDSStreamDataset"},
}


@pytest.mark.parametrize("module", list(_PUBLIC), ids=lambda m: m.__name__.rsplit(".", 1)[1])
def test_a_public_module_defines_only_what_users_call_or_receive(module: ModuleType) -> None:
    """Its public names are what users call or receive; what a stream's run reads is the source's.

    ``TFDSStreamDataset`` is what ``pass_dataset`` and ``run_dataset`` return, Grain's slicing hook
    included; its read (``_StreamRead``) and the shards' offset index stay internal.
    """
    defined = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and getattr(value, "__module__", None) == module.__name__
    }
    assert defined == _PUBLIC[module]


_IMPORTS = """
import importlib, json, sys
for name in sys.argv[1:]:
    importlib.import_module(name)
print(json.dumps({"tfds": "tensorflow_datasets" in sys.modules}))
"""


def _imported(*modules: str) -> dict[str, bool]:
    """Import ``modules`` in order in a fresh interpreter; whether TFDS was imported."""
    child = run_python(
        _IMPORTS,
        *modules,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    return json.loads(child.check().stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", ["datarax.sources._tfrecord", "datarax.sources._stream_order"])
def test_reading_records_and_ordering_them_imports_no_tensorflow_datasets(module: str) -> None:
    assert _imported(module) == {"tfds": False}


def test_the_positive_control_the_streaming_source_imports_tensorflow_datasets() -> None:
    assert _imported("datarax.sources.tfds_streaming_source") == {"tfds": True}


def test_the_damaged_record_error_is_caught_from_the_package() -> None:
    """Users meet one TFDS error, raised by a stream read: they import it where every source is."""
    import datarax.sources  # noqa: PLC0415 - the package's own lazy export is under test

    assert datarax.sources.DamagedRecordError is _tfrecord.DamagedRecordError
    assert "DamagedRecordError" in datarax.sources.__all__


_ERROR_ONLY = """
import json, sys
import datarax.sources
datarax.sources.DamagedRecordError
print(json.dumps({"tfds": "tensorflow_datasets" in sys.modules}))
"""


def test_catching_the_error_imports_no_tensorflow_datasets() -> None:
    child = run_python(
        _ERROR_ONLY,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    assert json.loads(child.check().stdout.strip().splitlines()[-1]) == {"tfds": False}


_WITHOUT_CRC = """
import sys
sys.modules["google_crc32c"] = None  # hides the installed package
try:
    exec(sys.argv[1])
except ImportError as error:
    print(type(error).__name__ + ": " + str(error))
else:
    print("imported")
"""


@pytest.mark.parametrize(
    "statement",
    [
        "import datarax.sources._tfrecord",
        "from datarax.sources import DamagedRecordError",
        "from datarax.sources import TFDSStreamingSource",
    ],
)
def test_reading_records_without_google_crc32c_names_the_extra(statement: str) -> None:
    child = run_python(
        _WITHOUT_CRC,
        statement,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    printed = child.check().stdout.strip().splitlines()[-1]
    assert printed.startswith("ImportError: ")
    assert "google_crc32c" in printed and "datarax[data]" in printed


@pytest.mark.parametrize("first", _MODULES)
def test_the_modules_import_whichever_comes_first(first: str) -> None:
    """A cycle between two modules fails when one of them is imported first."""
    _imported(first, *(module for module in _MODULES if module != first))
