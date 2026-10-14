"""The in-memory TFDS build of the test fixture, as recorded before it decoded through one writer.

``TFDSEagerSource`` at datarax ``460c47a`` read each split of the fixture
(:data:`~tests.test_common.tfds_fixture.FIXTURE`) whole through TFDS's own reader;
:data:`EAGER_BUILD` holds, per case, the TFDS options, each column's SHA-256 (field path to
digest) and the provenance names in row order, recorded by
``evidence/w2b/c5a5/impl/commit3/3b/record_golden.py``. Every build that decodes the fixture
(the eager source, ``materialize``) is held to it bit for bit.
"""

from __future__ import annotations

import hashlib
from typing import Any

import jax
import numpy as np


TRAIN = {  # the train split's columns
    "image": "5ae82d57dac0f261af01736d7966a0c53cc895333576bc6fe91507f44407f47c",
    "label": "cff022149390c370667dc9be265b7a23b242c696a94039b73e070e9269a2d661",
    "meta/score": "0bf0c60518d383040544172e242a798307e6e9cd47003b8e6c1ba94de24e953f",
}
TRAIN_NAMES = [  # record_0_<i> in the train split's row order
    6,
    16,
    19,
    12,
    14,
    18,
    5,
    13,
    15,
    4,
    10,
    17,
    0,
    8,
    3,
    1,
    9,
    7,
    11,
    2,
]
EAGER_BUILD: dict[str, tuple[dict[str, Any], dict[str, str], list[str]]] = {
    "train": ({}, TRAIN, [f"record_0_{i:02d}" for i in TRAIN_NAMES]),
    "test": (
        {"split": "test"},
        {
            "image": "ce4ed639b95697a30ce6043c54663e871c67a2017f784488179a0d91df4649ce",
            "label": "bdea935af2bbbb7ea658a856f2ba072967b08b2556b885ce7fca7f34a4d0d47d",
            "meta/score": "2f9e759aa0c4deee969f993b543c9f6bb530cd753046e250a2aa0e78c4c8fcf6",
        },
        [f"record_1_{i:02d}" for i in (1, 4, 3, 2, 5, 0)],
    ),
    "slice": (
        {"split": "train[3:17]"},
        {
            "image": "b7adab36ca01cb05bacf9906271ba6edf39be8783a38100b9d0e255f03947ed1",
            "label": "0bbb2b7d5c091c55f717a7a48c77184b99cc2d7d82551a6d54e19d9bf87c02a0",
            "meta/score": "0381f1f20279363580bc4f87bfd45de59ced6092b5b920a105330546d9037553",
        },
        [f"record_0_{i:02d}" for i in TRAIN_NAMES[3:17]],
    ),
    "supervised": (
        {"as_supervised": True},
        {key: TRAIN[key] for key in ("image", "label")},
        [],
    ),
    "filtered": ({"exclude_keys": {"name"}}, TRAIN, []),
}


def digests(columns: Any) -> dict[str, str]:
    """Each column's SHA-256, keyed by field path (``/`` between levels).

    Args:
        columns: Host columns, nested mappings of arrays.

    Returns:
        The hex digests.
    """
    flat = jax.tree_util.tree_flatten_with_path(columns)[0]
    return {
        "/".join(str(getattr(key, "key", key)) for key in path): hashlib.sha256(
            np.ascontiguousarray(leaf).tobytes()
        ).hexdigest()
        for path, leaf in flat
    }
