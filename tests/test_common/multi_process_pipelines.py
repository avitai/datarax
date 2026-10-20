"""The pipelines the multi-process tests run, built alike in every process and in the reference.

Each process of a ``jax.distributed`` run (``tests.test_common.processes``) and the single-process
reference build their pipelines here, so the processes and the reference differ only in where the
batches are placed. :data:`LENGTH` records in batches of :data:`BATCH_SIZE` end on a short final
batch whose rows equal one process's share of a full batch on two processes, the case a placement
must not mistake for a full batch.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from flax import nnx
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from datarax.core.config import BatchMixOperatorConfig, ElementOperatorConfig
from datarax.core.element_batch import ArrayValue
from datarax.core.index_words import from_words
from datarax.operators import ElementOperator
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.pipeline import Packing, Pipeline
from datarax.sources import MemorySource, MemorySourceConfig


LENGTH = 28
BATCH_SIZE = 8


def records() -> dict[str, Any]:
    """Every record's value, a three-wide image derived from it, and its name as provenance."""
    values = np.arange(LENGTH, dtype=np.float32)
    return {
        "value": values,
        "image": np.stack([values, values + 100, values + 200], axis=1),
        "name": [name(record) for record in range(LENGTH)],
    }


def name(record: int) -> str:
    """The provenance name of ``record``."""
    return f"r{record}"


def _noise(element: Any, key: jax.Array) -> Any:
    """Add noise keyed by the record's key to its value."""
    return element.update_data({"value": element.data["value"] + jax.random.normal(key, ())})


def draw_stages() -> list[nnx.Module]:
    """A per-record draw, then MixUp's batch draws: lambda and the partner permutation."""
    return [
        ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="aug"),
            fn=_noise,
            rngs=nnx.Rngs(aug=7),
        ),
        BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", alpha=0.4, data_field="image", stream_name="mix"),
            rngs=nnx.Rngs(mix=1),
        ),
    ]


def rows_on_every_device() -> NamedSharding:
    """A batch's rows split over every device of the run, on Auto axes."""
    mesh = jax.make_mesh((jax.device_count(),), ("data",), axis_types=(AxisType.Auto,))
    return NamedSharding(mesh, PartitionSpec("data"))


def pipeline(
    *,
    stages: list[nnx.Module] | None = None,
    sharding: NamedSharding | None = None,
    seed: int = 0,
    **options: Any,
) -> Pipeline:
    """One shuffled epoch over :func:`records`, placed by ``sharding``."""
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), records()),
        stages=[] if stages is None else stages,
        batch_size=BATCH_SIZE,
        rngs=nnx.Rngs(seed),
        shuffle=True,
        num_epochs=1,
        sharding=sharding,
        **options,
    )


PACKING = Packing(lengths={"tokens": 16, "value": 4}, max_segments=4, window=16)
"""How :func:`packed_pipeline` packs :func:`ragged_records`."""


def ragged_records() -> list[dict[str, Any]]:
    """:data:`LENGTH` records whose tokens spell their index, of 1 to 7 tokens each."""
    return [
        {
            "tokens": (100 * record + np.arange(1 + record % 7)).astype(np.int32),
            "value": np.float32(record),
            "name": name(record),
        }
        for record in range(LENGTH)
    ]


def packed_pipeline(*, sharding: NamedSharding | None = None, seed: int = 0) -> Pipeline[Packing]:
    """One shuffled epoch of :func:`ragged_records` packed into rows, placed by ``sharding``."""
    return Pipeline(
        source=MemorySource(MemorySourceConfig(), ragged_records()),
        stages=[],
        batch_size=BATCH_SIZE,
        rngs=nnx.Rngs(seed),
        shuffle=True,
        num_epochs=1,
        sharding=sharding,
        packing=PACKING,
    )


def shards(array: jax.Array) -> dict[str, list[Any]]:
    """This process's rows of a placed array, keyed by the global row each shard starts at."""
    return {
        str(shard.index[0].start or 0): np.asarray(shard.data).tolist()
        for shard in array.addressable_shards
    }


def record_shards(indices: ArrayValue) -> dict[str, list[Any]]:
    """This process's rows of a placed unit's ``indices``, as record indices by first global row.

    A batch's shard is its rows' record indices; a chunk's, ``(K, rows)``, keyed by the row axis
    after the chunk axis.

    Raises:
        TypeError: If ``indices`` is a host array, which a placed batch never holds.
    """
    if not isinstance(indices, jax.Array):
        raise TypeError(f"a placed batch's indices are a jax.Array, got {type(indices).__name__}")
    row_axis = indices.ndim - 2  # the words' axis is last
    return {
        str(shard.index[row_axis].start or 0): from_words(np.asarray(shard.data)).tolist()
        for shard in indices.addressable_shards
    }


def rng_counts(module: nnx.Module) -> list[int]:
    """Every RNG stream's count in ``module``, in graph order."""
    return [int(count) for count in jax.tree.leaves(nnx.state(module, nnx.RngCount))]


__all__ = [
    "BATCH_SIZE",
    "LENGTH",
    "PACKING",
    "draw_stages",
    "packed_pipeline",
    "pipeline",
    "record_shards",
    "ragged_records",
    "records",
    "rng_counts",
    "rows_on_every_device",
    "shards",
]
