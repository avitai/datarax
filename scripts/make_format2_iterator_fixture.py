"""Write the format-2 iterator checkpoint the migration test restores.

datarax 0.1.11 saved a pipeline iterator's state through substrax 0.1.9's checkpoint store
in format 2: one ``model`` payload holding the state dictionary and a sidecar with the
caller's metadata. datarax now writes format 3 and reads such a root through
``ITERATOR_STATE_FORMAT2``; this script writes one with the releases that produced it, so
the test reads a real one. Run it in isolation, never from the project venv::

    uv run --no-project --with "datarax==0.1.11" --with "substrax==0.1.9" \
        python scripts/make_format2_iterator_fixture.py tests/checkpoint/fixtures/format2

The fixture is a few kilobytes: a sixteen-record memory source, batch four, one epoch and a
half in.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from flax import nnx

from datarax.checkpoint import IteratorCheckpoint
from datarax.pipeline import Pipeline
from datarax.pipeline.iteration import PipelineIterator
from datarax.sources import MemorySource, MemorySourceConfig


RECORDS = 16
BATCH = 4
STEP = 6


def main(argv: list[str] | None = None) -> int:
    """Write the fixture under the root named on the command line."""
    parser = argparse.ArgumentParser(
        description="Write the format-2 iterator checkpoint the migration test restores."
    )
    parser.add_argument("root", type=Path, help="Directory the fixture is written under")
    root = parser.parse_args(argv).root / "iterator_state"
    if root.exists():
        shutil.rmtree(root)

    source = MemorySource(
        MemorySourceConfig(shuffle=True),
        data={"x": np.arange(RECORDS, dtype=np.float32)},
        rngs=nnx.Rngs(0, shuffle=0),
    )
    pipeline = Pipeline(source=source, stages=[], batch_size=BATCH, rngs=nnx.Rngs(0))
    batches_per_epoch = RECORDS // BATCH
    # One iterator serves one epoch; ``reset`` starts the next, two batches into it.
    iterator = iter(pipeline)
    assert isinstance(iterator, PipelineIterator)
    for _ in range(batches_per_epoch):
        next(iterator)
    pipeline.reset()
    iterator = iter(pipeline)
    assert isinstance(iterator, PipelineIterator)
    for _ in range(STEP - batches_per_epoch):
        next(iterator)
    with IteratorCheckpoint(root, max_to_keep=1) as checkpoint:
        checkpoint.save(iterator, step=STEP, metadata={"run": "fixture", "epoch": 1})
    print(f"iterator_state at step {STEP}: {iterator.get_state()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
