# Grain and Datarax Side by Side

Each tutorial here runs one job with Google Grain and with Datarax, so the two APIs sit
next to each other on the same records. The scripts import and exercise both libraries;
Grain is installed as a Datarax dependency.

| Tutorial | What it compares |
|---|---|
| [`01_grain_datarax_quickref.py`](01_grain_datarax_quickref.py) | Reading records, per-record randomness, batching, and resuming an interrupted epoch from saved iterator state: what each checkpoint holds and where each library's randomness comes from |
| [`02_randomness_and_learnable_operators_tutorial.py`](02_randomness_and_learnable_operators_tutorial.py) | What a record's randomness depends on in each library, reproduced and checked under another shuffle order and batch size; then a learnable stage and a learnable mixture of image operators, fitted by differentiating an epoch of `Pipeline.scan` |
| [`03_sharding_guide.py`](03_sharding_guide.py) | One program run as every process, or as one: which rows of a global batch each process reads under Grain's `ShardByJaxProcess` and a Datarax `Pipeline(..., sharding=...)`, that a Datarax global batch and its per-record draws do not depend on the process count, and a training step fed straight from the pipeline |
| [`04_resumed_training_guide.py`](04_resumed_training_guide.py) | Model, optimizer and loader state saved together through one `substrax.checkpoint.OrbaxCheckpointStore` at a mid-epoch step, restored into fresh objects, and both resumed runs reproducing the uninterrupted runs loss for loss, with `calibrax.metrics.functional.mse` as the loss |

Every tutorial is a Jupytext script with a paired notebook, follows the same structure as
the other numbered examples, and is checked by `scripts/validate_examples.py`,
`scripts/check_sync.py` and `tests/examples`.

## Running

```bash
source ./activate.sh
python examples/comparison/01_grain_datarax_quickref.py
python examples/comparison/02_randomness_and_learnable_operators_tutorial.py
python examples/comparison/03_sharding_guide.py
python examples/comparison/04_resumed_training_guide.py
```

Each script prints what it measures and exits non-zero if a claim it makes does not hold
on the run: a resumed loader diverging from the uninterrupted one, a record's randomness
not following the rule the tutorial states, or a learnable stage missing its target.

## Measured numbers

Throughput and memory for both libraries are on the
[framework comparison page](../../docs/benchmarks/comparison.md). The tutorials make no
speed claim.
