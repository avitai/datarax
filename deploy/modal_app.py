"""Modal launcher for running Datarax examples on a GPU.

Examples are real use cases, so their published output, the terminal text a docs page quotes
and the figures it embeds, comes from a run on an accelerator rather than from this
machine's CPU. This launcher builds the project environment from ``uv.lock`` with the
``cuda12``, ``data``, ``tfds`` and ``examples`` extras, prepares the example datasets once
into a persistent volume, runs the requested example scripts on the chosen GPU, and brings
each script's log and everything it wrote under ``AVITAI_OUTPUT_DIR`` back to a local
directory.

Usage::

    modal run deploy/modal_app.py --task probe
    modal run deploy/modal_app.py --task examples --paths "examples/core/01_simple_pipeline.py"
    modal run deploy/modal_app.py --task examples --gpu A100-80GB --out temp/modal_examples
    modal run --detach deploy/modal_app.py --task examples --run nightly --paths "..."
    modal run deploy/modal_app.py --task fetch --run nightly

Logs and outputs are written to the ``datarax-example-outputs`` volume under the run's name
and committed after every script, so a detached run, or one whose client lost its
connection, is fetched afterwards with ``--task fetch``.

Requires a Modal account (``modal setup`` for browser auth). Modal is a host-side launcher
only; it is deliberately not a Datarax dependency.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import modal


LOGGER = logging.getLogger("datarax.deploy")

APP_NAME = "datarax-examples"
REPO_PATH = "/root/datarax"
DATASET_VOLUME = "datarax-example-datasets"
DATASET_PATH = "/root/datasets"
OUTPUT_VOLUME = "datarax-example-outputs"
OUTPUT_PATH = "/root/outputs"

DEFAULT_GPU = "L40S"

# The example scripts whose docs pages quote a GPU run, in the order they are run.
DEFAULT_EXAMPLES = (
    "examples/comparison/01_grain_datarax_quickref.py",
    "examples/comparison/02_randomness_and_learnable_operators_tutorial.py",
    "examples/comparison/03_sharding_guide.py",
    "examples/comparison/04_resumed_training_guide.py",
)

# Excludes for the build-time repo copy: local virtualenvs and caches, generated
# documentation, private notes, and every ``.env*`` file. activate.sh sources
# .datarax.env, which names a local virtualenv that does not exist inside the image.
_IMAGE_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/.env*",
    "**/.datarax.env",
    "**/node_modules",
    "**/site",
    "**/htmlcov",
    "**/memory-bank",
    "**/temp",
    "**/outputs",
    "**/checkpoints",
    "**/benchmark_results",
    "**/test_results",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.mypy_cache",
]

app = modal.App(APP_NAME)
datasets = modal.Volume.from_name(DATASET_VOLUME, create_if_missing=True)
# Logs and outputs go to a volume, committed after every script, so a run's results survive
# the client that launched it: a long run can be detached and fetched later, and a client that
# loses its connection loses nothing.
outputs = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

# Pin uv to the version that wrote uv.lock, for build stability. `--frozen` installs the
# exact pinned versions without re-resolving; `--locked` re-checks consistency, which fails
# spuriously on Modal because its managed Python differs from the local interpreter.
_UV_VERSION = "0.11.25"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(f"uv=={_UV_VERSION}")
    .add_local_dir(".", REPO_PATH, copy=True, ignore=_IMAGE_IGNORE)
    .run_commands(
        f"cd {REPO_PATH} && uv sync --frozen --no-dev"
        " --extra cuda12 --extra data --extra tfds --extra examples"
    )
    .workdir(REPO_PATH)
)

# Deterministic GPU kernels so a published number is reproducible from run to run; no
# preallocation so a short example does not grab the whole card. Datasets live on the
# volume so a second run does not download them again.
_RUN_ENV = {
    "JAX_PLATFORMS": "cuda",
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_CLIENT_MEM_FRACTION": "0.9",
    "TF_CPP_MIN_LOG_LEVEL": "1",
    "TFDS_DATA_DIR": f"{DATASET_PATH}/tensorflow_datasets",
    "KERAS_HOME": f"{DATASET_PATH}/keras",
}


def _run(argv: list[str], log: Path | None = None, output_dir: str | None = None) -> int:
    """Run a command in the synced project environment, tee-ing its output to ``log``."""
    env = {**os.environ, **_RUN_ENV}
    if output_dir is not None:
        env["AVITAI_OUTPUT_DIR"] = output_dir
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH inside the image")
    completed = subprocess.run(  # nosec B603
        [uv, "run", "--no-sync", *argv],
        cwd=REPO_PATH,
        env=env,
        check=False,
        capture_output=log is not None,
        text=True,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout + completed.stderr)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
    return completed.returncode


@app.function(image=image, gpu=DEFAULT_GPU, timeout=15 * 60)
def probe() -> None:
    """Report the JAX backend and devices this GPU exposes; confirms the image builds."""
    _run(
        [
            "python",
            "-c",
            "import jax; print(jax.__version__, jax.default_backend(), jax.devices())",
        ]
    )


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=6 * 60 * 60,
    volumes={DATASET_PATH: datasets, OUTPUT_PATH: outputs},
)
def examples(run: str, paths: list[str], prepare_datasets: bool) -> str:
    """Run example scripts on the GPU, writing their logs and outputs to the output volume.

    Args:
        run: Name of this run; everything lands under ``<run>/`` in the output volume.
        paths: Example scripts, relative to the repository root, run in order.
        prepare_datasets: Whether to run ``scripts/prepare_example_datasets.py`` first, which
            fetches the slow-host archives into the dataset volume once.

    Returns:
        The summary: one tab-separated ``<exit code> <script>`` line per script. The volume holds
        ``<run>/logs/<script stem>.log`` per script, ``<run>/logs/summary.txt``, and whatever
        the scripts wrote under ``AVITAI_OUTPUT_DIR`` (``<run>/examples/...``).
    """
    output = Path(OUTPUT_PATH) / run
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if prepare_datasets:
        _run(["python", "scripts/prepare_example_datasets.py"], log=logs / "prepare_datasets.log")
        datasets.commit()
    lines = []
    for path in paths:
        code = _run(
            ["python", "-u", path], log=logs / f"{Path(path).stem}.log", output_dir=str(output)
        )
        lines.append(f"{code}\t{path}")
        (logs / "summary.txt").write_text("\n".join(lines) + "\n")
        outputs.commit()
    return "\n".join(lines) + "\n"


def fetch(run: str, destination: Path) -> int:
    """Download everything a run wrote to the output volume into ``destination``."""
    count = 0
    for entry in outputs.listdir(run, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = destination / entry.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in outputs.read_file(entry.path):
                handle.write(chunk)
        count += 1
    return count


@app.local_entrypoint()
def main(
    task: str = "probe",
    gpu: str = DEFAULT_GPU,
    paths: str = "",
    run: str = "",
    out: str = "temp/modal_examples",
    prepare_datasets: bool = False,
) -> None:
    """Dispatch a task to Modal.

    Args:
        task: ``"probe"``, ``"examples"`` (run, then fetch) or ``"fetch"`` (download a run
            that was launched with ``--detach`` or whose client lost its connection).
        gpu: Modal GPU spec, for example ``"L40S"``, ``"A100-80GB"`` or ``"H100"``.
        paths: Example scripts to run, space-separated; the comparison tutorials by default.
        run: The run's name in the output volume; a UTC timestamp when empty.
        out: Local directory that receives ``<run>/``; ``temp/`` is gitignored.
        prepare_datasets: Fetch the slow-host datasets into the volume before the examples.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if task == "probe":
        probe.with_options(gpu=gpu).remote()
        return
    if task not in ("examples", "fetch"):
        msg = f"Unknown task {task!r}; expected 'probe', 'examples' or 'fetch'"
        raise ValueError(msg)
    name = run or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if task == "examples":
        scripts = paths.split() if paths else list(DEFAULT_EXAMPLES)
        LOGGER.info("run %s: %d scripts", name, len(scripts))
        summary = examples.with_options(gpu=gpu).remote(name, scripts, prepare_datasets)
        LOGGER.info("%s", summary)
    count = fetch(name, Path(out))
    LOGGER.info("%d files of run %s written under %s", count, name, Path(out) / name)
