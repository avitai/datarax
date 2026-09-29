"""Operators keep their data's dtype, as Flax NNX layers do, with x64 on or off.

An operator without parameters returns each floating field in its input's dtype, as
``nnx.Dropout`` does: the values it draws (in JAX's default dtype) and the constants it holds are
applied in the data's dtype. An operator holding ``nnx.Param`` is a Flax layer: its parameters
are created in ``param_dtype`` (float32 by default) and its arithmetic runs in the promotion of
its inputs and parameters (``flax.nnx.nn.dtypes.promote_dtype``), so a bfloat16 input gives
float32 and a float64 input float64. Nothing raises in either x64 mode: x64 is a program-wide
setting (jax ``docs/101/default_dtypes.md``), float64 is what scientific users run, and TPUs run
float32 and bfloat16. Checked over every operator kind the per-record fixture records, plus CREPE,
eagerly and under ``nnx.jit``, where a repeated call must compile nothing.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

import datarax
from datarax.core import batch_ops
from datarax.core.operator import OperatorModule
from datarax.operators.modality.audio.f0_operator import CrepeF0Config, CrepeF0Operator
from datarax.operators.modality.image import functional
from datarax.pipeline.dag import name_records
from tests.scripts.script_loader import load_script


generator = load_script("generate_per_record_outputs")

# One second of a 440 Hz tone for each record the fixture names.
_AUDIO = np.tile(
    0.5 * np.sin(2 * np.pi * 440.0 * np.arange(16000) / 16000.0), (len(generator.INDICES), 1)
)

CASES = [
    *generator.CASES,
    (
        "crepe f0",
        lambda: CrepeF0Operator(CrepeF0Config(capacity="tiny"), rngs=nnx.Rngs(0)),
        {"audio": _AUDIO},
    ),
]

DTYPES = {
    "float32": (jnp.float32, False),
    "float64 under x64": (jnp.float64, True),
    "float32 under x64": (jnp.float32, True),
    "bfloat16": (jnp.bfloat16, False),
    "bfloat16 under x64": (jnp.bfloat16, True),
}


def _floats_as(data: dict[str, Any], dtype: Any) -> dict[str, Any]:
    """``data`` with every floating array cast to ``dtype`` and every other array as it is."""
    return jax.tree.map(
        lambda x: jnp.asarray(x, dtype) if np.issubdtype(np.asarray(x).dtype, np.floating) else x,
        data,
    )


def _holds_parameters(operator: nnx.Module) -> bool:
    return any(isinstance(node, nnx.Param) for _, node in nnx.iter_graph(operator))


@nnx.jit
def _jitted(operator: OperatorModule, batch: Any) -> Any:
    return operator(batch)


def _call(operator: OperatorModule, batch: Any, transform: str) -> Any:
    """``operator(batch)`` eagerly, or under ``nnx.jit``; a repeated jitted call compiles nothing.

    A value whose dtype depends on something other than the input's (a weak Python scalar, a
    default-dtype draw) changes the jitted signature and retraces; the repeated call catches it.
    """
    if transform == "eager":
        return operator(batch)
    out = _jitted(operator, batch)
    with expect_compiles(0):
        _jitted(operator, batch)
    return out


@pytest.mark.parametrize("transform", ["eager", "nnx.jit"])
@pytest.mark.parametrize("dtype_name", list(DTYPES))
@pytest.mark.parametrize(("name", "build", "data"), CASES, ids=[case[0] for case in CASES])
def test_an_operator_returns_its_data_in_the_data_dtype(
    name: str, build: Any, data: Any, dtype_name: str, transform: str
) -> None:
    """Without parameters: the data's dtype. With parameters: that, or its Flax promotion."""
    dtype, x64 = DTYPES[dtype_name]
    with jax.enable_x64(x64):
        operator = build()
        batch = name_records(
            batch_ops.from_arrays(_floats_as(data, dtype)), generator.INDICES, generator.EPOCH
        )
        out = _call(operator, batch, transform).data
        promoted = jnp.promote_types(dtype, jnp.float32)
    allowed = {jnp.dtype(dtype)} | ({jnp.dtype(promoted)} if _holds_parameters(operator) else set())

    floating = [
        (jax.tree_util.keystr(path), leaf.dtype)
        for path, leaf in jax.tree_util.tree_flatten_with_path(out)[0]
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    assert floating, f"{name} returned no floating field"
    wrong = [(path, str(leaf_dtype)) for path, leaf_dtype in floating if leaf_dtype not in allowed]
    assert wrong == [], f"{name} with {dtype_name}: {wrong}, allowed {sorted(map(str, allowed))}"


JITTERS = {
    "every adjustment": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1},
    "saturation alone": {"saturation": 0.2},
    "hue alone": {"hue": 0.1},
}


@pytest.mark.parametrize("transform", ["eager", "jax.jit"])
@pytest.mark.parametrize("keyed", [False, True], ids=["fixed factors", "drawn factors"])
@pytest.mark.parametrize("jitter", list(JITTERS))
@pytest.mark.parametrize("dtype_name", list(DTYPES))
def test_color_jitter_returns_the_image_dtype(
    dtype_name: str, jitter: str, keyed: bool, transform: str
) -> None:
    """``functional.color_jitter``, public and built by no operator class, keeps the image's dtype.

    Its drawn factors take JAX's default dtype and its fixed ones are Python numbers; both are
    applied in the image's dtype, on every path (saturation and hue together or alone).
    """
    dtype, x64 = DTYPES[dtype_name]

    def apply(image: jax.Array, key: jax.Array | None) -> jax.Array:
        return functional.color_jitter(image, **JITTERS[jitter], key=key)

    with jax.enable_x64(x64):
        image = jnp.asarray(np.random.default_rng(0).random((8, 8, 3)), dtype)
        key = jax.random.key(0) if keyed else None
        if transform == "eager":
            out = apply(image, key)
        else:
            jitted = jax.jit(apply)
            out = jitted(image, key)
            with expect_compiles(0):
                jitted(image, key)

    assert out.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("x64", [False, True], ids=["x64 off", "x64 on"])
def test_a_shuffled_pipeline_runs_and_keeps_its_dtypes(x64: bool) -> None:
    """The epoch order and a stochastic stage run in either mode; the stage keeps its dtype."""
    with jax.enable_x64(x64):
        entries = generator.pipeline_entries()

    values = [entry for key, entry in entries.items() if key.endswith("['value']")]
    assert values
    assert all(value.dtype == np.float32 for value in values)


def _concrete_operator_classes() -> set[type]:
    for module in pkgutil.walk_packages(datarax.__path__, "datarax."):
        importlib.import_module(module.name)

    def subclasses(cls: type) -> set[type]:
        return {sub for child in cls.__subclasses__() for sub in (child, *subclasses(child))}

    return {
        cls
        for cls in subclasses(OperatorModule)
        if cls.__module__.startswith("datarax") and not inspect.isabstract(cls)
    }


def test_the_contract_covers_every_operator_class() -> None:
    """Every concrete operator class in the package is built by a case, itself or by a subclass.

    A class new to the package fails here until a case builds it, so the contract cannot
    silently shrink to the operators it happened to list.
    """
    built = {
        type(node)
        for _, build, _ in CASES
        for _, node in nnx.iter_graph(build())
        if isinstance(node, OperatorModule)
    }

    uncovered = {
        cls.__qualname__
        for cls in _concrete_operator_classes()
        if not any(issubclass(covered, cls) for covered in built)
    }
    assert uncovered == set()


REPOSITORY = Path(__file__).resolve().parents[2]


def test_the_package_creates_no_jax_array_at_import() -> None:
    """A module-level JAX array takes the default dtype of the mode the process started in.

    Under x64 from the start it is float64 and promotes float32 data; ``jax.enable_x64`` later
    cannot change it, so the contract above, which switches x64 on after import, would not see
    it. ``crepe_model.CENTS_MAPPING`` was one: CREPE returned float64 pitch for float32 audio.
    """
    arrays = []
    for path in sorted((REPOSITORY / "src").rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
                for call in ast.walk(node.value):
                    owner = call.func if isinstance(call, ast.Call) else None
                    while isinstance(owner, ast.Attribute):
                        owner = owner.value
                    if isinstance(owner, ast.Name) and owner.id in {"jax", "jnp"}:
                        arrays.append(f"{path.relative_to(REPOSITORY)}:{node.lineno}")
                        break

    assert arrays == []


@pytest.mark.slow
def test_the_contract_holds_with_x64_on_from_the_start() -> None:
    """The contract, in a process that starts with x64 on, as scientific users run it."""
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "--no-cov",
            "-k",
            "not from_the_start and not creates_no_jax_array",
        ],
        cwd=REPOSITORY,
        env={**os.environ, "JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stdout[-4000:]
