"""Record the per-record outputs of every operator kind, as a fixture to compare against.

The redesign of the operator framework changes how a record's PRNG key reaches ``apply`` and how
``_vmap_apply`` vectorizes. Neither is meant to change what an operator produces. This script
records what every operator kind produces today, so that a later commit that alters the machinery
has to show the outputs unchanged, entry by entry, rather than assert it.

Every entry calls ``_apply_on_raw(data, {}, None, INDICES, EPOCH)`` with fixed record indices and a
fixed epoch, which is the path the pipeline's fused chain takes, and the shipped operators reach
through their own overrides where they have one. A second group runs two epochs of a shuffled
pipeline, so the fixture also pins what shuffling and the epoch counter feed the stages.

Entries named in ``INTENDED_CHANGE`` are expected to differ after the record key reaches stochastic
children: each is a case where a ``probability=1.0`` wrapper is classified deterministic
(``ProbabilisticOperatorConfig`` sets ``stochastic`` from ``0 < probability < 1``) and therefore
passes no random parameters to a stochastic child. They are recorded so the change is visible and
deliberate, not so it is forbidden.

``CrepeF0Operator`` is deliberately absent: it needs pretrained weights that the test environment
does not carry, which would make this fixture depend on a download. ``tests/fixtures/crepe/``
covers that model.

Run it against a specific revision by pointing PYTHONPATH at that revision's ``src``, which the
script verifies before recording anything:

    git worktree add --detach /tmp/wt-baseline <commit>
    PYTHONPATH=/tmp/wt-baseline/src python scripts/generate_per_record_outputs.py \
        --expect-root /tmp/wt-baseline --out /tmp/baseline.npz
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import datarax
from datarax.core.config import (
    BatchMixOperatorConfig,
    ElementOperatorConfig,
    MapOperatorConfig,
)
from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig
from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators.batch_mix_operator import BatchMixOperator
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.element_operator import ElementOperator
from datarax.operators.map_operator import MapOperator
from datarax.operators.modality.audio.loudness_operator import LoudnessConfig, LoudnessOperator
from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)
from datarax.operators.modality.image.contrast_operator import (
    ContrastOperator,
    ContrastOperatorConfig,
)
from datarax.operators.modality.image.dropout_operator import DropoutOperator, DropoutOperatorConfig
from datarax.operators.modality.image.noise_operator import NoiseOperator, NoiseOperatorConfig
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)
from datarax.operators.modality.image.rotation_operator import (
    RotationOperator,
    RotationOperatorConfig,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig
from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.utils.external import ExternalAdapterConfig, ExternalLibraryAdapter, PureJaxAdapter


BATCH = 4
INDICES = jnp.arange(BATCH, dtype=jnp.int32) + 10
EPOCH = jnp.int32(2)
IMAGE = jnp.linspace(0.1, 0.9, BATCH * 16 * 16 * 3, dtype=jnp.float32).reshape(BATCH, 16, 16, 3)
IMAGE_LABEL: dict[str, jax.Array] = {"image": IMAGE, "label": jnp.arange(BATCH, dtype=jnp.int32)}
IMAGE_ONLY: dict[str, jax.Array] = {"image": IMAGE}
# Tones at real frequencies plus broadband noise, so every FFT bin carries energy. This is not
# decoration: a tone alone leaves most bins at the loudness operator's 1e-20 power floor, where
# log10 turns one input ULP into 0.44 dB (measured 3.7e6x amplification, against 32x here) and
# float32 differs from a float64 reference by 8 dB. Such a value cannot be recorded and compared
# on another machine, because it is not a meaningful number on this one either.
_AUDIO_TIME = jnp.arange(4096, dtype=jnp.float32) / 16000.0
_AUDIO_TONES = jnp.stack(
    [
        0.5 * jnp.sin(2 * jnp.pi * 220.0 * (index + 1) * _AUDIO_TIME)
        + 0.3 * jnp.sin(2 * jnp.pi * 437.0 * (index + 1) * _AUDIO_TIME)
        for index in range(BATCH)
    ]
)
_AUDIO_NOISE = jnp.asarray(
    np.random.default_rng(0).standard_normal((BATCH, 4096)).astype(np.float32) * 0.1
)
AUDIO: dict[str, jax.Array] = {"audio": _AUDIO_TONES + _AUDIO_NOISE}

PIPELINE_RECORDS = 16
PIPELINE_BATCH = 4

INTENDED_CHANGE = (
    "probabilistic p=1 over a stochastic child",
    "probabilistic p=1 nested in another p=1 wrapper",
    "composite whose only stochastic descendant is behind a p=1 wrapper",
    "deterministic external adapter inside a stochastic composite",
    "operator built after a p=1 wrapper from one Rngs",
)


def rngs() -> nnx.Rngs:
    """Return the RNG state every entry builds its operator from."""
    return nnx.Rngs(0, augment=1, batch_mix=2, composite=3)


def stochastic_kwargs(stochastic: bool) -> dict[str, Any]:
    """Return the config fields that put an operator in stochastic mode."""
    return {"stochastic": True, "stream_name": "augment"} if stochastic else {}


def brightness(stochastic: bool) -> BrightnessOperator:
    """Return a brightness operator in the requested mode."""
    if stochastic:
        config = BrightnessOperatorConfig(field_key="image", **stochastic_kwargs(True))
    else:
        config = BrightnessOperatorConfig(field_key="image", brightness_delta=0.1)
    return BrightnessOperator(config, rngs=rngs())


def noise(
    stochastic: bool, mode: Literal["gaussian", "salt_pepper", "poisson"] = "gaussian"
) -> NoiseOperator:
    """Return a noise operator in the requested mode."""
    extra = stochastic_kwargs(stochastic)
    return NoiseOperator(NoiseOperatorConfig(field_key="image", mode=mode, **extra), rngs=rngs())


class ShiftToTarget(ModalityOperator):
    """A modality operator that writes its result to ``target_key``."""

    def apply(
        self,
        data: Any,
        state: Any,
        metadata: Any,
        random_params: Any = None,
        stats: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Shift the field by a drawn amount, or by a fixed amount when deterministic."""
        del stats
        image = self._extract_field(data, self.config.field_key)
        drawn = self.config.stochastic and random_params is not None
        shift = random_params["shift"] if drawn else 0.05
        return self._remap_field(data, image + shift), state, metadata

    def generate_random_params(self, element_keys: Any, data_shapes: Any) -> Any:
        """Draw one shift per record."""
        del data_shapes
        return {"shift": jax.vmap(lambda key: jax.random.uniform(key, ()))(element_keys)}


class Fuse(CrossModalOperator):
    """A cross-modal operator that writes one new field from two inputs."""

    def apply(
        self,
        data: Any,
        state: Any,
        metadata: Any,
        random_params: Any = None,
        stats: Any = None,
    ) -> tuple[Any, Any, Any]:
        """Combine the two input fields into the declared output field."""
        del random_params, stats
        image, label = self._extract_inputs(data)
        fused = jnp.mean(image) + label.astype(jnp.float32)
        return self._store_outputs(data, [fused]), state, metadata


def element_noise(element: Any, key: jax.Array) -> Any:
    """Add drawn noise to an element's image."""
    image = element.data["image"]
    return element.update_data({"image": image + 0.1 * jax.random.normal(key, image.shape)})


def external_noise(data: dict[str, Any], key: jax.Array) -> dict[str, Any]:
    """Add drawn noise to a raw data dict's image."""
    return {**data, "image": data["image"] + 0.1 * jax.random.normal(key, data["image"].shape)}


def composite(strategy: CompositionStrategy, **extra: Any) -> CompositeOperatorModule:
    """Return a composite over one deterministic and one stochastic child."""
    children = [brightness(False), noise(True)]
    return CompositeOperatorModule(
        CompositeOperatorConfig(strategy=strategy, operators=children, **extra), rngs=rngs()
    )


def bright_mean(data: dict[str, Any]) -> jax.Array:
    """Whether the image's mean exceeds the midpoint."""
    return jnp.mean(data["image"]) > 0.5


def always_on(operator: OperatorModule) -> ProbabilisticOperator:
    """Wrap an operator in a wrapper that always applies it."""
    return ProbabilisticOperator(
        ProbabilisticOperatorConfig(operator=operator, probability=1.0), rngs=rngs()
    )


def shared_rngs_pair(index: int) -> OperatorModule:
    """Return one of two stochastic operators built from a single Rngs, in build order."""
    shared = rngs()
    first = BrightnessOperator(
        BrightnessOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=shared
    )
    second = NoiseOperator(
        NoiseOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=shared
    )
    return (first, second)[index]


def operator_after_wrapper() -> OperatorModule:
    """Return an operator built from a shared Rngs after a probability=1.0 wrapper."""
    shared = rngs()
    always_on(
        BrightnessOperator(
            BrightnessOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=shared
        )
    )
    return NoiseOperator(
        NoiseOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=shared
    )


CASES: list[tuple[str, Callable[[], OperatorModule], dict[str, jax.Array]]] = [
    ("brightness deterministic", lambda: brightness(False), IMAGE_LABEL),
    ("brightness stochastic", lambda: brightness(True), IMAGE_LABEL),
    (
        "contrast deterministic",
        lambda: ContrastOperator(
            ContrastOperatorConfig(field_key="image", contrast_factor=1.2), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "contrast stochastic",
        lambda: ContrastOperator(
            ContrastOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "dropout pixel deterministic",
        lambda: DropoutOperator(
            DropoutOperatorConfig(field_key="image", dropout_rate=0.3), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "dropout pixel stochastic",
        lambda: DropoutOperator(
            DropoutOperatorConfig(field_key="image", dropout_rate=0.3, **stochastic_kwargs(True)),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "dropout channel stochastic",
        lambda: DropoutOperator(
            DropoutOperatorConfig(
                field_key="image", dropout_rate=0.3, mode="channel", **stochastic_kwargs(True)
            ),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    ("noise gaussian deterministic", lambda: noise(False), IMAGE_LABEL),
    ("noise gaussian stochastic", lambda: noise(True), IMAGE_LABEL),
    ("noise salt_pepper stochastic", lambda: noise(True, "salt_pepper"), IMAGE_LABEL),
    ("noise poisson stochastic", lambda: noise(True, "poisson"), IMAGE_LABEL),
    (
        "patch dropout deterministic",
        lambda: PatchDropoutOperator(
            PatchDropoutOperatorConfig(field_key="image", num_patches=2, patch_size=(4, 4)),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "patch dropout stochastic",
        lambda: PatchDropoutOperator(
            PatchDropoutOperatorConfig(
                field_key="image", num_patches=2, patch_size=(4, 4), **stochastic_kwargs(True)
            ),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "rotation deterministic",
        lambda: RotationOperator(
            RotationOperatorConfig(field_key="image", angle_range=(10.0, 10.0))
        ),
        IMAGE_LABEL,
    ),
    (
        "rotation stochastic",
        lambda: RotationOperator(
            RotationOperatorConfig(field_key="image", **stochastic_kwargs(True)), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "map full-tree deterministic",
        lambda: MapOperator(MapOperatorConfig(), fn=lambda x, key: x * 2),
        IMAGE_ONLY,
    ),
    (
        "map full-tree stochastic",
        lambda: MapOperator(
            MapOperatorConfig(**stochastic_kwargs(True)),
            fn=lambda x, key: x + 0.1 * jax.random.normal(key, x.shape),
            rngs=rngs(),
        ),
        IMAGE_ONLY,
    ),
    (
        "map subtree stochastic",
        lambda: MapOperator(
            MapOperatorConfig(subtree={"image": None}, **stochastic_kwargs(True)),
            fn=lambda x, key: x + 0.1 * jax.random.normal(key, x.shape),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "element deterministic",
        lambda: ElementOperator(ElementOperatorConfig(), fn=element_noise),
        IMAGE_LABEL,
    ),
    (
        "element stochastic",
        lambda: ElementOperator(
            ElementOperatorConfig(**stochastic_kwargs(True)), fn=element_noise, rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "batch mix mixup",
        lambda: BatchMixOperator(BatchMixOperatorConfig(), rngs=rngs()),
        IMAGE_LABEL,
    ),
    (
        "batch mix cutmix",
        lambda: BatchMixOperator(BatchMixOperatorConfig(mode="cutmix"), rngs=rngs()),
        IMAGE_LABEL,
    ),
    (
        "external adapter stochastic",
        lambda: ExternalLibraryAdapter(ExternalAdapterConfig(), external_noise, rngs=rngs()),
        IMAGE_LABEL,
    ),
    (
        "external adapter deterministic",
        lambda: ExternalLibraryAdapter(
            ExternalAdapterConfig(stochastic=False, stream_name=None), external_noise
        ),
        IMAGE_LABEL,
    ),
    (
        "pure jax adapter",
        lambda: PureJaxAdapter(
            ExternalAdapterConfig(stochastic=False, stream_name=None),
            lambda d: {**d, "image": d["image"] * 2},
        ),
        IMAGE_LABEL,
    ),
    (
        "selector over two stochastic children",
        lambda: SelectorOperator(
            SelectorOperatorConfig(operators=[brightness(True), noise(True)]), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "probabilistic p=0.5",
        lambda: ProbabilisticOperator(
            ProbabilisticOperatorConfig(operator=noise(True), probability=0.5), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "probabilistic p=0",
        lambda: ProbabilisticOperator(
            ProbabilisticOperatorConfig(operator=brightness(False), probability=0.0), rngs=rngs()
        ),
        IMAGE_LABEL,
    ),
    (
        "probabilistic p=1 over a deterministic child",
        lambda: always_on(brightness(False)),
        IMAGE_LABEL,
    ),
    ("probabilistic p=1 over a stochastic child", lambda: always_on(noise(True)), IMAGE_LABEL),
    (
        "probabilistic p=1 nested in another p=1 wrapper",
        lambda: always_on(always_on(noise(True))),
        IMAGE_LABEL,
    ),
    (
        "composite whose only stochastic descendant is behind a p=1 wrapper",
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[brightness(False), always_on(noise(True))],
            ),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "deterministic external adapter inside a stochastic composite",
        lambda: CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[
                    ExternalLibraryAdapter(
                        ExternalAdapterConfig(stochastic=False, stream_name=None), external_noise
                    ),
                    noise(True),
                ],
            ),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    ("first of two stochastic operators from one Rngs", lambda: shared_rngs_pair(0), IMAGE_LABEL),
    ("second of two stochastic operators from one Rngs", lambda: shared_rngs_pair(1), IMAGE_LABEL),
    ("operator built after a p=1 wrapper from one Rngs", operator_after_wrapper, IMAGE_LABEL),
    ("composite SEQUENTIAL", lambda: composite(CompositionStrategy.SEQUENTIAL), IMAGE_LABEL),
    (
        "composite DYNAMIC_SEQUENTIAL",
        lambda: composite(CompositionStrategy.DYNAMIC_SEQUENTIAL),
        IMAGE_LABEL,
    ),
    (
        "composite CONDITIONAL_SEQUENTIAL",
        lambda: composite(
            CompositionStrategy.CONDITIONAL_SEQUENTIAL, conditions=[bright_mean, bright_mean]
        ),
        IMAGE_LABEL,
    ),
    (
        "composite PARALLEL mean",
        lambda: composite(CompositionStrategy.PARALLEL, merge_strategy="mean"),
        IMAGE_ONLY,
    ),
    (
        "composite PARALLEL dict",
        lambda: composite(CompositionStrategy.PARALLEL, merge_strategy="dict"),
        IMAGE_LABEL,
    ),
    (
        "composite WEIGHTED_PARALLEL static",
        lambda: composite(CompositionStrategy.WEIGHTED_PARALLEL, weights=[0.7, 0.3]),
        IMAGE_LABEL,
    ),
    (
        "composite WEIGHTED_PARALLEL learnable",
        lambda: composite(
            CompositionStrategy.WEIGHTED_PARALLEL, weights=[0.7, 0.3], learnable_weights=True
        ),
        IMAGE_LABEL,
    ),
    (
        "composite WEIGHTED_PARALLEL weight_key",
        lambda: composite(CompositionStrategy.WEIGHTED_PARALLEL, weight_key="op_weights"),
        {**IMAGE_LABEL, "op_weights": jnp.full((BATCH, 2), 0.5, dtype=jnp.float32)},
    ),
    (
        "composite CONDITIONAL_PARALLEL sum",
        lambda: composite(
            CompositionStrategy.CONDITIONAL_PARALLEL,
            conditions=[bright_mean, bright_mean],
            merge_strategy="sum",
        ),
        IMAGE_ONLY,
    ),
    ("composite ENSEMBLE_MEAN", lambda: composite(CompositionStrategy.ENSEMBLE_MEAN), IMAGE_ONLY),
    ("composite ENSEMBLE_SUM", lambda: composite(CompositionStrategy.ENSEMBLE_SUM), IMAGE_ONLY),
    ("composite ENSEMBLE_MAX", lambda: composite(CompositionStrategy.ENSEMBLE_MAX), IMAGE_ONLY),
    ("composite ENSEMBLE_MIN", lambda: composite(CompositionStrategy.ENSEMBLE_MIN), IMAGE_ONLY),
    (
        "composite BRANCHING",
        lambda: composite(
            CompositionStrategy.BRANCHING, router=lambda d: bright_mean(d).astype(jnp.int32)
        ),
        IMAGE_LABEL,
    ),
    (
        "modality subclass writing target_key, deterministic",
        lambda: ShiftToTarget(ModalityOperatorConfig(field_key="image", target_key="shifted")),
        IMAGE_LABEL,
    ),
    (
        "modality subclass writing target_key, stochastic",
        lambda: ShiftToTarget(
            ModalityOperatorConfig(
                field_key="image", target_key="shifted", **stochastic_kwargs(True)
            ),
            rngs=rngs(),
        ),
        IMAGE_LABEL,
    ),
    (
        "cross-modal subclass adding a field",
        lambda: Fuse(
            CrossModalOperatorConfig(input_fields=["image", "label"], output_fields=["fused"])
        ),
        IMAGE_LABEL,
    ),
    ("loudness", lambda: LoudnessOperator(LoudnessConfig(n_fft=512)), AUDIO),
]


def entries_for(name: str, outputs: Any) -> dict[str, np.ndarray]:
    """Return one array per leaf of ``outputs``, keyed by the case name and the leaf's path."""
    leaves = jax.tree_util.tree_flatten_with_path(outputs)[0]
    return {f"{name}{jax.tree_util.keystr(path)}": np.asarray(leaf) for path, leaf in leaves}


def operator_entries() -> dict[str, np.ndarray]:
    """Return the recorded outputs of every operator case."""
    recorded: dict[str, np.ndarray] = {}
    for name, build, data in CASES:
        operator = build()
        out_data, _ = operator._apply_on_raw(data, {}, None, INDICES, EPOCH)  # noqa: SLF001
        recorded.update(entries_for(name, out_data))
        print(f"  recorded {name}")
    return recorded


def pipeline_entries() -> dict[str, np.ndarray]:
    """Return two epochs of a shuffled pipeline, batch by batch."""
    values = jnp.linspace(0.0, 1.0, PIPELINE_RECORDS * 8, dtype=jnp.float32).reshape(
        PIPELINE_RECORDS, 8
    )
    source = MemorySource(
        MemorySourceConfig(shuffle=True),
        data={"value": np.asarray(values), "id": np.arange(PIPELINE_RECORDS)},
        rngs=nnx.Rngs(0, shuffle=1),
    )
    stage = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="jitter"),
        fn=lambda element, key: element.update_data(
            {
                "value": element.data["value"]
                + 0.1 * jax.random.normal(key, element.data["value"].shape)
            }
        ),
        rngs=nnx.Rngs(0, jitter=2),
    )
    pipeline = Pipeline(source=source, stages=[stage], batch_size=PIPELINE_BATCH, rngs=nnx.Rngs(0))

    recorded: dict[str, np.ndarray] = {}
    for epoch in range(2):
        for index, batch in enumerate(iter(pipeline)):
            recorded.update(entries_for(f"pipeline epoch {epoch} batch {index}", batch))
        pipeline.reset()
        print(f"  recorded pipeline epoch {epoch}")
    return recorded


def main() -> None:
    """Record every entry and write the fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="path of the .npz to write")
    parser.add_argument(
        "--expect-root",
        type=pathlib.Path,
        required=True,
        help="the tree datarax must be imported from, so a revision is recorded on purpose",
    )
    arguments = parser.parse_args()

    imported = pathlib.Path(datarax.__file__).resolve()
    expected = arguments.expect_root.resolve()
    if not imported.is_relative_to(expected):
        raise SystemExit(f"datarax imported from {imported}, which is not under {expected}")
    print(f"datarax imported from {imported}")

    recorded = operator_entries()
    recorded.update(pipeline_entries())
    recorded["case names"] = np.array([name for name, _, _ in CASES])
    recorded["intended change"] = np.array(INTENDED_CHANGE)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    # The archive writer declares a named `allow_pickle` parameter, so a type checker reads
    # arbitrary entry names as that argument. The names are the fixture's keys and carry
    # spaces and brackets, so they cannot be passed any other way.
    np.savez_compressed(arguments.out, **recorded)  # type: ignore[reportArgumentType]
    size_kb = arguments.out.stat().st_size / 1024
    print(f"wrote {len(recorded)} entries to {arguments.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
