"""Datarax adapter -- reference implementation for the benchmark framework.

Wraps Datarax's public API (from_source, MemorySource, DAGExecutor)
using the BenchmarkAdapter lifecycle: setup -> warmup -> iterate -> teardown.

Supports all 25 scenarios with transform chaining via
OperatorNode and topology dispatch for DAG-based scenarios. External
backends (TFDS, HuggingFace) are created via config.extra["backend"].

Design ref: Section 7.2 of the benchmark report.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from benchmarks.adapters import register
from benchmarks.adapters._utils import cast_to_float32, normalize_uint8
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig
from datarax import OperatorNode, from_source
from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Element
from datarax.operators.element_operator import ElementOperator
from datarax.sources import MemorySource, MemorySourceConfig


# ---------------------------------------------------------------------------
# Transform function library
# ---------------------------------------------------------------------------
# Each function follows the ElementOperator signature:
#   fn(element: Element, key: jax.Array) -> Element
#
# normalize_uint8 / cast_to_float32 are shared with Grain adapter (DRY).
# They use jax.tree.map (JIT-safe), NOT apply_to_dict (not JIT-safe).


def _normalize(element: Element, key: jax.Array) -> Element:
    """Normalize uint8 images to [0, 1] float32."""
    return element.replace(data=jax.tree.map(normalize_uint8, element.data))


def _cast_to_float32(element: Element, key: jax.Array) -> Element:
    """Cast all arrays to float32."""
    return element.replace(data=jax.tree.map(cast_to_float32, element.data))


def _scale(element: Element, key: jax.Array) -> Element:
    """Scale all values by 2.0."""
    new_data = jax.tree.map(lambda x: x * 2.0, element.data)
    return element.replace(data=new_data)


def _clip(element: Element, key: jax.Array) -> Element:
    """Clip values to [0, 1]."""
    new_data = jax.tree.map(lambda x: jnp.clip(x, 0.0, 1.0), element.data)
    return element.replace(data=new_data)


def _add(element: Element, key: jax.Array) -> Element:
    """Add 0.1 to all values."""
    new_data = jax.tree.map(lambda x: x + 0.1, element.data)
    return element.replace(data=new_data)


def _multiply(element: Element, key: jax.Array) -> Element:
    """Multiply all values by 0.9."""
    new_data = jax.tree.map(lambda x: x * 0.9, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# Stochastic transform functions (use key parameter for per-element RNG)
# ---------------------------------------------------------------------------


def _gaussian_noise(element: Element, key: jax.Array) -> Element:
    """Add Gaussian noise (std=0.05) to all arrays. Stochastic."""

    def add_noise(x):
        noise = jax.random.normal(key, shape=x.shape) * 0.05
        return x + noise

    return element.replace(data=jax.tree.map(add_noise, element.data))


def _random_brightness(element: Element, key: jax.Array) -> Element:
    """Adjust brightness by random delta in [-0.2, 0.2]. Stochastic."""
    delta = jax.random.uniform(key, minval=-0.2, maxval=0.2)
    new_data = jax.tree.map(lambda x: x + delta, element.data)
    return element.replace(data=new_data)


def _random_scale(element: Element, key: jax.Array) -> Element:
    """Scale by random factor in [0.8, 1.2]. Stochastic."""
    factor = jax.random.uniform(key, minval=0.8, maxval=1.2)
    new_data = jax.tree.map(lambda x: x * factor, element.data)
    return element.replace(data=new_data)


# ---------------------------------------------------------------------------
# Transform registries
# ---------------------------------------------------------------------------

# Deterministic: transform name -> function mapping
_TRANSFORM_FNS: dict[str, Any] = {
    "Normalize": _normalize,
    "CastToFloat32": _cast_to_float32,
    "Scale": _scale,
    "Clip": _clip,
    "Add": _add,
    "Multiply": _multiply,
}

# Stochastic: transform name -> function mapping
_STOCHASTIC_TRANSFORM_FNS: dict[str, Any] = {
    "GaussianNoise": _gaussian_noise,
    "RandomBrightness": _random_brightness,
    "RandomScale": _random_scale,
}

# Combined for setup() transform loop
_ALL_TRANSFORM_FNS: dict[str, Any] = {**_TRANSFORM_FNS, **_STOCHASTIC_TRANSFORM_FNS}


@register
class DataraxAdapter(BenchmarkAdapter):
    """BenchmarkAdapter implementation for Datarax.

    Supports all 25 benchmark scenarios. For standard sequential pipelines,
    chains OperatorNode wrappers for each transform in config.transforms.
    """

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._config: ScenarioConfig | None = None

    @property
    def name(self) -> str:
        return "Datarax"

    @property
    def version(self) -> str:
        import datarax

        return datarax.__version__

    def is_available(self) -> bool:
        return importlib.util.find_spec("datarax") is not None

    def _create_operator(
        self,
        transform_name: str,
        rngs: nnx.Rngs,
    ) -> ElementOperator:
        """Map a transform name to a Datarax ElementOperator."""
        fn = _TRANSFORM_FNS.get(transform_name)
        if fn is not None:
            config = ElementOperatorConfig(stochastic=False)
            return ElementOperator(config, fn=fn, rngs=rngs)

        fn = _STOCHASTIC_TRANSFORM_FNS.get(transform_name)
        if fn is not None:
            config = ElementOperatorConfig(stochastic=True, stream_name="augment")
            return ElementOperator(config, fn=fn, rngs=rngs)

        raise ValueError(
            f"Unknown transform '{transform_name}'. Available: {sorted(_ALL_TRANSFORM_FNS)}"
        )

    def _create_source(
        self,
        config: ScenarioConfig,
        data: Any,
        rngs: nnx.Rngs,
    ) -> Any:
        """Create the appropriate data source based on config."""
        backend = config.extra.get("backend") if config.extra else None

        if backend == "tfds_eager":
            from datarax.sources import TFDSEagerConfig, TFDSEagerSource

            src_config = TFDSEagerConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                seed=config.seed,
                as_supervised=True,
            )
            return TFDSEagerSource(src_config, rngs=rngs)

        if backend == "tfds_streaming":
            from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource

            src_config = TFDSStreamingConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                as_supervised=True,
            )
            return TFDSStreamingSource(src_config, rngs=rngs)

        if backend == "hf_eager":
            from datarax.sources import HFEagerConfig, HFEagerSource

            src_config = HFEagerConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
                seed=config.seed,
            )
            return HFEagerSource(src_config, rngs=rngs)

        if backend == "hf_streaming":
            from datarax.sources import HFStreamingConfig, HFStreamingSource

            src_config = HFStreamingConfig(
                name=config.extra["dataset_name"],
                split=config.extra["split"],
                shuffle=False,
            )
            return HFStreamingSource(src_config, rngs=rngs)

        # Default: MemorySource
        source_config = MemorySourceConfig(shuffle=False)
        return MemorySource(config=source_config, data=data, rngs=rngs)

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        self._config = config
        rngs = nnx.Rngs(config.seed, augment=config.seed + 1)

        source = self._create_source(config, data, rngs)
        pipeline = from_source(source, batch_size=config.batch_size)

        for transform_name in config.transforms:
            if transform_name in _ALL_TRANSFORM_FNS:
                op = self._create_operator(transform_name, rngs)
                pipeline = pipeline >> OperatorNode(op)

        self._pipeline = pipeline

    def warmup(self, num_batches: int = 3) -> None:
        """Run warmup batches to trigger JIT compilation."""
        for i, batch in enumerate(self._pipeline):
            if i >= num_batches:
                break
            data = batch.get_data()  # Works with both Batch and BatchView
            for leaf in jax.tree.leaves(data):
                if hasattr(leaf, "block_until_ready"):
                    leaf.block_until_ready()

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._pipeline

    def _materialize_batch(self, batch: Any) -> list[Any]:
        data = batch.get_data()  # Works with both Batch and BatchView
        # block_until_ready is JAX-specific; numpy arrays don't have it.
        # When the pipeline has no operators, data stays as numpy (zero-copy).
        for leaf in jax.tree.leaves(data):
            if hasattr(leaf, "block_until_ready"):
                leaf.block_until_ready()
        return jax.tree.leaves(data)

    def teardown(self) -> None:
        self._pipeline = None
        self._config = None

    def supported_scenarios(self) -> set[str]:
        """All 28 benchmark scenarios supported by Datarax."""
        return {
            "CV-1",
            "CV-2",
            "CV-3",
            "CV-4",
            "NLP-1",
            "NLP-2",
            "TAB-1",
            "TAB-2",
            "MM-1",
            "MM-2",
            "PC-1",
            "PC-2",
            "PC-3",
            "PC-4",
            "PC-5",
            "IO-1",
            "IO-2",
            "IO-3",
            "IO-4",
            "DIST-1",
            "DIST-2",
            "PR-1",
            "PR-2",
            "AUG-1",
            "AUG-2",
            "AUG-3",
            "NNX-1",
            "XFMR-1",
        }
