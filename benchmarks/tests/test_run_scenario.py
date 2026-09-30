"""Tests for ``run_scenario``: the timed iteration compiles nothing after warmup.

A compile inside the timed ``iterate`` is time the benchmark reports as data loading; the
lifecycle refuses it with :class:`~substrax.testing.compiles.CompileCountError`, naming the
adapter, scenario and variant. Warmup may compile: that is what it is for.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import numpy as np
import pytest
from substrax.testing.compiles import CompileCountError

from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig
from benchmarks.scenarios.base import run_scenario, ScenarioVariant


_BATCH_SIZE = 4
_NUM_BATCHES = 3


class _NumpyAdapter(PipelineAdapter):
    """Serves NumPy batches; compiles nothing anywhere. Records its lifecycle calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    @property
    def name(self) -> str:
        return "FakeNumpy"

    @property
    def version(self) -> str:
        return "0.0.0"

    def is_available(self) -> bool:
        return True

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        self._config = config
        self.calls.append(("setup", len(data["x"])))

    def warmup(self, num_batches: int = 3, *, timed_batches: int | None = None) -> None:
        self.calls.append(("warmup", (num_batches, timed_batches)))
        super().warmup(num_batches, timed_batches=timed_batches)

    def teardown(self) -> None:
        self.calls.append(("teardown", None))
        super().teardown()

    def _iterate_batches(self) -> Iterator[Any]:
        while True:
            yield np.ones((_BATCH_SIZE, 2), dtype=np.float32)

    def _materialize_batch(self, batch: Any) -> list[Any]:
        return [batch]


class _CompilingAdapter(_NumpyAdapter):
    """Runs every batch through a freshly built jitted function: each batch compiles."""

    @property
    def name(self) -> str:
        return "FakeCompiling"

    def _materialize_batch(self, batch: Any) -> list[Any]:
        doubled = jax.jit(lambda x: x * 2)(batch)
        return [jax.block_until_ready(doubled)]


class _WarmupCompilingAdapter(_NumpyAdapter):
    """Compiles in warmup only; the timed batches reuse the warmed program."""

    @property
    def name(self) -> str:
        return "FakeWarmupCompiling"

    def __init__(self) -> None:
        super().__init__()
        self._double = jax.jit(lambda x: x * 2)

    def _materialize_batch(self, batch: Any) -> list[Any]:
        return [jax.block_until_ready(self._double(batch))]


def _variant() -> ScenarioVariant:
    return ScenarioVariant(
        config=ScenarioConfig(
            scenario_id="FAKE-1",
            dataset_size=12,
            element_shape=(2,),
            batch_size=_BATCH_SIZE,
            transforms=[],
            extra={"variant_name": "tiny"},
        ),
        data_generator=lambda n: {"x": np.ones((n, 2), dtype=np.float32)},
    )


def test_a_timed_iteration_compiling_nothing_is_measured() -> None:
    result = run_scenario(
        _NumpyAdapter(), _variant(), num_batches=_NUM_BATCHES, warmup_batches=1, num_repetitions=2
    )
    assert result.timing is not None
    assert result.timing.num_batches == _NUM_BATCHES


def test_a_compile_in_warmup_is_allowed() -> None:
    result = run_scenario(
        _WarmupCompilingAdapter(),
        _variant(),
        num_batches=_NUM_BATCHES,
        warmup_batches=1,
        num_repetitions=1,
    )
    assert result.timing is not None
    assert result.timing.num_batches == _NUM_BATCHES


def test_a_compile_in_the_timed_iteration_is_refused_naming_the_run() -> None:
    with pytest.raises(CompileCountError) as raised:
        run_scenario(
            _CompilingAdapter(),
            _variant(),
            num_batches=_NUM_BATCHES,
            warmup_batches=1,
            num_repetitions=1,
        )
    assert raised.value.expected == 0
    assert len(raised.value.names) == _NUM_BATCHES
    notes = "\n".join(getattr(raised.value, "__notes__", []))
    assert "FakeCompiling" in notes
    assert "FAKE-1" in notes
    assert "tiny" in notes


def test_the_adapter_is_torn_down_when_the_timed_iteration_is_refused() -> None:
    adapter = _CompilingAdapter()
    with pytest.raises(CompileCountError):
        run_scenario(
            adapter, _variant(), num_batches=_NUM_BATCHES, warmup_batches=1, num_repetitions=1
        )
    assert adapter.calls[-1] == ("teardown", None)


def test_warmup_is_told_how_many_batches_the_timed_iteration_serves() -> None:
    adapter = _NumpyAdapter()
    run_scenario(adapter, _variant(), num_batches=_NUM_BATCHES, warmup_batches=2, num_repetitions=1)
    assert ("warmup", (2, _NUM_BATCHES)) in adapter.calls


def test_the_data_is_generated_at_the_configured_dataset_size() -> None:
    adapter = _NumpyAdapter()
    run_scenario(adapter, _variant(), num_batches=_NUM_BATCHES, warmup_batches=1, num_repetitions=1)
    assert adapter.calls[0] == ("setup", 12)


class TestScenarioVariantSize:
    """A variant's data is generated at its configured size, which can be set smaller."""

    def test_generate_data_uses_the_configured_dataset_size(self) -> None:
        data = _variant().generate_data()
        assert data["x"].shape == (12, 2)

    def test_with_dataset_size_changes_only_the_size(self) -> None:
        original = _variant()
        shrunk = original.with_dataset_size(8)
        assert shrunk.config.dataset_size == 8
        assert shrunk.generate_data()["x"].shape == (8, 2)
        assert shrunk.config.batch_size == original.config.batch_size
        assert shrunk.config.element_shape == original.config.element_shape
        assert shrunk.config.transforms == original.config.transforms
        assert shrunk.config.extra == original.config.extra
        assert original.config.dataset_size == 12

    def test_with_dataset_size_refuses_an_empty_dataset(self) -> None:
        with pytest.raises(ValueError, match="dataset_size"):
            _variant().with_dataset_size(0)
