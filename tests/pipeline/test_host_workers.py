"""The worker machinery the host stage reads through: the run's JAX settings and the arrival checks.

A spawned worker and a new thread start without the JAX settings the training process made in
code or holds in a context manager: precision (``jax_enable_x64``) and
``jax_threefry_partitionable``, which changes the order a naming key gives.
:class:`~datarax.pipeline.host_workers.JaxSettings` reads them where a run is opened, sets them
in a worker and enters them on a read thread; a change of either opens a new run. The default
impl is not carried: naming pins its own, so a change of it continues the run. The training process
counts the units arriving from workers and refuses one out of turn or a run ending short.
"""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager

import jax
import numpy as np
import pytest
from flax import nnx
from substrax.testing import restored_jax_config

from datarax.pipeline import Pipeline
from datarax.pipeline.host_workers import check_all_arrived, check_arrival, JaxSettings
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _on_a_new_thread[T](read: Callable[[], T]) -> T:
    found: list[T] = []
    thread = threading.Thread(target=lambda: found.append(read()))
    thread.start()
    thread.join()
    return found[0]


_OVERRIDES: dict[str, Callable[[], AbstractContextManager[object]]] = {
    "x64": lambda: jax.enable_x64(not jax.config.jax_enable_x64),
    "threefry_partitionable": lambda: jax.threefry_partitionable(
        not jax.config.jax_threefry_partitionable
    ),
}
"""Each setting's context manager, entered with a value other than the global one."""


class TestJaxSettings:
    """What a run is read under, read on the consumer and carried where it is not inherited."""

    def test_the_global_settings(self) -> None:
        settings = JaxSettings.current()
        assert settings == JaxSettings(
            x64=bool(jax.config.jax_enable_x64),
            threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        )
        assert hash(settings) == hash(dataclasses.replace(settings))
        with pytest.raises(dataclasses.FrozenInstanceError):
            settings.x64 = True  # type: ignore[misc]

    @pytest.mark.parametrize("name", sorted(_OVERRIDES))
    def test_a_context_manager_s_override_is_read(self, name: str) -> None:
        outside = JaxSettings.current()
        with _OVERRIDES[name]():
            inside = JaxSettings.current()
        assert getattr(inside, name) != getattr(outside, name)
        assert dataclasses.replace(inside, **{name: getattr(outside, name)}) == outside

    @pytest.mark.parametrize("name", sorted(_OVERRIDES))
    def test_entered_on_a_new_thread_restores_the_override_there(self, name: str) -> None:
        with _OVERRIDES[name]():
            settings = JaxSettings.current()
        # The control: a new thread reads the global value, not the consumer's override.
        assert _on_a_new_thread(JaxSettings.current) != settings

        def entered() -> JaxSettings:
            with settings.entered():
                return JaxSettings.current()

        assert _on_a_new_thread(entered) == settings
        assert JaxSettings.current() != settings  # and leaves the caller's thread as it was

    def test_apply_sets_every_setting_globally(self) -> None:
        settings = JaxSettings(x64=True, threefry_partitionable=False)
        with restored_jax_config() as changed:
            settings.apply()
            assert JaxSettings.current() == settings
            assert _on_a_new_thread(JaxSettings.current) == settings
        assert set(changed) == {"jax_enable_x64", "jax_threefry_partitionable"}


def _memory() -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), {"x": np.arange(40 * 2, dtype=np.int32).reshape(40, 2)}
    )


@pytest.mark.parametrize("name", sorted(_OVERRIDES))
def test_a_change_of_any_setting_opens_a_new_run_naming_it(name: str) -> None:
    """A run opened under an override is ended by a call outside it, which names the settings."""
    pipe = Pipeline(
        source=_memory(), stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=True, num_epochs=None
    )
    with _OVERRIDES[name]():
        batches = iter(pipe.raw_batches())
        next(batches)
    pipe.raw_batches()  # opens the run of the settings outside the override
    with pytest.raises(RuntimeError, match="jax_settings="):
        next(batches)
    pipe.close()


def test_a_change_of_the_default_impl_continues_the_run() -> None:
    """Naming pins its impl, so the caller's default impl is no part of a run's identity."""
    pipe = Pipeline(
        source=_memory(), stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=True, num_epochs=None
    )
    batches = iter(pipe.raw_batches())
    next(batches)
    with jax.default_prng_impl("rbg"):
        next(iter(pipe.raw_batches()))  # continues the open run
    next(batches)  # the run was not ended
    pipe.close()


class TestArrivalChecks:
    """Units from workers arrive in turn and all of them, or the run stops naming the lost one."""

    def test_the_unit_due_is_accepted(self) -> None:
        check_arrival(3, 3)

    def test_a_unit_out_of_turn_names_the_lost_one(self) -> None:
        with pytest.raises(RuntimeError, match=r"unit 3 of the run never arrived.*unit 4 came"):
            check_arrival(3, 4)

    @pytest.mark.parametrize("total", [5, None])
    def test_every_unit_or_no_end_is_accepted(self, total: int | None) -> None:
        check_all_arrived(5, total)

    def test_a_run_ending_short_names_the_first_missing_unit(self) -> None:
        with pytest.raises(RuntimeError, match=r"after 4 of its 5 units: unit 4 never arrived"):
            check_all_arrived(4, 5)
