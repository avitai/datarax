"""Record every kind-equal host value a program reads while JAX traces.

A ``HostValue`` compares equal to every holder of its kind, so a program that read one would serve
it to every module sharing the graphdef. :class:`HolderRecorder` notes, while JAX traces:

* ``taken``: a ``HostValue`` taken off an NNX module (``nnx.Pytree.__getattribute__``);
* ``read``: an attribute of a ``HostValue`` read, by whatever route the holder arrived;
* ``captured``: a ``HostValue`` reachable from a host callback's closure, bound ``self`` or
  partial arguments (``jax.experimental.io_callback``, ``jax.pure_callback``,
  ``jax.debug.callback``, and every ``datarax`` module name bound to one of them).

JAX is tracing when its trace state differs from the eager one, the test flax's own NNX
transforms make (``flax.nnx.tracers``). JAX keeps one eager trace for every thread, so a host
thread's eager work is not recorded.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable, Iterator
from types import TracebackType
from typing import Any

import jax
import jax.experimental
from flax import nnx
from jax.extend.core import get_opaque_trace_state

from datarax.core.host_values import HostValue


_CALLBACKS: tuple[tuple[Any, str], ...] = (
    (jax.experimental, "io_callback"),
    (jax, "pure_callback"),
    (jax.debug, "callback"),
)
_REACH_DEPTH = 6


def _tracing(eager: Any) -> bool:
    return get_opaque_trace_state(convention="nnx") != eager


def _holders_in(value: Any) -> Iterator[HostValue]:
    if isinstance(value, HostValue):
        yield value
    elif isinstance(value, tuple | list | frozenset | set):
        for item in value:
            yield from _holders_in(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _holders_in(item)


def _reachable(value: Any, depth: int, seen: set[int]) -> Iterator[Any]:
    if depth > _REACH_DEPTH or id(value) in seen:
        return
    seen.add(id(value))
    yield value
    inner: list[Any] = []
    if isinstance(value, functools.partial):
        inner += [value.func, *value.args, *value.keywords.values()]
    for link in ("__self__", "__wrapped__"):  # a bound method's object, a wrapper's function
        if hasattr(value, link):
            inner.append(getattr(value, link))
    if isinstance(value, tuple | list | frozenset | set):
        inner += list(value)
    if isinstance(value, dict):
        inner += list(value.values())
    if isinstance(value, nnx.Pytree):  # a captured module carries its holders into the callback
        inner += list(vars(value).values())
    for cell in getattr(value, "__closure__", None) or ():
        try:
            inner.append(cell.cell_contents)
        except ValueError:  # an empty cell: a closure variable not yet bound
            continue
    for item in inner:
        yield from _reachable(item, depth + 1, seen)


class HolderRecorder:
    """Records each kind-equal holder a program takes, reads or captures while JAX traces.

    Attributes:
        taken: ``(how, holder type name)`` per event, in order; ``how`` is ``"taken"``,
            ``"read"`` or ``"captured"``.
    """

    def __init__(self) -> None:
        """Start with nothing recorded and nothing patched."""
        self.taken: list[tuple[str, str]] = []
        self._undo: list[Callable[[], None]] = []
        self._eager: Any = None

    def __enter__(self) -> HolderRecorder:
        """Clear JAX's caches, then patch the module, holder and callback entry points.

        A program served from a cache is not traced again, so what it reads would go unrecorded:
        ``jax.clear_caches()`` makes every program the block runs trace afresh, those held in
        datarax's graphdef-keyed caches included, since each holds a ``jax.jit`` function.

        Returns:
            The recorder.
        """
        jax.clear_caches()
        self._eager = get_opaque_trace_state(convention="nnx")
        self._patch_modules()
        self._patch_holders()
        self._patch_callbacks()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore every entry point patched, last first."""
        while self._undo:
            self._undo.pop()()

    def _set(self, owner: Any, name: str, value: Any) -> None:
        missing = object()
        original = vars(owner).get(name, missing)
        setattr(owner, name, value)
        if original is missing:  # inherited: removing the patch restores the inherited one
            self._undo.append(lambda: delattr(owner, name))
        else:
            self._undo.append(lambda: setattr(owner, name, original))

    def _patch_modules(self) -> None:
        original = nnx.Pytree.__getattribute__
        recorder = self

        def getattribute(module: Any, name: str) -> Any:
            value = original(module, name)
            if not name.startswith("__") and _tracing(recorder._eager):
                for holder in _holders_in(value):
                    recorder.taken.append(("taken", type(holder).__name__))
            return value

        self._set(nnx.Pytree, "__getattribute__", getattribute)

    def _patch_holders(self) -> None:
        original = HostValue.__getattribute__
        recorder = self

        def getattribute(holder: HostValue, name: str) -> Any:
            if not name.startswith("__") and _tracing(recorder._eager):
                recorder.taken.append(("read", type(holder).__name__))
            return original(holder, name)

        self._set(HostValue, "__getattribute__", getattribute)

    def _patch_callbacks(self) -> None:
        for owner, name in _CALLBACKS:
            original = getattr(owner, name)
            wrapper = self._capturing(original)
            self._set(owner, name, wrapper)
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith("datarax") and module is not None:
                    for attribute, value in list(vars(module).items()):
                        if value is original:
                            self._set(module, attribute, wrapper)

    def _capturing(self, callback_api: Callable[..., Any]) -> Callable[..., Any]:
        recorder = self

        @functools.wraps(callback_api)
        def wrapper(callback: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            if _tracing(recorder._eager):
                for value in _reachable(callback, 0, set()):
                    if isinstance(value, HostValue):
                        recorder.taken.append(("captured", type(value).__name__))
            return callback_api(callback, *args, **kwargs)

        return wrapper
