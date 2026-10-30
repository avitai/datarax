"""A test ends with the threads it started: the thread check the test suite runs after every test.

Work a test leaves running on another thread runs inside the next tests: it makes device arrays a
later live-array check reports, and Python calls a later count includes. The check waits a bounded
time for the threads a test started, then fails the test that left one running. Library threads
that live as long as the process are made at session start, not left to the first test that
needs them.
"""

from __future__ import annotations

import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
import tqdm
from grain._src.python.ipc.shared_memory_array import SharedMemoryArray
from substrax.testing import run_python

from tests.thread_check import grain_thread_names, threads_left


_REPOSITORY = Path(__file__).resolve().parents[1]


def test_the_children_import_the_check_from_this_repository() -> None:
    """A child puts the repository first on its path; its parent directory holds other clones."""
    assert (_REPOSITORY / "tests" / "thread_check.py").is_file()
    assert (_REPOSITORY / "pyproject.toml").read_text().count('name = "datarax"') == 1


@pytest.fixture
def blocked() -> Iterator[threading.Thread]:
    """A thread that runs until the test ends."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name="blocked-for-the-test")
    thread.start()
    try:
        yield thread
    finally:
        release.set()
        thread.join(10)


def test_loading_the_check_imports_no_jax() -> None:
    """The root conftest loads it before tests/conftest.py sets the JAX environment."""
    program = textwrap.dedent(
        f"""
        import sys
        from substrax.testing import report
        sys.path.insert(0, {str(_REPOSITORY)!r})
        import tests.thread_check
        report({{"jax": "jax" in sys.modules}})
        """
    )

    assert run_python(program, timeout=120).check().report_as(dict[str, bool]) == {"jax": False}


def test_a_thread_that_ends_within_the_bound_is_not_left() -> None:
    before = frozenset(threading.enumerate())
    threading.Thread(target=threading.Event().wait, args=(0.05,), name="short").start()

    assert threads_left(before, within=2.0) == []


def test_a_thread_still_running_after_the_bound_is_left(blocked: threading.Thread) -> None:
    before = frozenset(threading.enumerate()) - {blocked}

    assert threads_left(before, within=0.1) == [blocked]


def test_the_grain_threads_are_listed_by_name() -> None:
    thread = threading.Thread(target=threading.Event().wait, args=(0.2,), name="grain-test_0")
    thread.start()
    try:
        assert "grain-test_0" in grain_thread_names()
    finally:
        thread.join(10)

    assert "grain-test_0" not in grain_thread_names()


def test_making_a_tqdm_bar_starts_no_monitor_thread() -> None:
    before = frozenset(threading.enumerate())

    with tqdm.tqdm(total=1, disable=False, file=_Discard()):
        pass

    assert threads_left(before, within=0.0) == []


def test_grain_s_shared_memory_release_threads_exist_before_any_test() -> None:
    before = frozenset(threading.enumerate())

    SharedMemoryArray.enable_async_del(1)  # Grain's own call; a no-op once the pool exists

    assert frozenset(threading.enumerate()) == before


_CHILD_TESTS = """
import threading

_RELEASE = threading.Event()


def test_leaves_a_thread_running():
    threading.Thread(target=_RELEASE.wait, name="left-by-the-test", daemon=True).start()


def test_ends_its_thread_within_the_bound():
    threading.Thread(target=threading.Event().wait, args=(0.2,), name="short-tail").start()
"""


def _run_child(tmp_path: Path, selected: str, *, without: str = "") -> tuple[int, str]:
    """Run one test of ``_CHILD_TESTS`` in a fresh pytest with the thread check.

    Args:
        tmp_path: Where the child's test file goes.
        selected: The test of ``_CHILD_TESTS`` to run.
        without: A module the child cannot import, or ``""`` for none.

    Returns:
        The child's exit code and its output.
    """
    (tmp_path / "test_child.py").write_text(_CHILD_TESTS)
    program = textwrap.dedent(
        f"""
        import sys
        import pytest

        sys.path.insert(0, {str(_REPOSITORY)!r})
        if {without!r}:
            sys.modules[{without!r}] = None  # an environment without it
        sys.exit(pytest.main([
            {str(tmp_path / "test_child.py")!r} + "::" + {selected!r},
            "-p", "tests.thread_check", "-p", "no:randomly", "-p", "no:cacheprovider",
            "-q", "--no-header", "-o", "addopts=", "--rootdir", {str(tmp_path)!r},
        ]))
        """
    )
    result = run_python(program, timeout=120, cwd=tmp_path)
    return result.returncode, result.stdout + result.stderr


def test_a_test_that_leaves_a_thread_running_fails_naming_it(tmp_path: Path) -> None:
    code, output = _run_child(tmp_path, "test_leaves_a_thread_running")

    assert code == 1
    assert "left-by-the-test" in output
    assert "Event.wait" in output


def test_a_test_whose_thread_ends_within_the_bound_passes(tmp_path: Path) -> None:
    code, output = _run_child(tmp_path, "test_ends_its_thread_within_the_bound")

    assert code == 0, output


def test_the_check_runs_where_tqdm_is_not_installed(tmp_path: Path) -> None:
    """tqdm comes only with the data extras; a job without them still runs the check."""
    code, output = _run_child(tmp_path, "test_ends_its_thread_within_the_bound", without="tqdm")

    assert code == 0, output


class _Discard:
    """A stream that drops what is written to it."""

    def write(self, text: str) -> int:
        """Drop ``text``.

        Args:
            text: What would be written.

        Returns:
            The length of ``text``.
        """
        return len(text)

    def flush(self) -> None:
        """Nothing to flush."""
