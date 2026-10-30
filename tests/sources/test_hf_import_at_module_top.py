"""HuggingFace ``datasets`` is imported at the top of its integration module, and only there.

``datarax.sources.hf_source`` is the HuggingFace integration: it imports ``datasets`` on its first
line of imports, so a missing ``data`` extra fails where the module is imported, with an
``ImportError`` naming the extra, and never inside a constructor in the middle of a run.
``import datarax.sources`` imports no ``datasets``: the HuggingFace names, ``from_hf`` among them,
are exported lazily. Each check runs in a fresh interpreter, since the test process may have
imported ``datasets`` already; ``sys.modules["datasets"] = None`` hides an installed package.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from substrax.testing import run_python

from tests.jax_test_environment import forwarded_jax_environment


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_SECONDS = 300.0
_HIDDEN = "import sys; sys.modules['datasets'] = None\n"


def _run[T](program: str, shape: type[T]) -> T:
    """What ``program`` sends with ``substrax.testing.report``, as ``shape``."""
    result = run_python(
        program,
        timeout=_CHILD_SECONDS,
        env=forwarded_jax_environment(os.environ),
        cwd=_REPO_ROOT,
    )
    return result.check().report_as(shape)


def test_importing_the_sources_package_imports_no_dataset_library() -> None:
    seen = _run(
        "import sys\n"
        "from substrax.testing import report\n"
        "import datarax.sources\n"
        "report([name in sys.modules for name in "
        "('datasets', 'tensorflow_datasets', 'datarax.sources.hf_source')])\n",
        list[bool],
    )

    assert seen == [False, False, False]


@pytest.mark.parametrize(
    "statement",
    [
        "import datarax.sources.hf_source",
        "from datarax.sources import from_hf",
        "from datarax.sources import HFEagerSource",
    ],
)
def test_importing_the_integration_without_datasets_names_the_extra(statement: str) -> None:
    printed = _run(
        _HIDDEN
        + "from substrax.testing import report\n"
        + "try:\n"
        + f"    {statement}\n"
        + "except ImportError as error:\n"
        + "    report(type(error).__name__ + ': ' + str(error))\n"
        + "else:\n"
        + "    report('imported')\n",
        str,
    )

    assert printed.startswith("ImportError: ")
    assert "datarax[data]" in printed


def test_from_hf_is_the_integration_module_s_and_imports_datasets_where_it_is_imported() -> None:
    pytest.importorskip("datasets")
    seen = _run(
        "import sys\n"
        "from substrax.testing import report\n"
        "import datarax.sources\n"
        "before = 'datasets' in sys.modules\n"
        "from datarax.sources import from_hf\n"
        "report([before, 'datasets' in sys.modules, from_hf.__module__])\n",
        tuple[bool, bool, str],
    )

    assert seen == (False, True, "datarax.sources.hf_source")
