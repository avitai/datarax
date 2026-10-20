"""The pack plans a decoded copy keeps: planned once over its records, reused by every later run.

Planning a dataset of frontier size takes long (``datarax.core.pack_plan``), and a copy's records
never change, so a plan is computed once and kept beside the copy, as levanter keeps its
pre-packed dataset and OLMo-core its stored plan. Each lives in ``plan-<digest>/``, the digest the
SHA-256 of what the plan depends on: the copy's content digest, the planner's version, and the
caller's planning parameters (the packed fields and their row lengths, the segment cap, the
window, the planning seed). It is planned into ``plan-<digest>.partial/`` under the copy's lock
and renamed into place once its files are durable, so a directory without the suffix holds a
finished plan, and a plan that died is planned again. Rebuilding the copy removes every plan
(``clear_copy``).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from datarax.core.pack_plan import PackPlan, PLAN_META, PLANNER_VERSION
from datarax.sources.decoded_copy._files import CopyLock, fsync_directory, fsync_file
from datarax.sources.decoded_copy._layout import PARTIAL
from datarax.sources.decoded_copy._manifest import read_manifest


def plan_directory(directory: str | os.PathLike[str], parameters: Mapping[str, Any]) -> Path:  # noqa: DOC502 - read_manifest raises
    """Where the plan of the copy in ``directory`` for ``parameters`` is kept.

    Args:
        directory: A finished copy's directory.
        parameters: The planning parameters, JSON values.

    Returns:
        ``<directory>/plan-<digest>``.

    Raises:
        FileNotFoundError: If the copy is incomplete.
        ValueError: If the copy is of another format.
    """
    root = Path(directory)
    key = {
        "content": read_manifest(root)["content"]["digest"],
        "planner": PLANNER_VERSION,
        "parameters": parameters,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return root / f"plan-{digest}"


def copy_plan(  # noqa: DOC502 - plan_directory raises
    directory: str | os.PathLike[str],
    *,
    parameters: Mapping[str, Any],
    plan: Callable[[Path], PackPlan],
) -> PackPlan:
    """The pack plan of the copy in ``directory`` for ``parameters``, planned once.

    Args:
        directory: A finished copy's directory.
        parameters: Everything the plan depends on besides the copy's content and the planner's
            version, as JSON values: the packed fields and their row lengths, the segment cap,
            the window and the planning seed.
        plan: Writes the plan into the directory it is given (``pack_plan.plan_rows``); called
            only when no finished plan for these parameters is kept.

    Returns:
        The kept plan, read through memory maps.

    Raises:
        FileNotFoundError: If the copy is incomplete.
        ValueError: If the copy is of another format.
    """
    final = plan_directory(directory, parameters)
    if (final / PLAN_META).is_file():
        return PackPlan.open(final)
    with CopyLock(directory):
        if (final / PLAN_META).is_file():  # another process planned it meanwhile
            return PackPlan.open(final)
        partial = final.with_name(f"{final.name}{PARTIAL}")
        if partial.exists():  # a plan that died
            shutil.rmtree(partial)
        plan(partial)
        fsync_file(partial / PLAN_META)
        fsync_directory(partial)
        partial.rename(final)
        fsync_directory(final.parent)
    return PackPlan.open(final)


__all__ = ["copy_plan", "plan_directory"]
