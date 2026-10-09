"""A spy on the host stage's one placement: every unit it places, and the thread placing it."""

from __future__ import annotations

import threading
from typing import Any

import jax
import pytest

from datarax.pipeline import host_stage
from datarax.pipeline.run_units import HostElement


class PlacementSpy:
    """Every unit the host stage places (``host_stage._place``), and the thread that placed it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Wrap the placement for the test's duration.

        Args:
            monkeypatch: The test's monkeypatch, which restores the placement.
        """
        self.threads: list[int] = []
        self.leaf_types: set[type] = set()
        place = host_stage._place  # noqa: SLF001 - the one placement the spy counts

        def counted(element: HostElement) -> Any:
            self.threads.append(threading.get_ident())
            self.leaf_types |= {type(leaf) for leaf in jax.tree.leaves(element.batch)}
            return place(element)

        monkeypatch.setattr(host_stage, "_place", counted)

    def placed_on_the_caller(self, units: int) -> bool:
        """Whether exactly ``units`` units were placed, each on the calling thread."""
        return len(self.threads) == units and set(self.threads) <= {threading.get_ident()}
