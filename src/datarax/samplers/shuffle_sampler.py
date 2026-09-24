"""Shuffle sampler for known-size datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from flax import nnx

from datarax.core.config import SamplerConfig
from datarax.core.sampler import SamplerModule
from datarax.samplers._validation import validate_seed
from datarax.samplers.index_shuffle import index_shuffle


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShuffleSamplerConfig(SamplerConfig):
    """Configuration for :class:`ShuffleSampler`."""

    dataset_size: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate shuffle sampler configuration."""
        object.__setattr__(self, "stochastic", False)
        object.__setattr__(self, "stream_name", None)
        super().__post_init__()
        if self.dataset_size < 0:
            raise ValueError(f"dataset_size must be non-negative, got {self.dataset_size}")
        validate_seed(self.seed)


class ShuffleSampler(SamplerModule):
    """Checkpointable sampler serving a keyed shuffle of ``[0, dataset_size)``."""

    config: ShuffleSamplerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: ShuffleSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the shuffle sampler."""
        super().__init__(config, rngs=rngs, name=name)
        self.dataset_size = config.dataset_size
        self.seed = config.seed
        self.position = nnx.Variable(0)
        self._resume_next_iter = nnx.Variable(False)

    def __len__(self) -> int:
        """Return the number of indices in one epoch."""
        return self.dataset_size

    def __iter__(self) -> Iterator[int]:
        """Yield shuffled dataset indices, resuming after restored checkpoints.

        Each position is mapped through ``index_shuffle`` on demand, so a full epoch never
        materializes a permutation array. The order is a function of ``seed`` alone, so a
        restored position replays it exactly.
        """
        start = self.position.get_value() if self._resume_next_iter.get_value() else 0
        if not self._resume_next_iter.get_value():
            self.position.set_value(0)

        for cursor in range(start, self.dataset_size):
            self.position.set_value(cursor + 1)
            yield index_shuffle(cursor, self.seed, self.dataset_size)

        self._resume_next_iter.set_value(False)

    def get_state(self) -> dict[str, Any]:
        """Return checkpoint state for replaying the current order."""
        state = super().get_state()
        state["sampler_state"] = {
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "position": self.position.get_value(),
        }
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore checkpoint state for deterministic replay."""
        custom = self._split_state(state, {"sampler_state"})
        sampler_state = custom.get("sampler_state")
        if sampler_state is None:
            return

        self.dataset_size = sampler_state["dataset_size"]
        self.seed = sampler_state["seed"]
        self.position.set_value(sampler_state["position"])
        self._resume_next_iter.set_value(True)

    def reset(self, seed: int | None = None) -> None:
        """Reset the sampler to the beginning of an epoch."""
        if seed is not None:
            validate_seed(seed)
            self.seed = seed
        self.position.set_value(0)
        self._resume_next_iter.set_value(False)
