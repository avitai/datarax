"""Grain-backed shuffle sampler for known-size datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import grain

from datarax.core.config import StructuralConfig
from datarax.core.sampler import SamplerModule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShuffleSamplerConfig(StructuralConfig):
    """Configuration for Grain ``IndexSampler`` shuffling."""

    dataset_size: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate shuffle sampler configuration."""
        object.__setattr__(self, "stochastic", False)
        object.__setattr__(self, "stream_name", None)
        super().__post_init__()
        if self.dataset_size < 0:
            raise ValueError(f"dataset_size must be non-negative, got {self.dataset_size}")
        if self.seed < 0 or self.seed >= 2**32:
            raise ValueError("seed must be in [0, 2**32)")


class ShuffleSampler(SamplerModule):
    """Checkpointable wrapper around Grain ``IndexSampler``."""

    config: ShuffleSamplerConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        config: ShuffleSamplerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a Grain-backed shuffle sampler."""
        super().__init__(config, rngs=rngs, name=name)
        self.dataset_size = config.dataset_size
        self.seed = config.seed
        self.position = nnx.Variable(0)
        self._resume_next_iter = nnx.Variable(False)

    @staticmethod
    def create_static_iterator(dataset_size: int, seed: int = 0) -> Iterator[int]:
        """Create a one-epoch Grain ``IndexSampler`` iterator."""
        sampler = grain.samplers.IndexSampler(
            num_records=dataset_size,
            shard_options=grain.sharding.NoSharding(),
            shuffle=True,
            seed=seed,
            num_epochs=1,
        )
        for metadata in sampler:
            if metadata.record_key is None:
                raise ValueError("Grain IndexSampler emitted metadata without a record key")
            yield int(metadata.record_key)

    def __len__(self) -> int:
        """Return the number of indices in one epoch."""
        return self.dataset_size

    def __iter__(self) -> Iterator[int]:
        """Yield shuffled dataset indices, resuming after restored checkpoints."""
        start = self.position.get_value() if self._resume_next_iter.get_value() else 0
        if not self._resume_next_iter.get_value():
            self.position.set_value(0)

        order = list(self.create_static_iterator(self.dataset_size, self.seed))
        for cursor in range(start, self.dataset_size):
            self.position.set_value(cursor + 1)
            yield order[cursor]

        self._resume_next_iter.set_value(False)

    def get_state(self) -> dict[str, Any]:
        """Return checkpoint state for replaying the current Grain order."""
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
            if seed < 0 or seed >= 2**32:
                raise ValueError("seed must be in [0, 2**32)")
            self.seed = seed
        self.position.set_value(0)
        self._resume_next_iter.set_value(False)
