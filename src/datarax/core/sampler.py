"""Base module for sampler components in Datarax.

This module defines the base class for all Datarax sampler components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

from typing import Any, Iterator

from flax import nnx

from datarax.core.structural import StructuralModule
from datarax.core.config import StructuralConfig


class SamplerModule(StructuralModule):
    """Enhanced base module for all Datarax sampler components.

    A SamplerModule determines the order in which records are accessed and
    processed. It handles global data transformations like shuffling and
    epoch management.

    This class extends StructuralModule for non-parametric structural processing.
    Concrete samplers define their own config classes extending StructuralConfig.

    Args:
        config: StructuralConfig or subclass with sampler-specific parameters
        rngs: Random number generators (required if stochastic=True)
        name: Optional name for the sampler

    Examples:
        from dataclasses import dataclass
        from datarax.core.config import StructuralConfig
        from datarax.core.sampler import SamplerModule
        from flax import nnx

        class SequentialSamplerConfig(StructuralConfig):
            num_records: int = 100
            num_epochs: int = 1
        SequentialSamplerConfig = dataclass(SequentialSamplerConfig)

        class SequentialSamplerModule(SamplerModule):
            def process(self, dataset_size):
                return list(range(min(self.config.num_records, dataset_size)))
            def __iter__(self):
                yield from self.process(100)

        config = SequentialSamplerConfig(stochastic=False, num_records=10)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))
    """

    def __init__(
        self,
        config: StructuralConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize SamplerModule with config.

        Args:
            config: Sampler configuration (already validated)
            rngs: Random number generators (required if stochastic=True)
            name: Optional sampler name
        """
        super().__init__(config, rngs=rngs, name=name)

        # Initialize last computed stats for caching
        self._last_computed_stats: dict[str, Any] | None = None

    def requires_rng_streams(self) -> list[str] | None:
        """Get the list of RNG streams required by this module.

        Returns:
            A list of required RNG stream names, or None if no RNG is required.
        """
        if self.stream_name is None:
            return None
        return [self.stream_name]

    def _compute_statistics(self, data: Any) -> dict[str, Any] | None:
        """Compute statistics from data.

        Uses precomputed_stats if available, otherwise calls batch_stats_fn.
        This method is safe to call directly (handles errors gracefully).

        Args:
            data: Input data to compute statistics from

        Returns:
            Dictionary of statistics, or None if computation fails/not configured
        """
        # Priority 1: Precomputed stats (static)
        if self.config.precomputed_stats is not None:
            # Handle both nnx.Variable and dict
            if hasattr(self.config.precomputed_stats, "get_value"):
                return self.config.precomputed_stats.get_value()
            return self.config.precomputed_stats

        # Priority 2: Dynamic computation via batch_stats_fn
        if self.config.batch_stats_fn is not None:
            try:
                return self.config.batch_stats_fn(data)
            except Exception:
                return None

        return None

    def __call__(self, n: int, *args, **kwargs) -> list[int]:  # type: ignore[override]
        """Enhanced sampling interface with caching and statistics.

        Args:
            n: The number of indices to sample.
            *args: Additional positional arguments (unused, for signature compatibility).
            **kwargs: Additional keyword arguments (unused, for signature compatibility).

        Returns:
            A list of sampled indices.
        """
        # Validate input
        if n < 0:
            raise ValueError(f"Number of samples must be non-negative, got {n}")

        # Handle empty sampling
        if n == 0:
            return []

        # Compute cache key once if caching is enabled
        cache_key: int | None = None
        if self.config.cacheable and self._cache is not None:
            cache_key = self._compute_cache_key(n)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Perform sampling
        result = self._sample_impl(n)

        # Compute statistics if enabled
        if self.config.batch_stats_fn is not None or self.config.precomputed_stats is not None:
            stats = self._compute_statistics(result)
            self._last_computed_stats = stats

        # Cache result if enabled
        if self.config.cacheable and self._cache is not None and cache_key is not None:
            self._cache[cache_key] = result

        # Update iteration count
        # Uses [...] for in-place mutation of IterationCount Variable (new NNX API)
        self._iteration_count[...] += 1

        return result

    def _sample_impl(self, n: int) -> list[int]:
        """Implementation method for sampling.

        Subclasses should override this method to provide their specific
        sampling logic.

        Args:
            n: The number of indices to sample.

        Returns:
            A list of sampled indices.
        """
        # Default implementation uses the sample method
        return self.sample(n)

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over indices into the dataset.

        Returns:
            An iterator that yields indices for data access.
        """
        raise NotImplementedError("Subclasses must implement __iter__")

    def __len__(self) -> int:
        """Return the total number of indices that will be sampled in one epoch.

        Returns:
            The total number of indices in the sampler.
        """
        msg = "Length determination not supported."
        raise NotImplementedError(msg)

    def sample(self, n: int) -> list[int]:
        """Return a list of sampled indices.

        This method returns all indices that would be yielded by the iterator,
        collected into a list. This is useful when you need all indices upfront
        rather than iterating through them one by one.

        Args:
            n: The number of indices to sample (typically the dataset size).

        Returns:
            A list of sampled indices.

        Note:
            The default implementation simply collects all indices from the iterator.
            Subclasses may override this for more efficient implementations.
        """
        # Update iteration count for direct sample calls
        # Uses [...] for in-place mutation of IterationCount Variable (new NNX API)
        self._iteration_count[...] += 1

        # Set dataset_size if the sampler needs it
        if hasattr(self, "dataset_size") and getattr(self, "dataset_size", None) is None:
            setattr(self, "dataset_size", n)

        # Collect all indices from the iterator
        return list(self)

    def get_state(self) -> dict[str, Any]:
        """Return the current state for checkpointing purposes.

        This extends the serializable state from DataraxModule with any
        additional state specific to this sampler.

        Returns:
            A dictionary containing the internal state of the Sampler.
        """
        # Use the enhanced DataraxModule implementation
        state = super().get_state()
        state.update(
            {
                "stream_name": self.stream_name,
            }
        )
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state from a checkpoint.

        This restores both the DataraxModule state and any additional
        state specific to this sampler.

        Args:
            state: A dictionary containing the internal state to restore.
        """
        # Use the enhanced DataraxModule implementation
        super().set_state(state)
        if "stream_name" in state:
            self.stream_name = state["stream_name"]

    def reset(self, seed: int | None = None) -> None:
        """Reset the sampler state, typically used to start a new epoch.

        Args:
            seed: Optional seed to use for shuffling or other random
                operations. If None, the sampler should use its default or
                previously set seed.
        """
        pass
