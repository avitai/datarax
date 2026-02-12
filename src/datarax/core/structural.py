"""StructuralModule - unified non-parametric structural processor module.

This module provides StructuralModule, which unifies BatcherModule, SamplerModule,
SharderModule, and other structural processors into a single base class for all
non-parametric, structural data organization operations.

Key Features:

- Config-based initialization with StructuralConfig (frozen/immutable)
- Stochastic mode (with RNG for random organization)
- Deterministic mode (fixed organization)
- Single process() method for all structural operations
- No learnable parameters (compile-time constants only)
- JIT compatibility
- Statistics system (inherited from DataraxModule)
"""

from typing import Any

from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.module import DataraxModule


class StructuralModule(DataraxModule):
    """Base class for non-parametric structural processors.

    Structural modules organize/reorganize data without learnable parameters.
    Configuration is immutable (frozen dataclass) representing compile-time constants.

    Structural modules change data structure/organization, not data values.
    They are NOT differentiable and have no learnable parameters.

    The structural pattern uses a single process() method:

    - process() - Transforms input structure (abstract method)

    Args:
        config: StructuralConfig (already validated via __post_init__, frozen)
        rngs: Random number generators (required if stochastic=True)
        name: Optional name for the structural module

    Attributes:
        config: Structural module configuration (immutable)
        stochastic: Whether this module uses randomness (from config)
        stream_name: RNG stream name (from config, required if stochastic=True)

    Examples:
        Deterministic batcher:

        ```python
        config = BatcherConfig(stochastic=False, batch_size=32)
        batcher = BatcherModule(config)
        batches = batcher.process(elements)
        ```

        Stochastic sampler:

        ```python
        config = SamplerConfig(stochastic=True, stream_name="sampler", num_samples=100)
        sampler = SamplerModule(config, rngs=nnx.Rngs(42))
        indices = sampler.process(dataset_size=1000)
        ```
    """

    def __init__(
        self,
        config: StructuralConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize StructuralModule with config.

        Args:
            config: Structural module configuration (already validated, frozen)
            rngs: Random number generators (required if stochastic=True)
            name: Optional module name

        Raises:
            ValueError: If stochastic=True but rngs is None
        """
        super().__init__(config, rngs=rngs, name=name)

        # Runtime validation: Stochastic modules require rngs
        if config.stochastic and rngs is None:
            raise ValueError(
                f"Stochastic structural modules require rngs parameter. "
                f"Pass rngs=nnx.Rngs(..., {config.stream_name}=...)"
            )

        # Convenience properties (avoid repeated config access)
        self.stochastic = config.stochastic
        self.stream_name = config.stream_name

    # ========================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ========================================================================

    def process(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        """Process input structure.

        This method transforms the structure/organization of input data
        without modifying the data values themselves.

        Subclasses MUST implement this method.

        The input/output types depend on the specific structural processor:

        - Batcher: list[Element] -> list[Batch]
        - Sampler: int -> list[int]
        - Sharder: Batch -> Sharded[Batch]
        - Splitter: Dataset -> tuple[Dataset, Dataset]

        Args:
            input: Input to process (type varies by processor)
            *args: Additional positional arguments (processor-specific)
            **kwargs: Additional keyword arguments (processor-specific)

        Returns:
            Processed output (type varies by processor)

        Examples:
            Batcher implementation:

            ```python
            def process(self, elements: list[Element]) -> list[Batch]:
                batches = []
                for i in range(0, len(elements), self.config.batch_size):
                    batch_elements = elements[i:i + self.config.batch_size]
                    batches.append(Batch.from_elements(batch_elements))
                return batches
            ```

            Sampler implementation (deterministic):

            ```python
            def process(self, dataset_size: int) -> list[int]:
                return list(range(min(self.config.num_samples, dataset_size)))
            ```

            Sampler implementation (stochastic):

            ```python
            def process(self, dataset_size: int) -> list[int]:
                rng = self.rngs[self.config.stream_name]()
                indices = jax.random.choice(
                    rng, dataset_size, shape=(self.config.num_samples,),
                    replace=self.config.replacement
                )
                return indices.tolist()
            ```
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement process() method")

    # ========================================================================
    # Concrete Methods (final implementation)
    # ========================================================================

    def __call__(self, input: Any, *args, **kwargs) -> Any:
        """Main entry point for structural processing.

        Delegates to process() method. No caching or statistics by default
        (structural operations typically don't benefit from caching).

        Args:
            input: Input to process
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Processed output
        """
        # Delegate to process()
        # Note: Structural modules typically don't use caching or statistics,
        # but the infrastructure is available if needed
        return self.process(input, *args, **kwargs)
