"""PipelineAdapter abstract base class and supporting dataclasses.

Defines the framework-agnostic interface for benchmarking any data loading
framework. Each adapter wraps a specific library (Datarax, Grain, PyTorch
DataLoader, etc.) and exposes a uniform setup -> warmup -> iterate -> teardown
lifecycle.

The ``iterate()`` and ``warmup()`` methods use the Template Method pattern:
subclasses implement ``_iterate_batches()`` and ``_materialize_batch()`` to
plug in framework-specific behavior, while the base class handles timing
bookkeeping and ``IterationResult`` construction.

Design ref: Section 7.1 of the benchmark report.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

from calibrax.core import BenchmarkAdapter as CalibraxBenchmarkAdapter


@dataclass
class ScenarioConfig:
    """Configuration for a benchmark scenario.

    Attributes:
        scenario_id: Unique scenario identifier (e.g., "CV-1").
        dataset_size: Number of elements in the synthetic dataset.
        element_shape: Shape of each element (e.g., (256, 256, 3) for images).
        batch_size: Batch size for data loading.
        transforms: List of transform names to apply.
        num_workers: Number of parallel workers (0 = single-threaded).
        seed: Random seed for reproducibility.
        extra: Additional scenario-specific parameters.
    """

    scenario_id: str
    dataset_size: int
    element_shape: tuple[int, ...]
    batch_size: int
    transforms: list[str]
    num_workers: int = 0
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationResult:
    """Result of iterating through a data pipeline.

    Attributes:
        num_batches: Number of batches consumed.
        num_elements: Total elements processed.
        total_bytes: Total bytes processed.
        wall_clock_sec: Total wall-clock time in seconds.
        per_batch_times: List of per-batch wall-clock times.
        first_batch_time: Time for the first batch (includes startup).
        extra_metrics: Additional framework-specific metrics.
    """

    num_batches: int
    num_elements: int
    total_bytes: int
    wall_clock_sec: float
    per_batch_times: list[float]
    first_batch_time: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class PipelineAdapter(CalibraxBenchmarkAdapter, ABC):
    """Abstract adapter for benchmarking any data loading framework.

    Subclasses wrap a specific framework and expose a uniform lifecycle:
    ``setup() -> warmup() -> iterate() -> teardown()``.

    The ``iterate()`` and ``warmup()`` methods are concrete Template Methods
    that delegate to two abstract hooks:

    - ``_iterate_batches()``: yields raw batches from the framework
    - ``_materialize_batch(batch)``: converts a batch to array-likes with
      ``.shape[0]`` and ``.nbytes`` (e.g. numpy arrays, JAX arrays)

    Subclasses MUST implement these two hooks plus ``setup()`` and
    ``teardown()``.  Override ``warmup()`` or ``iterate()`` only when the
    framework needs non-standard behavior (e.g. iterator reset after warmup).

    Design ref: Section 7.1 of the benchmark report.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter's human-readable display label, e.g. 'Datarax'."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Version string of the wrapped framework."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the wrapped framework is importable."""
        ...

    @abstractmethod
    def setup(self, config: ScenarioConfig, data: Any) -> None:
        """Build the data pipeline from config and synthetic data."""
        ...

    def teardown(self) -> None:
        """Release resources shared by all adapters.

        Resets ``_config`` to ``None``.  Subclasses should call
        ``super().teardown()`` **after** their own cleanup so that
        ``_config`` is still available during framework-specific teardown.
        """
        self._config = None

    def __init__(self) -> None:
        """Initialize the underlying calibrax adapter base.

        Sets ``_config`` to ``None``.  Subclasses MUST call
        ``super().__init__()`` before setting their own attributes.
        """
        super().__init__(target={"name": "pipeline"})
        self._config: ScenarioConfig | None = None

    def supports_scenario(self, scenario_id: str) -> bool:
        """Check whether this adapter supports a given scenario."""
        return scenario_id in self.supported_scenarios()

    def available_transforms(self) -> set[str]:
        """Return the set of transform names this adapter can execute.

        Override this to enable automatic scenario support detection.
        When overridden, ``supported_scenarios()`` will automatically
        derive which scenarios this adapter can run based on whether it
        has all required transforms.

        Returns:
            Set of transform name strings (e.g. {"Normalize", "CastToFloat32"}).
            Default returns empty set, which means supported_scenarios()
            must be overridden directly.
        """
        return set()

    def supported_scenarios(self) -> set[str]:
        """Return the set of scenario IDs this adapter can run.

        Default implementation derives support from ``available_transforms()``:
        a scenario is supported if the adapter has all transforms it requires
        (or the scenario requires no transforms). Adapters that don't use the
        transform registry pattern should override this directly.
        """
        transforms = self.available_transforms()
        if not transforms:
            return set()

        from benchmarks.scenarios import discover_scenarios

        supported: set[str] = set()
        for mod in discover_scenarios():
            for variant in mod.VARIANTS.values():
                required = set(variant.config.transforms)
                if required <= transforms:
                    supported.add(mod.SCENARIO_ID)
                    break
        return supported

    # ------------------------------------------------------------------
    # Template Method: iterate & warmup
    # ------------------------------------------------------------------

    @staticmethod
    def _close_iterator(iterator: Iterator[Any]) -> None:
        close_fn = getattr(iterator, "close", None)
        if callable(close_fn):
            close_fn()

    def _iter_exact_batches(self, num_batches: int) -> Iterator[Any]:
        iterator = iter(self._iterate_batches())
        try:
            yield from islice(iterator, max(0, num_batches))
        finally:
            self._close_iterator(iterator)

    def warmup(self, num_batches: int = 3) -> None:
        """Run warmup batches to trigger JIT compilation, caching, etc.

        Default implementation iterates and materializes batches.
        Override if the framework needs extra steps (e.g. iterator reset).
        """
        for batch in self._iter_exact_batches(num_batches):
            self._materialize_batch(batch)

    def iterate(self, num_batches: int) -> IterationResult:
        """Consume up to *num_batches* and return timing data.

        Uses ``time.perf_counter()`` for wall-clock measurement and
        delegates batch production / materialization to the two hooks
        ``_iterate_batches()`` and ``_materialize_batch()``.
        """
        per_batch_times: list[float] = []
        total_elements = 0
        total_bytes = 0
        first_batch_time: float | None = None

        start = time.perf_counter()

        for batch in self._iter_exact_batches(num_batches):
            batch_start = time.perf_counter()
            arrays = self._materialize_batch(batch)
            batch_end = time.perf_counter()

            if first_batch_time is None:
                first_batch_time = batch_end - start

            per_batch_times.append(batch_end - batch_start)

            if arrays:
                total_elements += arrays[0].shape[0]
                total_bytes += sum(a.nbytes for a in arrays)

        wall_clock = time.perf_counter() - start

        return IterationResult(
            num_batches=len(per_batch_times),
            num_elements=total_elements,
            total_bytes=total_bytes,
            wall_clock_sec=wall_clock,
            per_batch_times=per_batch_times,
            first_batch_time=first_batch_time or 0.0,
        )

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _iterate_batches(self) -> Iterator[Any]:
        """Yield raw batches from the framework's data loader.

        Each yielded value is passed to ``_materialize_batch()``.
        """
        ...

    @abstractmethod
    def _materialize_batch(self, batch: Any) -> list[Any]:
        """Convert a framework-specific batch to a list of array-likes.

        Each element must expose ``.shape[0]`` (batch dimension) and
        ``.nbytes`` (byte count).  Numpy arrays and JAX arrays both
        satisfy this contract.
        """
        ...

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_peak_memory_mb(self) -> float | None:
        """Return current process RSS in MB, or None if unavailable."""
        try:
            import psutil

            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers used by the registry (adapters/__init__.py)
    # ------------------------------------------------------------------

    @classmethod
    def _get_registry_name(cls) -> str:
        """Get the adapter name for registry keying."""
        try:
            return cls().name
        except (ImportError, AttributeError):
            return cls.__name__

    @classmethod
    def _check_available(cls) -> bool:
        """Instantiate to check availability (used by registry)."""
        try:
            return cls().is_available()
        except (ImportError, ModuleNotFoundError):
            return False

    @classmethod
    def _supported(cls) -> set[str]:
        """Instantiate to get supported scenarios (used by registry)."""
        try:
            return cls().supported_scenarios()
        except (ImportError, AttributeError):
            return set()
