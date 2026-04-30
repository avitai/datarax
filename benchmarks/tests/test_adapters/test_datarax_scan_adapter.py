"""Tests for DataraxScanAdapter — the whole-epoch scan-mode adapter.

The scan adapter is a second measurement dimension over the same
Datarax pipeline as DataraxAdapter, exercising ``Pipeline.scan``
instead of the iterator path. It is a sibling adapter (subclass of
DataraxAdapter), so most lifecycle assertions inherit from the iter
adapter's contract; this test file focuses on the scan-specific
contract (distinct ``name``, valid ``IterationResult`` shape, scan
mode marker in extra metrics).
"""

from benchmarks.adapters.datarax_scan_adapter import DataraxScanAdapter
from benchmarks.tests.test_adapters.conftest import assert_valid_iteration_result


class TestDataraxScanAdapterProperties:
    """Test scan-adapter properties distinct from iter adapter."""

    def test_name_is_distinct_from_iter_adapter(self) -> None:
        adapter = DataraxScanAdapter()
        assert adapter.name == "Datarax-scan"

    def test_is_available(self) -> None:
        adapter = DataraxScanAdapter()
        assert adapter.is_available() is True


class TestDataraxScanAdapterLifecycle:
    """Test the scan-mode lifecycle: setup → warmup → iterate → teardown."""

    def test_iterate_returns_valid_result(self, nlp1_small_config, small_token_data) -> None:
        adapter = DataraxScanAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=2)
        result = adapter.iterate(num_batches=4)
        assert_valid_iteration_result(result)
        assert result.num_batches == 4
        adapter.teardown()

    def test_iterate_marks_scan_mode_in_extra_metrics(
        self, nlp1_small_config, small_token_data
    ) -> None:
        adapter = DataraxScanAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=2)
        result = adapter.iterate(num_batches=4)
        assert result.extra_metrics.get("scan_mode") == 1.0
        adapter.teardown()

    def test_iterate_counts_elements_as_num_batches_times_batch_size(
        self, nlp1_small_config, small_token_data
    ) -> None:
        adapter = DataraxScanAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=2)
        result = adapter.iterate(num_batches=4)
        assert result.num_elements == 4 * nlp1_small_config.batch_size
        adapter.teardown()

    def test_warmup_primes_scan_body_cache(self, nlp1_small_config, small_token_data) -> None:
        """After warmup, Pipeline.scan should hit cache on the next call."""
        adapter = DataraxScanAdapter()
        adapter.setup(nlp1_small_config, small_token_data)
        adapter.warmup(num_batches=4)
        cache_size_after_warmup = len(adapter._pipeline._scan_body_cache)
        assert cache_size_after_warmup >= 1

        adapter.iterate(num_batches=4)
        # Same (step_fn, length) — cache should not grow.
        assert len(adapter._pipeline._scan_body_cache) == cache_size_after_warmup
        adapter.teardown()
