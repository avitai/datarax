"""Tests for TimingCollector and TimingSample.

TDD tests written first per Section 6.2.3 of the benchmark report.
Verifies: dataclass creation, measure_iteration() behavior,
sync_fn calling, first_batch_time capture, perf_counter usage,
and edge cases (empty iterator, short iterator).
"""

from unittest.mock import MagicMock, patch

from datarax.benchmarking.timing import TimingCollector, TimingSample


class TestTimingSample:
    """Tests for TimingSample dataclass."""

    def test_creation_with_all_fields(self):
        sample = TimingSample(
            wall_clock_sec=1.5,
            per_batch_times=[0.1, 0.2, 0.3],
            first_batch_time=0.15,
            num_batches=3,
            num_elements=96,
        )
        assert sample.wall_clock_sec == 1.5
        assert sample.per_batch_times == [0.1, 0.2, 0.3]
        assert sample.first_batch_time == 0.15
        assert sample.num_batches == 3
        assert sample.num_elements == 96

    def test_field_access(self):
        sample = TimingSample(
            wall_clock_sec=2.0,
            per_batch_times=[0.5, 0.5],
            first_batch_time=0.6,
            num_batches=2,
            num_elements=64,
        )
        assert len(sample.per_batch_times) == sample.num_batches


class TestTimingCollector:
    """Tests for TimingCollector."""

    def test_measure_simple_iterator(self):
        """measure_iteration() consumes batches and returns TimingSample."""
        data = [list(range(10)) for _ in range(5)]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert isinstance(result, TimingSample)
        assert result.num_batches == 5
        assert len(result.per_batch_times) == 5
        assert result.wall_clock_sec > 0
        assert result.first_batch_time > 0

    def test_sync_fn_called_each_iteration(self):
        """sync_fn is called once per batch for GPU synchronization."""
        sync_fn = MagicMock()
        data = [1, 2, 3]
        collector = TimingCollector(sync_fn=sync_fn)

        collector.measure_iteration(iter(data), num_batches=3)

        assert sync_fn.call_count == 3

    def test_sync_fn_default_is_noop(self):
        """Default sync_fn doesn't raise and is a no-op."""
        collector = TimingCollector()
        data = [1, 2, 3]

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.num_batches == 3

    def test_first_batch_time_captured_separately(self):
        """first_batch_time captures time from start to end of first batch."""
        data = [1, 2, 3]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.first_batch_time > 0
        # first_batch_time should be >= first per-batch time (includes startup)
        assert result.first_batch_time >= result.per_batch_times[0]

    @patch("datarax.benchmarking.timing.time")
    def test_uses_perf_counter_not_time_time(self, mock_time):
        """Verify time.perf_counter() is used, not time.time()."""
        # Set up perf_counter to return incrementing values
        mock_time.perf_counter.side_effect = [
            0.0,  # overall_start
            0.1,  # batch_start (batch 0)
            0.2,  # batch_end (batch 0) / sync
            0.3,  # overall end
        ]
        data = [1]
        collector = TimingCollector()

        collector.measure_iteration(iter(data), num_batches=1)

        # perf_counter should have been called, time.time() should NOT
        assert mock_time.perf_counter.call_count >= 3
        mock_time.time.assert_not_called()

    def test_count_fn_counts_elements(self):
        """count_fn is used to count elements per batch."""
        batches = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        collector = TimingCollector()

        result = collector.measure_iteration(
            iter(batches),
            num_batches=3,
            count_fn=len,
        )

        assert result.num_elements == 9  # 3 + 2 + 4

    def test_default_count_fn_counts_one_per_batch(self):
        """Without count_fn, each batch counts as 1 element."""
        data = [1, 2, 3]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=3)

        assert result.num_elements == 3

    def test_empty_iterator(self):
        """Empty iterator produces zero-batch sample."""
        collector = TimingCollector()

        result = collector.measure_iteration(iter([]))

        assert result.num_batches == 0
        assert result.per_batch_times == []
        assert result.first_batch_time == 0.0
        assert result.num_elements == 0
        assert result.wall_clock_sec >= 0

    def test_iterator_shorter_than_requested(self):
        """When iterator has fewer items than num_batches, consume what's available."""
        data = [1, 2]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert result.num_batches == 2
        assert len(result.per_batch_times) == 2

    def test_num_batches_none_exhausts_iterator(self):
        """When num_batches is None, exhaust the entire iterator."""
        data = [1, 2, 3, 4, 5]
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=None)

        assert result.num_batches == 5

    def test_per_batch_times_all_positive(self):
        """Every per-batch time should be positive."""
        data = list(range(10))
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=10)

        assert all(t >= 0 for t in result.per_batch_times)

    def test_wall_clock_at_least_sum_of_batches(self):
        """Wall clock should be >= sum of per-batch times."""
        data = list(range(5))
        collector = TimingCollector()

        result = collector.measure_iteration(iter(data), num_batches=5)

        assert result.wall_clock_sec >= sum(result.per_batch_times)
