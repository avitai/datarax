"""Tests for the CachingIterator (iteration-boundary cache)."""

from datarax.pipeline.nodes.cache import CachingIterator


class TestCachingIterator:
    """CachingIterator replays cached elements without recomputing them."""

    def test_replays_from_cache_without_recomputing(self):
        calls: list[int] = []

        def source():
            for i in range(3):
                calls.append(i)
                yield i

        cached = CachingIterator(source())
        assert list(cached) == [0, 1, 2]
        assert calls == [0, 1, 2]
        # Second pass is served entirely from cache; the source is not re-invoked.
        assert list(cached) == [0, 1, 2]
        assert calls == [0, 1, 2]

    def test_serves_cached_then_computes_new_positions(self):
        cached = CachingIterator(iter([10, 20, 30]))
        assert next(cached) == 10
        assert next(cached) == 20

        restarted = iter(cached)
        assert next(restarted) == 10  # cached
        assert next(restarted) == 20  # cached
        assert next(restarted) == 30  # computed from source, now cached

    def test_propagates_stop_iteration(self):
        cached = CachingIterator(iter([1]))
        assert next(cached) == 1
        try:
            next(cached)
            raise AssertionError("expected StopIteration")
        except StopIteration:
            pass
