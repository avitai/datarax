"""Tests for the Prefetcher class."""

import time
import unittest

from datarax.control.prefetcher import Prefetcher


class TestPrefetcher(unittest.TestCase):
    """Tests for the Prefetcher class."""

    def test_basic_prefetching(self):
        """Test basic prefetching functionality."""
        prefetcher = Prefetcher(buffer_size=2)
        data = list(range(10))
        result = list(prefetcher.prefetch(iter(data)))
        self.assertEqual(result, data)

    def test_empty_iterator(self):
        """Test prefetching from an empty iterator."""
        prefetcher = Prefetcher(buffer_size=2)
        result = list(prefetcher.prefetch(iter([])))
        self.assertEqual(result, [])

    def test_single_item(self):
        """Test prefetching a single item."""
        prefetcher = Prefetcher(buffer_size=1)
        result = list(prefetcher.prefetch(iter([42])))
        self.assertEqual(result, [42])

    def test_buffer_size_one(self):
        """Test prefetching with buffer size of 1."""
        prefetcher = Prefetcher(buffer_size=1)
        data = list(range(5))
        result = list(prefetcher.prefetch(iter(data)))
        self.assertEqual(result, data)

    def test_buffer_size_large(self):
        """Test prefetching with large buffer size."""
        prefetcher = Prefetcher(buffer_size=100)
        data = list(range(50))
        result = list(prefetcher.prefetch(iter(data)))
        self.assertEqual(result, data)

    def test_slow_iterator(self):
        """Test prefetching with a slow iterator."""

        def slow_generator():
            for i in range(5):
                time.sleep(0.01)  # Simulate slow data generation
                yield i

        prefetcher = Prefetcher(buffer_size=3)
        result = list(prefetcher.prefetch(slow_generator()))
        self.assertEqual(result, [0, 1, 2, 3, 4])

    def test_exception_in_iterator(self):
        """Test that exceptions from the source iterator are propagated."""

        def error_generator():
            yield 1
            yield 2
            raise ValueError("Test error")

        prefetcher = Prefetcher(buffer_size=2)
        iterator = prefetcher.prefetch(error_generator())

        # Should yield the first two items
        self.assertEqual(next(iterator), 1)
        self.assertEqual(next(iterator), 2)

        # Should raise the exception
        with self.assertRaises(ValueError) as ctx:
            next(iterator)
        self.assertIn("Test error", str(ctx.exception))

    def test_early_termination(self):
        """Test that prefetching stops cleanly when iterator is not fully consumed."""

        def long_generator():
            yield from range(1000)

        prefetcher = Prefetcher(buffer_size=5)
        iterator = prefetcher.prefetch(long_generator())

        # Only consume first 3 items
        result = [next(iterator) for _ in range(3)]
        self.assertEqual(result, [0, 1, 2])

        # Iterator should be cleanly terminated when it goes out of scope
        # (tested by not consuming remaining items)

    def test_multiple_prefetch_instances(self):
        """Test using multiple prefetcher instances simultaneously."""
        prefetcher1 = Prefetcher(buffer_size=2)
        prefetcher2 = Prefetcher(buffer_size=3)

        data1 = list(range(5))
        data2 = list(range(10, 15))

        result1 = list(prefetcher1.prefetch(iter(data1)))
        result2 = list(prefetcher2.prefetch(iter(data2)))

        self.assertEqual(result1, data1)
        self.assertEqual(result2, data2)

    def test_prefetcher_initialization(self):
        """Test Prefetcher initialization with different buffer sizes."""
        prefetcher = Prefetcher(buffer_size=5)
        self.assertEqual(prefetcher.buffer_size, 5)

        prefetcher_default = Prefetcher()
        self.assertEqual(prefetcher_default.buffer_size, 2)

    def test_string_data(self):
        """Test prefetching with non-numeric data."""
        prefetcher = Prefetcher(buffer_size=2)
        data = ["apple", "banana", "cherry", "date"]
        result = list(prefetcher.prefetch(iter(data)))
        self.assertEqual(result, data)

    def test_dict_data(self):
        """Test prefetching with dictionary data."""
        prefetcher = Prefetcher(buffer_size=2)
        data = [{"id": i, "value": i * 2} for i in range(5)]
        result = list(prefetcher.prefetch(iter(data)))
        self.assertEqual(result, data)


if __name__ == "__main__":
    unittest.main()
