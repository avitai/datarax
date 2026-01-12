"""Tests for typing module functionality.

This module tests the protocols defined in the Datarax typing system.
"""

from typing import Any

import pytest

from datarax.typing import Checkpointable, CheckpointableIterator


class TestCheckpointableProtocol:
    """Test Checkpointable protocol."""

    def test_checkpointable_implementation(self):
        """Test that a class correctly implements Checkpointable."""

        class ValidCheckpointable:
            def get_state(self) -> dict[str, Any]:
                return {"a": 1}

            def set_state(self, state: dict[str, Any]) -> None:
                pass

        obj = ValidCheckpointable()
        assert isinstance(obj, Checkpointable)

    def test_missing_methods(self):
        """Test that missing methods fail protocol check."""

        class MissingSetState:
            def get_state(self) -> dict[str, Any]:
                return {}

        class MissingGetState:
            def set_state(self, state: dict[str, Any]) -> None:
                pass

        assert not isinstance(MissingSetState(), Checkpointable)
        assert not isinstance(MissingGetState(), Checkpointable)


class TestCheckpointableIteratorProtocol:
    """Test CheckpointableIterator protocol."""

    def test_checkpointable_iterator_implementation(self):
        """Test that a class correctly implements CheckpointableIterator."""

        class ValidIterator:
            def __iter__(self):
                return self

            def __next__(self):
                return 1

            def get_state(self) -> dict[str, Any]:
                return {}

            def set_state(self, state: dict[str, Any]) -> None:
                pass

        obj = ValidIterator()
        assert isinstance(obj, CheckpointableIterator)
        # Should also be Checkpointable
        assert isinstance(obj, Checkpointable)

    def test_missing_iterator_methods(self):
        """Test that missing iterator methods fail protocol check."""

        class MissingIter:
            def __next__(self):
                return 1

            def get_state(self) -> dict[str, Any]:
                return {}

            def set_state(self, state: dict[str, Any]) -> None:
                pass

        assert not isinstance(MissingIter(), CheckpointableIterator)


if __name__ == "__main__":
    pytest.main([__file__])
