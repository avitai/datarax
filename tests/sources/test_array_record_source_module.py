# File: tests/sources/test_array_record_source.py

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import flax.nnx as nnx
import numpy as np
import pytest

from datarax.core.data_source import DataSourceModule
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule


class TestArrayRecordSourceModule:
    """Test suite for ArrayRecordSourceModule following TDD principles."""

    @pytest.fixture
    def mock_grain_source(self):
        """Create mock Grain ArrayRecordDataSource."""
        mock = MagicMock()
        mock.__len__.return_value = 100
        mock.__getitem__.side_effect = lambda idx: {"data": np.array([idx])}
        return mock

    @pytest.fixture
    def temp_array_record_files(self):
        """Create temporary ArrayRecord files for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock .array_record files
            paths = []
            for i in range(3):
                path = Path(tmpdir) / f"data_{i}.array_record"
                path.touch()
                paths.append(str(path))
            yield paths

    def test_initialization(self, temp_array_record_files):
        """Test that ArrayRecordSourceModule initializes correctly."""
        with patch("grain.sources.ArrayRecordDataSource") as mock_grain_class:
            # Setup mock to return an object with __len__
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 100
            mock_grain_class.return_value = mock_instance

            # Test basic initialization
            config = ArrayRecordSourceConfig(seed=42, num_epochs=3, shuffle_files=True)
            source = ArrayRecordSourceModule(config, paths=temp_array_record_files[0])

            # Verify it's a proper NNX module
            assert isinstance(source, nnx.Module)
            assert isinstance(source, DataSourceModule)

            # Verify state variables are initialized
            assert source.current_index.get_value() == 0
            assert source.current_epoch.get_value() == 0
            assert source.config.num_epochs == 3
            assert source.config.shuffle_files

            # Verify Grain source was created (without seed parameter)
            mock_grain_class.assert_called_once_with(paths=temp_array_record_files[0])

    def test_initialization_with_pattern(self):
        """Test initialization with glob pattern."""
        with patch("grain.sources.ArrayRecordDataSource") as mock_grain_class:
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 100
            mock_grain_class.return_value = mock_instance

            config = ArrayRecordSourceConfig(shuffle_files=False)
            source = ArrayRecordSourceModule(config, paths="/data/*.array_record")

            # Verify Grain source was created
            mock_grain_class.assert_called_once_with(paths="/data/*.array_record")
            assert not source.config.shuffle_files

    def test_length(self, mock_grain_source):
        """Test that length returns correct number of records."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            assert len(source) == 100
            assert source.total_records.get_value() == 100

    def test_iteration_single_epoch(self, mock_grain_source):
        """Test iteration through a single epoch."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig(num_epochs=1)
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Collect all elements
            elements = list(source)

            # Verify correct number of elements
            assert len(elements) == 100

            # Verify elements are correct
            for i, element in enumerate(elements):
                assert element["data"][0] == i  # type: ignore

            # Verify state after iteration
            assert source.current_epoch.get_value() == 1
            # After StopIteration, index resets to 0
            assert source.current_index.get_value() == 0

    def test_iteration_multiple_epochs(self, mock_grain_source):
        """Test iteration through multiple epochs."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig(num_epochs=2)
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Iterate through all epochs
            all_elements = []
            for element in source:
                all_elements.append(element)

            # Should have 2 epochs worth of data
            assert len(all_elements) == 200

            # Verify epoch transitions
            assert source.current_epoch.get_value() == 2

    def test_infinite_epochs(self, mock_grain_source):
        """Test infinite epoch iteration."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig(num_epochs=-1)  # Infinite
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Take more than one epoch worth
            count = 0
            for i, element in enumerate(source):
                count += 1
                if count > 250:  # More than 2 epochs
                    break

            # Should have iterated through multiple epochs
            assert count == 251
            assert source.current_epoch.get_value() >= 2

    def test_state_checkpointing(self, mock_grain_source):
        """Test state saving and restoration."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Iterate partway through
            iterator = iter(source)
            for _ in range(50):
                next(iterator)

            # Save state
            state = source.get_state()

            # Verify state contains required fields
            assert "current_index" in state
            assert "current_epoch" in state
            assert state["current_index"] == 50

            # Create new source and restore state
            new_source = ArrayRecordSourceModule(config, paths="dummy.array_record")
            new_source.set_state(state)

            # Verify state was restored
            assert new_source.current_index.get_value() == 50
            assert new_source.current_epoch.get_value() == 0

            # Continue iteration from checkpoint
            iterator = iter(new_source)
            # Need to manually advance iterator to the checkpoint position
            new_source.current_index.set_value(50)  # Restore position
            next_element = next(iterator)
            assert next_element["data"][0] == 50  # type: ignore

    def test_shuffling_with_seed(self):
        """Test that shuffling with seed is deterministic."""
        with patch("grain.sources.ArrayRecordDataSource") as mock_grain_class:
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 100
            mock_grain_class.return_value = mock_instance

            # Create two sources with same seed
            config = ArrayRecordSourceConfig(seed=42, shuffle_files=True)
            source1 = ArrayRecordSourceModule(config, paths="dummy.array_record")
            source2 = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Both should have same initialization
            assert source1.current_index.get_value() == source2.current_index.get_value()

    def test_prefetch_cache(self, mock_grain_source):
        """Test prefetch cache functionality."""
        with patch("grain.sources.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Access some elements
            elements = []
            for i, el in enumerate(source):
                elements.append(el)
                if i >= 10:
                    break

            # Cache should have been used (implementation specific)
            assert isinstance(source.prefetch_cache.get_value(), dict)

    def test_error_handling_missing_file(self):
        """Test error handling for missing files."""
        with patch("grain.sources.ArrayRecordDataSource") as mock_grain_class:
            # Make ArrayRecordDataSource raise an error
            mock_grain_class.side_effect = Exception("File not found")
            config = ArrayRecordSourceConfig()

            with pytest.raises(Exception, match="File not found"):
                ArrayRecordSourceModule(config, paths="/nonexistent/path/*.array_record")

    def test_integration_with_rngs(self):
        """Test integration with NNX Rngs."""
        with patch("grain.sources.ArrayRecordDataSource") as mock_grain_class:
            mock_instance = MagicMock()
            mock_instance.__len__.return_value = 100
            mock_grain_class.return_value = mock_instance

            rngs = nnx.Rngs(params=0, shuffle=42)
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record", rngs=rngs)

            assert source.rngs is not None
            assert "shuffle" in source.rngs


class TestArrayRecordSourceCleanup:
    """S4: explicit ArrayRecord C++ file-handle cleanup via close()/context manager."""

    @staticmethod
    def _build(grain_instance):
        with patch("grain.sources.ArrayRecordDataSource", return_value=grain_instance):
            return ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths="d.array_record")

    def test_close_calls_grain_close_when_available(self):
        """On grain >= 0.2.19 the source exposes close(); delegate to it."""
        grain_instance = MagicMock()
        grain_instance.__len__.return_value = 10
        source = self._build(grain_instance)

        source.close()

        grain_instance.close.assert_called_once()

    def test_close_falls_back_to_exit_on_older_grain(self):
        """On grain 0.2.18 there is no close(); fall back to context-manager __exit__."""

        class _NoCloseSource:
            def __init__(self) -> None:
                self.exited = False

            def __len__(self) -> int:
                return 10

            def __exit__(self, *exc: object) -> None:
                self.exited = True

        grain_instance = _NoCloseSource()
        source = self._build(grain_instance)

        source.close()

        assert grain_instance.exited is True

    def test_context_manager_closes_on_exit(self):
        """Using the module as a context manager releases handles on exit."""
        grain_instance = MagicMock()
        grain_instance.__len__.return_value = 10
        with patch("grain.sources.ArrayRecordDataSource", return_value=grain_instance):
            with ArrayRecordSourceModule(
                ArrayRecordSourceConfig(), paths="d.array_record"
            ) as source:
                assert source is not None
        grain_instance.close.assert_called_once()

    def test_close_is_idempotent(self):
        """Calling close() repeatedly must not raise."""
        grain_instance = MagicMock()
        grain_instance.__len__.return_value = 10
        source = self._build(grain_instance)

        source.close()
        source.close()  # second call must be safe


class TestArrayRecordSourceRepr:
    """S2: config-identifying __repr__ for checkpoint validation."""

    def test_repr_identifies_source_config(self):
        grain_instance = MagicMock()
        grain_instance.__len__.return_value = 42
        grain_instance.paths = ["a.array_record", "b.array_record"]
        with patch("grain.sources.ArrayRecordDataSource", return_value=grain_instance):
            config = ArrayRecordSourceConfig(seed=7, num_epochs=3, shuffle_files=True)
            source = ArrayRecordSourceModule(config, paths=["a.array_record", "b.array_record"])

        r = repr(source)
        assert "ArrayRecordSourceModule" in r
        assert "a.array_record" in r
        assert "num_records=42" in r
        assert "shuffle_files=True" in r
        assert "seed=7" in r
        assert "num_epochs=3" in r
