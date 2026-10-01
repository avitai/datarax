"""Tests for the eager source operations (datarax.sources.source_ops).

Tests the standalone composition helpers that both HFEagerSource and
TFDSEagerSource delegate to. These functions handle iteration,
batching, reset, config validation, and key filtering.
"""

import unittest
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from datarax.sources.source_ops import (
    eager_get_batch,
    eager_iter,
    eager_reset,
    filter_keys,
    validate_eager_config,
)


class TestEagerIter(unittest.TestCase):
    """Tests for eager_iter."""

    def _make_vars(self, index_val=0, epoch_val=0):
        """Create mock nnx.Variable-like objects."""
        index_var = MagicMock()
        index_var.get_value.return_value = index_val
        epoch_var = MagicMock()
        epoch_var.get_value.return_value = epoch_val
        return index_var, epoch_var

    def test_iterates_all_elements(self):
        """Should yield exactly `length` elements."""
        data = {"x": jnp.arange(5)}
        index_var, epoch_var = self._make_vars()
        # epoch_var.get_value needs to return updated value after set_value
        epoch_var.get_value.return_value = 1

        elements = list(
            eager_iter(
                data,
                length=5,
                index_var=index_var,
                epoch_var=epoch_var,
                build_element=lambda d, idx: {"x": d["x"][idx]},
            )
        )
        self.assertEqual(len(elements), 5)

    def test_resets_index_to_zero(self):
        """Should reset index to 0 at start of iteration."""
        data = {"x": jnp.arange(3)}
        index_var, epoch_var = self._make_vars(index_val=2, epoch_val=0)
        epoch_var.get_value.return_value = 1

        list(
            eager_iter(
                data,
                3,
                index_var,
                epoch_var,
                build_element=lambda d, idx: {"x": d["x"][idx]},
            )
        )
        index_var.set_value.assert_any_call(0)

    def test_increments_epoch(self):
        """Should increment epoch at start of iteration."""
        data = {"x": jnp.arange(3)}
        index_var, epoch_var = self._make_vars(epoch_val=0)
        # get_value returns 0 initially; set_value(0+1=1); get_value still returns 0
        # but we verify set_value was called with incremented value
        list(
            eager_iter(
                data,
                3,
                index_var,
                epoch_var,
                build_element=lambda d, idx: {"x": d["x"][idx]},
            )
        )
        epoch_var.set_value.assert_called_with(1)


class TestEagerGetBatch(unittest.TestCase):
    """Tests for eager_get_batch."""

    def _make_vars(self, index_val=0, epoch_val=0):
        index_var = MagicMock()
        index_var.get_value.return_value = index_val
        epoch_var = MagicMock()
        epoch_var.get_value.return_value = epoch_val
        return index_var, epoch_var

    def test_stateless_mode_returns_the_first_records(self):
        """Stateless mode returns the first batch_size elements."""
        data = {"x": jnp.arange(10)}
        index_var, epoch_var = self._make_vars()
        key = jax.random.key(0)

        batch = eager_get_batch(
            data,
            10,
            index_var,
            epoch_var,
            batch_size=4,
            key=key,
            gather_fn=lambda d, idx: {"x": d["x"][idx]},
        )
        self.assertTrue(jnp.array_equal(batch["x"], jnp.arange(4)))

    def test_stateful_mode_advances_index(self):
        """Without a key, should advance internal index."""
        data = {"x": jnp.arange(10)}
        index_var, epoch_var = self._make_vars(index_val=0, epoch_val=0)

        eager_get_batch(
            data,
            10,
            index_var,
            epoch_var,
            batch_size=3,
            key=None,
            gather_fn=lambda d, idx: {"x": d["x"][idx]},
        )
        index_var.set_value.assert_called_with(3)

    def test_stateful_epoch_wraps(self):
        """When index reaches end, epoch should increment."""
        data = {"x": jnp.arange(5)}
        index_var, epoch_var = self._make_vars(index_val=3, epoch_val=0)

        eager_get_batch(
            data,
            5,
            index_var,
            epoch_var,
            batch_size=3,
            key=None,
            gather_fn=lambda d, idx: {"x": d["x"][idx]},
        )
        # end = min(3+3, 5) = 5, wraps to 0
        index_var.set_value.assert_called_with(0)
        epoch_var.set_value.assert_called_with(1)


class TestEagerReset(unittest.TestCase):
    """Tests for eager_reset."""

    def test_resets_index_and_epoch(self):
        index_var = MagicMock()
        epoch_var = MagicMock()
        eager_reset(index_var, epoch_var)
        index_var.set_value.assert_called_with(0)
        epoch_var.set_value.assert_called_with(0)


class TestValidateEagerConfig(unittest.TestCase):
    """Tests for validate_eager_config."""

    def test_valid_config(self):
        """No error for valid name + split."""
        validate_eager_config("mnist", "train", None, None, "TestConfig")

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError, msg="name is required"):
            validate_eager_config(None, "train", None, None, "TestConfig")

    def test_missing_split_raises(self):
        with self.assertRaises(ValueError, msg="split is required"):
            validate_eager_config("mnist", None, None, None, "TestConfig")

    def test_both_keys_raises(self):
        with self.assertRaises(ValueError, msg="Cannot specify both"):
            validate_eager_config("mnist", "train", {"a"}, {"b"}, "TestConfig")

    def test_include_keys_only_ok(self):
        validate_eager_config("mnist", "train", {"image"}, None, "TestConfig")

    def test_exclude_keys_only_ok(self):
        validate_eager_config("mnist", "train", None, {"label"}, "TestConfig")

    # --- try_gcs / data_dir validation ---

    def test_try_gcs_false_with_data_dir_ok(self):
        """try_gcs=False (default) allows data_dir."""
        validate_eager_config(
            "mnist", "train", None, None, "TestConfig", try_gcs=False, data_dir="/tmp/data"
        )

    def test_try_gcs_true_without_data_dir_ok(self):
        """try_gcs=True without data_dir is valid."""
        validate_eager_config("mnist", "train", None, None, "TestConfig", try_gcs=True)

    def test_try_gcs_true_with_data_dir_raises(self):
        """try_gcs=True + data_dir raises ValueError."""
        with self.assertRaises(ValueError, msg="Cannot specify both try_gcs=True and data_dir"):
            validate_eager_config(
                "mnist", "train", None, None, "TestConfig", try_gcs=True, data_dir="/tmp/data"
            )

    def test_try_gcs_true_with_data_dir_error_includes_class_name(self):
        """Error message includes the config class name."""
        with self.assertRaises(ValueError) as ctx:
            validate_eager_config(
                "mnist", "train", None, None, "MyConfig", try_gcs=True, data_dir="/data"
            )
        self.assertIn("MyConfig", str(ctx.exception))

    def test_defaults_are_explicitly_supported(self):
        """Calling without try_gcs/data_dir uses explicit default behavior."""
        validate_eager_config("mnist", "train", None, None, "TestConfig")


class TestFilterKeys(unittest.TestCase):
    """Tests for filter_keys."""

    def test_no_filter(self):
        """No include/exclude returns element unchanged."""
        elem = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(filter_keys(elem, None, None), elem)

    def test_include_keys(self):
        elem = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(filter_keys(elem, {"a", "c"}, None), {"a": 1, "c": 3})

    def test_exclude_keys(self):
        elem = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(filter_keys(elem, None, {"b"}), {"a": 1, "c": 3})

    def test_include_missing_key_ignored(self):
        elem = {"a": 1, "b": 2}
        self.assertEqual(filter_keys(elem, {"a", "z"}, None), {"a": 1})


if __name__ == "__main__":
    unittest.main()


class TestPublicSurface(unittest.TestCase):
    """The helpers are exported from ``datarax.sources``; the private path is gone."""

    def test_sources_package_exports_the_helpers(self) -> None:
        import datarax.sources as sources

        for name in ("eager_iter", "eager_get_batch", "eager_reset", "resolve_wrapped_indices"):
            self.assertIn(name, sources.__all__)
            self.assertTrue(callable(getattr(sources, name)))

    def test_private_module_path_is_gone(self) -> None:
        import importlib

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("datarax.sources._eager_source_ops")
