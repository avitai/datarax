"""Tests for BatchMixOperator - MixUp and CutMix batch augmentation.

This module tests the BatchMixOperator, which performs batch-level sample mixing:
- MixUp: Linear interpolation between pairs of samples
- CutMix: Cut and paste patches between images

Test Coverage:
- Config validation (mode, alpha, field names)
- MixUp mode (linear interpolation, batch mixing)
- CutMix mode (patch mixing; the fraction kept as lambda)
- Labels untouched; partner and lambda written for the loss
- Batch size edge cases (single element, empty batch)
- JAX compatibility (JIT, reproducibility)
- Stochastic behavior (different keys, deterministic with same seed)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core import batch_ops
from datarax.core.element_batch import Batch, Element
from datarax.core.state_keys import MIX_LAMBDA, MIX_PARTNER
from datarax.operators.batch_mix_operator import (
    BatchMixOperator,
    BatchMixOperatorConfig,
)
from datarax.pipeline.dag import name_records


class TestBatchMixOperatorConfig:
    """Test BatchMixOperatorConfig validation and initialization."""

    def test_config_requires_valid_mode(self):
        """Mode must be 'mixup' or 'cutmix'."""
        with pytest.raises(ValueError, match="mode must be"):
            BatchMixOperatorConfig(mode="invalid")

    def test_config_mixup_mode_valid(self):
        """Verify mixup mode is accepted."""
        config = BatchMixOperatorConfig(mode="mixup")
        assert config.mode == "mixup"
        assert config.alpha == 1.0  # default

    def test_config_cutmix_mode_valid(self):
        """Verify cutmix mode is accepted."""
        config = BatchMixOperatorConfig(mode="cutmix")
        assert config.mode == "cutmix"

    def test_config_alpha_must_be_positive(self):
        """Alpha parameter must be positive."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            BatchMixOperatorConfig(mode="mixup", alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            BatchMixOperatorConfig(mode="mixup", alpha=-1.0)

    def test_config_custom_alpha(self):
        """Custom alpha should be accepted."""
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.5)
        assert config.alpha == 0.5

    def test_config_is_always_stochastic(self):
        """Verify the operator is always stochastic (uses random mixing)."""
        config = BatchMixOperatorConfig(mode="mixup")
        assert config.stochastic is True
        assert config.stream_name == "batch_mix"

    def test_config_custom_stream_name(self):
        """Custom stream name should be accepted."""
        config = BatchMixOperatorConfig(mode="mixup", stream_name="my_mixer")
        assert config.stream_name == "my_mixer"

    def test_config_custom_data_field(self):
        """Custom data field should be accepted."""
        config = BatchMixOperatorConfig(mode="cutmix", data_field="pixels")
        assert config.data_field == "pixels"


class TestBatchMixOperatorInit:
    """Test BatchMixOperator initialization."""

    def test_init_mixup_mode(self):
        """Initialize with MixUp mode."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)

        assert op.config.mode == "mixup"
        assert op.config.alpha == 1.0

    def test_init_cutmix_mode(self):
        """Initialize with CutMix mode."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)

        assert op.config.mode == "cutmix"

    def test_init_requires_rngs(self):
        """Verify the operator requires rngs (always stochastic)."""
        config = BatchMixOperatorConfig(mode="mixup")

        with pytest.raises(ValueError, match="require.*rngs"):
            BatchMixOperator(config, rngs=None)


class TestBatchMixOperatorMixUp:
    """Test MixUp mode functionality."""

    def test_mixup_produces_mixed_values(self):
        """Verify mixup produces linear combinations of samples."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=1.0, data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        # Create batch with distinct values
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([10.0])}),
                    Element(data={"value": jnp.array([20.0])}),
                    Element(data={"value": jnp.array([30.0])}),
                ]
            )
        )

        result = op(batch)
        result_data = result.data

        # Results should be in the range [0, 30] (linear combinations)
        assert jnp.all(result_data["value"] >= 0.0)
        assert jnp.all(result_data["value"] <= 30.0)
        # At least some values should be mixed (not exactly 0, 10, 20, 30)
        # This is probabilistic but alpha=1.0 makes pure originals unlikely

    def test_mixup_preserves_shape(self):
        """Verify mixup preserves batch shape."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", data_field="arr")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"arr": jnp.ones((3, 4))}),
                    Element(data={"arr": jnp.zeros((3, 4))}),
                ]
            )
        )

        result = op(batch)
        assert result.data["arr"].shape == (2, 3, 4)

    def test_mixup_mixes_only_its_data_field(self):
        """Every other field, labels included, is left as it was."""
        rngs = nnx.Rngs({"batch_mix": 42})
        op = BatchMixOperator(BatchMixOperatorConfig(mode="mixup", data_field="x"), rngs=rngs)
        batch = batch_ops.from_arrays(
            {
                "x": jnp.array([[0.0], [10.0], [20.0], [30.0]]),
                "y": jnp.array([[1.0], [2.0], [3.0], [4.0]]),
            }
        )

        result = op(batch)

        assert jnp.array_equal(result.data["y"], batch.data["y"])
        assert not jnp.array_equal(result.data["x"], batch.data["x"])

    def test_mixup_raw_path_preserves_batch_shape(self):
        """Verify DAG fused raw-batch path uses batch-level MixUp."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        batch_data = {
            "image": jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2),
            "label": jax.nn.one_hot(jnp.arange(4), 4),
        }

        out = op(batch_ops.from_arrays(batch_data))
        result_data, result_states = out.data, out.states

        assert set(result_states) == {MIX_PARTNER}
        assert result_data["image"].shape == batch_data["image"].shape
        assert result_data["label"].shape == batch_data["label"].shape

    def test_mixup_single_element_unchanged(self):
        """Verify mixup with single element returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([5.0])})]))

        result = op(batch)
        assert jnp.allclose(result.data["value"], jnp.array([[5.0]]))


class TestBatchMixOperatorCutMix:
    """Test CutMix mode functionality."""

    def test_cutmix_produces_patched_images(self):
        """Verify cutmix cuts and pastes patches between images."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", data_field="image")
        op = BatchMixOperator(config, rngs=rngs)

        # Create batch with distinct images (white and black)
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"image": jnp.ones((32, 32, 3))}),
                    Element(data={"image": jnp.zeros((32, 32, 3))}),
                ]
            )
        )

        result = op(batch)
        result_data = result.data

        # Result should contain both 0s and 1s (mixed patches)
        assert result_data["image"].shape == (2, 32, 32, 3)
        has_ones = jnp.any(result_data["image"] == 1.0)
        has_zeros = jnp.any(result_data["image"] == 0.0)
        assert has_ones and has_zeros, "CutMix should produce patched images"

    def test_cutmix_preserves_shape(self):
        """Verify cutmix preserves image shape."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"image": jnp.ones((64, 64, 3))}),
                    Element(data={"image": jnp.zeros((64, 64, 3))}),
                ]
            )
        )

        result = op(batch)
        assert result.data["image"].shape == (2, 64, 64, 3)

    def test_cutmix_raw_path_preserves_batch_shape(self):
        """Verify DAG fused raw-batch path uses batch-level CutMix."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", data_field="image")
        op = BatchMixOperator(config, rngs=rngs)

        batch_data = {
            "image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32),
            "label": jax.nn.one_hot(jnp.arange(4), 4),
        }

        out = op(batch_ops.from_arrays(batch_data))
        result_data, result_states = out.data, out.states

        assert set(result_states) == {MIX_PARTNER}
        assert result_data["image"].shape == batch_data["image"].shape
        assert result_data["label"].shape == batch_data["label"].shape

    def test_cutmix_single_element_unchanged(self):
        """Verify cutmix with single element returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        original_image = jnp.ones((32, 32, 3)) * 0.5
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"image": original_image})]))

        result = op(batch)
        assert jnp.allclose(result.data["image"], original_image[None, ...])

    def test_a_batch_without_the_data_field_is_refused(self):
        """A misspelt or absent field is an error, never a batch passed through unmixed."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", data_field="image")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_arrays({"value": jnp.array([[1.0], [2.0]])})

        with pytest.raises(ValueError, match="data\\['image'\\].*lacks: \\['value'\\]"):
            op(batch)

    def test_cutmix_refuses_a_field_that_is_not_a_batch_of_images(self):
        """CutMix pastes boxes into (B, H, W, C) images; any other shape is refused."""
        rngs = nnx.Rngs({"batch_mix": 42})
        op = BatchMixOperator(BatchMixOperatorConfig(mode="cutmix"), rngs=rngs)

        batch = batch_ops.from_arrays({"image": jnp.array([[1.0, 2.0], [3.0, 4.0]])})

        with pytest.raises(ValueError, match="B, H, W, C"):
            op(batch)


class TestBatchMixOperatorStochastic:
    """Test stochastic behavior of BatchMixOperator."""

    def test_different_keys_produce_different_results(self):
        """Different RNG keys should produce different mixing."""
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")

        results = set()
        for seed in range(20):
            rngs = nnx.Rngs({"batch_mix": seed})
            op = BatchMixOperator(config, rngs=rngs)

            batch = batch_ops.from_stacked(
                batch_ops.stack(
                    [
                        Element(data={"value": jnp.array([0.0])}),
                        Element(data={"value": jnp.array([100.0])}),
                    ]
                )
            )

            result = op(batch)
            # Round to avoid floating point precision issues
            val = round(float(result.data["value"][0, 0]), 2)
            results.add(val)

        # Should see variation in results
        assert len(results) > 1, "Different seeds should produce different mixing"

    def test_same_key_produces_same_result(self):
        """Same RNG key should produce identical mixing."""
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")

        rngs1 = nnx.Rngs({"batch_mix": 42})
        rngs2 = nnx.Rngs({"batch_mix": 42})

        op1 = BatchMixOperator(config, rngs=rngs1)
        op2 = BatchMixOperator(config, rngs=rngs2)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([100.0])}),
                ]
            )
        )

        result1 = op1(batch)
        result2 = op2(batch)

        assert jnp.allclose(result1.data["value"], result2.data["value"])

    def test_a_direct_call_mixes_the_same_batch_the_same_way(self):
        """Without record indices the batch is keyed on its first position, with no epoch."""
        operator = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", data_field="value"),
            rngs=nnx.Rngs({"batch_mix": 0}),
        )

        values = {"value": jnp.arange(8.0).reshape(4, 2)}
        rows = jnp.arange(4, dtype=jnp.uint32)
        named = batch_ops.from_arrays(values).replace(
            indices=jnp.stack([jnp.zeros_like(rows), rows], -1)
        )

        direct = operator(batch_ops.from_arrays(values))

        assert jnp.array_equal(direct["value"], operator(batch_ops.from_arrays(values))["value"])
        assert jnp.array_equal(direct["value"], operator(named)["value"])

    def test_a_batch_crossing_epochs_is_keyed_on_its_first_record_s_epoch(self):
        """A boundary batch carries each record's epoch; the batch key takes the first's."""
        operator = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", data_field="value"),
            rngs=nnx.Rngs({"batch_mix": 0}),
        )
        rows = jnp.array([8, 9, 0, 1], jnp.uint32)
        batch = batch_ops.from_arrays({"value": jnp.arange(8.0).reshape(4, 2)}).replace(
            indices=jnp.stack([jnp.zeros_like(rows), rows], -1)
        )

        crossing = operator(batch.replace(epochs=jnp.array([3, 3, 4, 4], jnp.int32)))
        first = operator(batch.replace(epochs=jnp.full(4, 3, jnp.int32)))
        other = operator(batch.replace(epochs=jnp.full(4, 4, jnp.int32)))

        assert jnp.array_equal(crossing["value"], first["value"])
        assert not jnp.array_equal(crossing["value"], other["value"])


class TestBatchMixOperatorJAX:
    """Test JAX compatibility of BatchMixOperator."""

    def test_jit_compilation_mixup(self):
        """Verify mixup works with JIT compilation."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([10.0])}),
                ]
            )
        )

        result = apply_op(op, batch)
        assert result.data["value"].shape == (2, 1)

    def test_jit_compilation_cutmix(self):
        """Verify cutmix works with JIT compilation."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"image": jnp.ones((32, 32, 3))}),
                    Element(data={"image": jnp.zeros((32, 32, 3))}),
                ]
            )
        )

        result = apply_op(op, batch)
        assert result.data["image"].shape == (2, 32, 32, 3)

    def test_jit_preserves_randomness(self):
        """JIT should preserve random behavior across calls."""
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([100.0])}),
                ]
            )
        )

        results = set()
        for seed in range(20):
            rngs = nnx.Rngs({"batch_mix": seed})
            op = BatchMixOperator(config, rngs=rngs)
            result = apply_op(op, batch)
            val = round(float(result.data["value"][0, 0]), 2)
            results.add(val)

        assert len(results) > 1, "JIT should preserve randomness"


class TestBatchMixOperatorDifferentiability:
    """Test gradient flow through MixUp batch mixing."""

    def test_mixup_is_differentiable_wrt_inputs(self):
        """MixUp path should produce finite gradients for input arrays."""
        config = BatchMixOperatorConfig(mode="mixup", alpha=1.0, data_field="value")
        op = BatchMixOperator(config, rngs=nnx.Rngs({"batch_mix": 0}))

        def loss(values):
            return jnp.sum(op(batch_ops.from_arrays({"value": values}))["value"])

        inputs = jnp.array([[1.0], [3.0]], dtype=jnp.float32)
        grad = jax.grad(loss)(inputs)

        assert jnp.all(jnp.isfinite(grad))
        assert jnp.allclose(grad, jnp.ones_like(inputs))


class TestBatchMixOperatorEdgeCases:
    """Test edge cases for BatchMixOperator."""

    def test_empty_batch_handling(self):
        """Empty batch should be handled gracefully."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_arrays({"value": jnp.zeros((0, 1))})

        result = op(batch)
        assert result.batch_size == 0

    def test_large_alpha_parameter(self):
        """Large alpha should produce more uniform mixing ratios."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=10.0, data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([100.0])}),
                ]
            )
        )

        result = op(batch)
        # With large alpha, lambda tends toward 0.5, so values near 50
        result_values = result.data["value"]
        # Just verify it runs without error and produces valid output
        assert jnp.all(result_values >= 0.0)
        assert jnp.all(result_values <= 100.0)

    def test_small_alpha_parameter(self):
        """Small alpha should produce more extreme mixing ratios."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.1, data_field="value")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([100.0])}),
                ]
            )
        )

        result = op(batch)
        # With small alpha, values tend toward extremes (0 or 100)
        result_values = result.data["value"]
        assert jnp.all(result_values >= 0.0)
        assert jnp.all(result_values <= 100.0)

    def test_it_has_no_per_record_form(self):
        """Mixing needs the whole batch, so the per-record form is refused."""
        rngs = nnx.Rngs({"batch_mix": 42})
        op = BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=rngs)

        assert not op.has_record_form
        with pytest.raises(TypeError, match="apply_batch"):
            op.apply_record(Element({"image": jnp.ones((2, 2, 1))}, index=jnp.zeros(2, jnp.uint32)))

    def test_cutmix_grayscale_image(self):
        """Verify cutmix works with grayscale images (H, W, 1)."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"image": jnp.ones((32, 32, 1))}),
                    Element(data={"image": jnp.zeros((32, 32, 1))}),
                ]
            )
        )

        result = op(batch)
        assert result.data["image"].shape == (2, 32, 32, 1)


def test_cutmix_refuses_data_that_is_not_a_mapping():
    """CutMix pastes into a named image field; a batch whose data is one array has none."""
    mixer = BatchMixOperator(
        BatchMixOperatorConfig(mode="cutmix", data_field="image"), rngs=nnx.Rngs(batch_mix=0)
    )
    batch = batch_ops.from_arrays(jnp.arange(4 * 8 * 8 * 3, dtype=jnp.float32).reshape(4, 8, 8, 3))

    with pytest.raises(ValueError, match="lacks"):
        mixer(batch)


class TestBatchMixOperatorInAPipeline:
    """BatchMixOperator as a pipeline stage and on a batch named by its records."""

    def test_batch_mix_runs_through_pipeline_step(self):
        from datarax.pipeline import Pipeline
        from datarax.sources.memory_source import MemorySource, MemorySourceConfig

        data = {"image": jnp.arange(8 * 4, dtype=jnp.float32).reshape(8, 4)}
        source = MemorySource(MemorySourceConfig(shuffle=False), data)
        mixer = BatchMixOperator(
            BatchMixOperatorConfig(mode="mixup", data_field="image"),
            rngs=nnx.Rngs(batch_mix=0),
        )
        pipeline = Pipeline(source=source, stages=[mixer], batch_size=4, rngs=nnx.Rngs(0))

        batch = pipeline.step()
        assert batch["image"].shape == (4, 4)

    def test_mixes_a_batch_named_by_its_records(self):
        """A batch whose rows carry record indices mixes like any other."""
        mixer = BatchMixOperator(
            BatchMixOperatorConfig(mode="cutmix", data_field="image"),
            rngs=nnx.Rngs(batch_mix=0),
        )
        data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32)}
        out = mixer(name_records(batch_ops.from_arrays(data), jnp.arange(4, dtype=jnp.int32), 0))
        assert out.data["image"].shape == (4, 8, 8, 3)


class TestMixingWritesPartnerAndLambda:
    """MixUp and CutMix leave labels untouched and tell the loss what they mixed (design D7)."""

    @staticmethod
    def _batch(labels: jax.Array) -> Batch:
        images = jax.random.uniform(jax.random.key(0), (6, 8, 8, 3))
        return name_records(
            batch_ops.from_arrays({"image": images, "label": labels}), jnp.arange(6) + 20, 0
        )

    @pytest.mark.parametrize("mode", ["mixup", "cutmix"])
    @pytest.mark.parametrize(
        "labels",
        [jnp.array([0, 3, 7, 9, 1, 2], jnp.int32), jax.nn.one_hot(jnp.arange(6) % 4, 4)],
        ids=["integer", "one-hot"],
    )
    def test_labels_are_untouched(self, mode: str, labels: jax.Array) -> None:
        op = BatchMixOperator(BatchMixOperatorConfig(mode=mode), rngs=nnx.Rngs(batch_mix=1))
        batch = self._batch(labels)
        out = op(batch)
        assert out.data["label"].dtype == labels.dtype
        assert jnp.array_equal(out.data["label"], labels)

    @pytest.mark.parametrize("mode", ["mixup", "cutmix"])
    def test_the_partner_is_a_permutation_and_lambda_a_fraction(self, mode: str) -> None:
        op = BatchMixOperator(BatchMixOperatorConfig(mode=mode), rngs=nnx.Rngs(batch_mix=1))
        out = op(self._batch(jnp.arange(6, dtype=jnp.int32)))
        partner = out.states[MIX_PARTNER]
        assert partner.dtype == jnp.int32
        assert sorted(partner.tolist()) == list(range(6))
        lam = out.batch_state[MIX_LAMBDA]
        assert lam.shape == ()
        assert 0.0 <= float(lam) <= 1.0

    def test_mixup_mixes_the_image_with_its_partner_by_lambda(self) -> None:
        op = BatchMixOperator(BatchMixOperatorConfig(mode="mixup"), rngs=nnx.Rngs(batch_mix=1))
        batch = self._batch(jnp.arange(6, dtype=jnp.int32))
        out = op(batch)
        lam, partner = out.batch_state[MIX_LAMBDA], out.states[MIX_PARTNER]
        image = batch.data["image"]
        expected = lam * image + (1 - lam) * image[partner]
        assert jnp.allclose(out.data["image"], expected, rtol=0, atol=1e-6)

    def test_cutmix_lambda_is_the_fraction_of_each_image_kept(self) -> None:
        op = BatchMixOperator(BatchMixOperatorConfig(mode="cutmix"), rngs=nnx.Rngs(batch_mix=1))
        batch = self._batch(jnp.arange(6, dtype=jnp.int32))
        out = op(batch)
        partner = out.states[MIX_PARTNER]
        row = next(i for i in range(6) if int(partner[i]) != i)  # a record mixed with another
        kept = jnp.mean(out.data["image"][row] == batch.data["image"][row])
        assert jnp.allclose(out.batch_state[MIX_LAMBDA], kept, atol=1e-6)

    @pytest.mark.parametrize("mode", ["mixup", "cutmix"])
    def test_a_loss_reading_partner_and_lambda_equals_the_closed_form(self, mode: str) -> None:
        op = BatchMixOperator(BatchMixOperatorConfig(mode=mode), rngs=nnx.Rngs(batch_mix=1))
        labels = jnp.array([0, 3, 7, 9, 1, 2], jnp.int32)
        out = op(self._batch(labels))
        logits = jax.random.normal(jax.random.key(3), (6, 10))

        def cross_entropy(targets: jax.Array) -> jax.Array:
            return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], 1)[:, 0]

        lam, partner = out.batch_state[MIX_LAMBDA], out.states[MIX_PARTNER]
        loss = lam * cross_entropy(out.data["label"]) + (1 - lam) * cross_entropy(
            out.data["label"][partner]
        )
        soft = lam * jax.nn.one_hot(labels, 10) + (1 - lam) * jax.nn.one_hot(labels[partner], 10)
        closed = -jnp.sum(soft * jax.nn.log_softmax(logits), axis=1)
        assert jnp.allclose(loss, closed, atol=1e-5)
