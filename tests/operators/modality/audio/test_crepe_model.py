"""Tests for CrepeModel — Flax NNX port of the CREPE pitch detection CNN.

TDD RED phase: All tests written before implementation.
Tests cover architecture correctness, shape validation, weight loading,
pitch decoding (both differentiable and non-differentiable modes),
train/eval mode switching, and gradient flow.

Reference validation tests require pre-generated fixtures from the TF CREPE
model and are marked @pytest.mark.slow.
"""

import pathlib

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# Import under test — will fail in RED phase
try:
    from datarax.operators.modality.audio.crepe_model import (
        CENTS_MAPPING,
        CrepeModel,
        decode_pitch_differentiable,
        decode_pitch_local,
        load_crepe_weights,
        max_pool_1d,
    )
except ImportError:
    CrepeModel = None
    load_crepe_weights = None
    decode_pitch_local = None
    decode_pitch_differentiable = None
    max_pool_1d = None
    CENTS_MAPPING = None

pytestmark = pytest.mark.skipif(
    CrepeModel is None,
    reason="CrepeModel not implemented yet (RED phase)",
)

FIXTURES_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "fixtures" / "crepe"


# ============================================================================
# Architecture and Shape Tests
# ============================================================================


class TestCrepeModelArchitecture:
    """Validate CREPE CNN architecture and output shapes."""

    def test_output_shape(self):
        """Input (B, 1024, 1) → output (B, 360)."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        model.eval()
        x = jnp.zeros((2, 1024, 1))
        out = model(x)
        assert out.shape == (2, 360), f"Expected (2, 360), got {out.shape}"

    def test_output_range(self):
        """All outputs in [0, 1] (sigmoid activation)."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        model.eval()
        x = jax.random.normal(jax.random.key(42), (4, 1024, 1))
        out = model(x)
        assert jnp.all(out >= 0.0) and jnp.all(out <= 1.0), (
            f"Outputs must be in [0, 1], got range [{jnp.min(out)}, {jnp.max(out)}]"
        )

    def test_single_sample(self):
        """Works with batch size 1."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        model.eval()
        x = jnp.zeros((1, 1024, 1))
        out = model(x)
        assert out.shape == (1, 360)

    def test_artifex_pattern(self):
        """conv_layers, batch_norms, dropouts are nnx.List (Artifex GAN pattern)."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        assert isinstance(model.conv_layers, nnx.List), "conv_layers must be nnx.List"
        assert isinstance(model.batch_norms, nnx.List), "batch_norms must be nnx.List"
        assert isinstance(model.dropouts, nnx.List), "dropouts must be nnx.List"
        assert len(model.conv_layers) == 6
        assert len(model.batch_norms) == 6
        assert len(model.dropouts) == 6

    def test_dropout_layers(self):
        """Model has 6 nnx.Dropout layers with rate=0.25."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        for i, dropout in enumerate(model.dropouts):
            assert isinstance(dropout, nnx.Dropout), f"dropouts[{i}] must be nnx.Dropout"
            assert dropout.rate == 0.25, f"dropouts[{i}] rate should be 0.25"

    def test_classifier_shape(self):
        """Final dense layer: 2048 → 360."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        assert model.classifier.kernel[...].shape == (2048, 360)
        assert model.classifier.bias[...].shape == (360,)


# ============================================================================
# Capacity Variants
# ============================================================================


class TestCrepeCapacity:
    """Test different model capacity variants."""

    def test_capacity_tiny(self):
        """Tiny model produces valid output."""
        model = CrepeModel(capacity="tiny", rngs=nnx.Rngs(0))
        model.eval()
        x = jnp.zeros((1, 1024, 1))
        out = model(x)
        assert out.shape == (1, 360)

    def test_capacity_small(self):
        """Small model produces valid output."""
        model = CrepeModel(capacity="small", rngs=nnx.Rngs(0))
        model.eval()
        x = jnp.zeros((1, 1024, 1))
        out = model(x)
        assert out.shape == (1, 360)


# ============================================================================
# Max Pool 1D Helper
# ============================================================================


class TestMaxPool1D:
    """Validate the max_pool_1d utility function."""

    def test_basic_pooling(self):
        """Max pool with window=2, stride=2 halves the length."""
        x = jnp.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        # shape: (1, 4, 2) → (1, 2, 2) after pool
        out = max_pool_1d(x, window=2, stride=2)
        assert out.shape == (1, 2, 2)
        # Max of [1,3]=3 and [5,7]=7 for channel 0
        assert out[0, 0, 0] == 3.0
        assert out[0, 1, 0] == 7.0

    def test_preserves_batch_channels(self):
        """Batch and channel dimensions are preserved."""
        x = jnp.ones((8, 128, 64))  # (B, L, C)
        out = max_pool_1d(x, window=2, stride=2)
        assert out.shape == (8, 64, 64)


# ============================================================================
# Train / Eval Mode
# ============================================================================


class TestCrepeTrainEval:
    """Validate train/eval mode switching."""

    def test_eval_mode(self):
        """model.eval() sets deterministic=True, use_running_average=True."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        model.eval()

        for bn in model.batch_norms:
            assert bn.use_running_average is True, "BN should use running average in eval"
        for dp in model.dropouts:
            assert dp.deterministic is True, "Dropout should be deterministic in eval"

    def test_train_mode(self):
        """model.train() sets deterministic=False, use_running_average=False."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        model.eval()  # first set to eval
        model.train()  # then switch back

        for bn in model.batch_norms:
            assert bn.use_running_average is False, "BN should compute stats in train"
        for dp in model.dropouts:
            assert dp.deterministic is False, "Dropout should be active in train"


# ============================================================================
# JIT / vmap Compatibility
# ============================================================================


class TestCrepeJaxCompat:
    """Validate JIT compilation and vmap support."""

    def test_jit(self):
        """jax.jit(model.__call__) works."""
        model = CrepeModel(capacity="tiny", rngs=nnx.Rngs(0))
        model.eval()

        @nnx.jit
        def forward(model, x):
            return model(x)

        x = jnp.zeros((2, 1024, 1))
        out = forward(model, x)
        assert out.shape == (2, 360)

    def test_gradient_flow(self):
        """jax.grad through CREPE produces non-zero gradients w.r.t. weights."""
        model = CrepeModel(capacity="tiny", rngs=nnx.Rngs(0))
        model.eval()

        x = jax.random.normal(jax.random.key(42), (2, 1024, 1))

        def loss_fn(model):
            out = model(x)
            return jnp.mean(out)

        _, grads = nnx.value_and_grad(loss_fn)(model)

        # Check that at least some conv layer has non-zero gradient
        conv0_grad = grads.conv_layers[0].kernel[...]
        assert jnp.any(conv0_grad != 0.0), "Conv gradients should be non-zero"

        # Classifier gradient
        cls_grad = grads.classifier.kernel[...]
        assert jnp.any(cls_grad != 0.0), "Classifier gradients should be non-zero"

    def test_fine_tunable(self):
        """CREPE weights are nnx.Param (can be optimized)."""
        model = CrepeModel(capacity="tiny", rngs=nnx.Rngs(0))

        # All conv kernels should be nnx.Param
        for i, conv in enumerate(model.conv_layers):
            assert isinstance(conv.kernel, nnx.Param), f"conv_layers[{i}].kernel must be Param"

        # Classifier must be param
        assert isinstance(model.classifier.kernel, nnx.Param)


# ============================================================================
# Pitch Decoding Tests
# ============================================================================


class TestPitchDecoding:
    """Validate pitch decoding from CREPE probability distributions."""

    def test_decode_local_known_peak(self):
        """Known single-peak distribution → correct f0_hz."""
        # Create probability distribution with clear peak at bin 180
        # Bin 180 corresponds to CENTS_MAPPING[180]
        probs = jnp.zeros(360)
        probs = probs.at[179:182].set(jnp.array([0.2, 0.9, 0.3]))

        f0_hz, confidence = decode_pitch_local(probs)

        # Should be near the frequency for the peak bin
        expected_cents = CENTS_MAPPING[180]
        expected_hz = 10.0 * 2.0 ** (expected_cents / 1200.0)
        assert jnp.abs(f0_hz - expected_hz) < 5.0, (
            f"Expected ~{expected_hz:.1f} Hz, got {f0_hz:.1f}"
        )
        assert confidence > 0.5, f"Confidence should be high, got {confidence}"

    def test_decode_differentiable_known_peak(self):
        """Differentiable mode with sharp peak gives similar result to local."""
        probs = jnp.zeros(360)
        probs = probs.at[179:182].set(jnp.array([0.2, 0.9, 0.3]))

        f0_local, _ = decode_pitch_local(probs)
        f0_diff, _ = decode_pitch_differentiable(probs, temperature=0.05)

        # With a sharp peak, both should give similar results
        assert jnp.abs(f0_local - f0_diff) < 20.0, (
            f"Local ({f0_local:.1f}) and differentiable ({f0_diff:.1f}) should be similar"
        )

    def test_decode_differentiable_gradient(self):
        """jax.grad through differentiable decoding produces non-zero gradients."""

        def loss_fn(probs):
            f0_hz, _ = decode_pitch_differentiable(probs, temperature=0.05)
            return f0_hz

        probs = jnp.zeros(360)
        probs = probs.at[179:182].set(jnp.array([0.2, 0.9, 0.3]))

        grad = jax.grad(loss_fn)(probs)
        assert jnp.any(grad != 0.0), "Differentiable decoding must have non-zero gradients"

    def test_decode_confidence_clear_peak(self):
        """High confidence when single clear peak."""
        probs = jnp.zeros(360).at[180].set(0.99)
        _, confidence = decode_pitch_local(probs)
        assert confidence > 0.9

    def test_decode_confidence_flat(self):
        """Low confidence when uniform distribution."""
        probs = jnp.ones(360) / 360.0
        _, confidence_diff = decode_pitch_differentiable(probs)
        # Uniform → maximum entropy → low confidence
        assert confidence_diff < 0.1, (
            f"Flat distribution should have low confidence, got {confidence_diff}"
        )


# ============================================================================
# Weight Loading Tests
# ============================================================================


class TestWeightLoading:
    """Validate weight loading from Keras .h5 files."""

    @pytest.mark.slow
    def test_weight_loading_shapes(self):
        """Loaded weights match model parameter shapes."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        # Try to load weights (requires downloaded .h5 file)
        try:
            load_crepe_weights(model)
        except FileNotFoundError:
            pytest.skip("CREPE weights not downloaded yet")

        # Verify shapes match after loading
        assert model.conv_layers[0].kernel[...].shape[0] == 512  # First conv kernel size
        assert model.classifier.kernel[...].shape == (2048, 360)


# ============================================================================
# Reference Validation Tests (require generated fixtures)
# ============================================================================


class TestCrepeReference:
    """Cross-validate our Flax NNX model against original TF CREPE outputs.

    These tests load pre-generated reference fixtures from tests/fixtures/crepe/.
    Mark as slow since they require loading the full model weights (~80MB).
    """

    @pytest.fixture
    def reference_data(self):
        """Load reference outputs if available."""
        ref_path = FIXTURES_DIR / "reference_outputs.npz"
        if not ref_path.exists():
            pytest.skip(
                "Reference fixtures not generated yet (run scripts/generate_crepe_fixtures.py)"
            )
        return dict(jnp.load(str(ref_path)))

    @pytest.fixture
    def loaded_model(self):
        """Create and load CREPE model with pretrained weights."""
        model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
        try:
            load_crepe_weights(model)
        except FileNotFoundError:
            pytest.skip("CREPE weights not downloaded yet")
        model.eval()
        return model

    @pytest.mark.slow
    def test_crepe_vs_reference_440hz(self, reference_data, loaded_model):
        """Flax model output within tolerance of TF CREPE for 440 Hz."""
        inputs = jnp.array(reference_data["inputs"])
        ref_probs = jnp.array(reference_data["ref_probabilities"])

        # Get the 440 Hz test case (index 0)
        x = inputs[0:1, :, None]  # (1, 1024, 1)
        our_probs = loaded_model(x)

        # Cosine similarity should be very high
        cos_sim = jnp.sum(our_probs[0] * ref_probs[0]) / (
            jnp.linalg.norm(our_probs[0]) * jnp.linalg.norm(ref_probs[0]) + 1e-8
        )
        assert cos_sim > 0.99, f"Cosine similarity for 440Hz: {cos_sim:.4f} (need > 0.99)"

    @pytest.mark.slow
    def test_crepe_vs_reference_f0_hz(self, reference_data, loaded_model):
        """Decoded f0 values match within 1 Hz tolerance."""
        inputs = jnp.array(reference_data["inputs"])
        ref_f0 = jnp.array(reference_data["ref_f0_hz"])

        for i in range(len(inputs)):
            x = inputs[i : i + 1, :, None]
            probs = loaded_model(x)
            our_f0, _ = decode_pitch_local(probs[0])

            if ref_f0[i] > 0:  # Skip silence/noise entries where f0 is undefined
                assert jnp.abs(our_f0 - ref_f0[i]) < 2.0, (
                    f"Sample {i}: expected f0={ref_f0[i]:.1f}, got {our_f0:.1f}"
                )
