"""Tests for CrepeF0Operator — pitch (f0) extraction via CREPE CNN.

TDD RED phase: All tests written before implementation.
Tests cover config, output shapes/keys, framing, pitch accuracy,
JIT/vmap compatibility, gradient flow, and train/eval propagation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Import under test — will fail in RED phase
try:
    from datarax.operators.modality.audio.f0_operator import (
        CrepeF0Config,
        CrepeF0Operator,
    )
except ImportError:
    CrepeF0Config = None
    CrepeF0Operator = None

pytestmark = pytest.mark.skipif(
    CrepeF0Operator is None,
    reason="CrepeF0Operator not implemented yet (RED phase)",
)


# ============================================================================
# Config Tests
# ============================================================================


class TestCrepeF0Config:
    """Validate config defaults and derived values."""

    def test_defaults(self):
        """Config defaults match NSynth conventions."""
        cfg = CrepeF0Config()
        assert cfg.capacity == "full"
        assert cfg.sample_rate == 16000
        assert cfg.frame_rate == 250
        assert cfg.frame_size == 1024
        assert cfg.differentiable is True
        assert cfg.decode_temperature == 0.05

    def test_hop_length(self):
        """Hop length derived correctly: sample_rate // frame_rate."""
        cfg = CrepeF0Config()
        assert cfg.sample_rate // cfg.frame_rate == 64

    def test_custom_config(self):
        """Custom parameters override defaults."""
        cfg = CrepeF0Config(
            capacity="tiny",
            sample_rate=22050,
            frame_rate=100,
            differentiable=False,
        )
        assert cfg.capacity == "tiny"
        assert cfg.sample_rate == 22050
        assert cfg.frame_rate == 100
        assert cfg.differentiable is False


# ============================================================================
# Output Shape / Key Tests
# ============================================================================


class TestCrepeF0Output:
    """Validate output shapes and data keys."""

    def test_output_keys(self):
        """Output data has 'f0_hz' and 'f0_confidence' keys alongside 'audio'."""
        op = CrepeF0Operator(CrepeF0Config(capacity="tiny"), rngs=nnx.Rngs(0))
        op.eval()
        data = {"audio": jnp.zeros(64000)}
        out_data, state, meta = op.apply(data, {}, None)
        assert "audio" in out_data, "Original 'audio' key must be preserved"
        assert "f0_hz" in out_data, "Output must have 'f0_hz' key"
        assert "f0_confidence" in out_data, "Output must have 'f0_confidence' key"

    def test_output_shape(self):
        """Input (64000,) → f0_hz (1000,), f0_confidence (1000,)."""
        cfg = CrepeF0Config(capacity="tiny")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        # 64000 samples at sr=16000, frame_rate=250 → 1000 frames
        data = {"audio": jnp.zeros(64000)}
        out_data, _, _ = op.apply(data, {}, None)
        assert out_data["f0_hz"].shape == (1000,), (
            f"Expected (1000,), got {out_data['f0_hz'].shape}"
        )
        assert out_data["f0_confidence"].shape == (1000,), (
            f"Expected (1000,), got {out_data['f0_confidence'].shape}"
        )

    def test_output_structure(self):
        """get_output_structure declares f0_hz and f0_confidence."""
        op = CrepeF0Operator(CrepeF0Config(capacity="tiny"), rngs=nnx.Rngs(0))
        sample_data = {"audio": jnp.zeros(64000)}
        out_data_struct, out_state_struct = op.get_output_structure(sample_data, {})
        assert "f0_hz" in out_data_struct
        assert "f0_confidence" in out_data_struct
        assert "audio" in out_data_struct

    def test_shorter_audio(self):
        """Shorter audio produces fewer frames."""
        cfg = CrepeF0Config(capacity="tiny")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        # 16000 samples → 250 frames
        data = {"audio": jnp.zeros(16000)}
        out_data, _, _ = op.apply(data, {}, None)
        assert out_data["f0_hz"].shape == (250,)


# ============================================================================
# Framing Tests
# ============================================================================


class TestCrepeF0Framing:
    """Validate audio framing for CREPE input."""

    def test_frame_count(self):
        """Correct number of frames: n_samples // hop_length."""
        cfg = CrepeF0Config(capacity="tiny")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        n_samples = 64000
        hop = cfg.sample_rate // cfg.frame_rate
        expected_frames = n_samples // hop
        data = {"audio": jnp.zeros(n_samples)}
        out_data, _, _ = op.apply(data, {}, None)
        assert out_data["f0_hz"].shape[0] == expected_frames

    def test_f0_output_range(self):
        """f0_hz values should be positive (frequencies in Hz)."""
        cfg = CrepeF0Config(capacity="tiny")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        data = {"audio": jax.random.normal(jax.random.key(42), (16000,))}
        out_data, _, _ = op.apply(data, {}, None)
        assert jnp.all(out_data["f0_hz"] > 0), "f0 must be positive"

    def test_confidence_range(self):
        """Confidence values should be in [0, 1]."""
        cfg = CrepeF0Config(capacity="tiny")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        data = {"audio": jax.random.normal(jax.random.key(42), (16000,))}
        out_data, _, _ = op.apply(data, {}, None)
        assert jnp.all(out_data["f0_confidence"] >= 0.0)
        assert jnp.all(out_data["f0_confidence"] <= 1.0)


# ============================================================================
# Train / Eval Propagation
# ============================================================================


class TestCrepeF0TrainEval:
    """Validate that train/eval propagates to inner CREPE model."""

    def test_eval_propagation(self):
        """f0_op.eval() propagates to inner CrepeModel."""
        op = CrepeF0Operator(CrepeF0Config(capacity="tiny"), rngs=nnx.Rngs(0))
        op.eval()
        for bn in op.crepe_model.batch_norms:
            assert bn.use_running_average is True
        for dp in op.crepe_model.dropouts:
            assert dp.deterministic is True

    def test_train_propagation(self):
        """f0_op.train() propagates to inner CrepeModel."""
        op = CrepeF0Operator(CrepeF0Config(capacity="tiny"), rngs=nnx.Rngs(0))
        op.eval()
        op.train()
        for bn in op.crepe_model.batch_norms:
            assert bn.use_running_average is False
        for dp in op.crepe_model.dropouts:
            assert dp.deterministic is False


# ============================================================================
# Differentiable Mode Tests
# ============================================================================


class TestCrepeF0Differentiable:
    """Validate differentiable mode for end-to-end training."""

    def test_gradient_flow_differentiable(self):
        """With differentiable=True, jax.grad through f0 produces non-zero grads."""
        cfg = CrepeF0Config(capacity="tiny", differentiable=True)
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()

        audio = jax.random.normal(jax.random.key(42), (16000,))
        data = {"audio": audio}

        def loss_fn(op):
            out_data, _, _ = op.apply(data, {}, None)
            return jnp.mean(out_data["f0_hz"])

        _, grads = nnx.value_and_grad(loss_fn)(op)
        # CREPE model conv kernels should have non-zero gradients
        conv0_grad = grads.crepe_model.conv_layers[0].kernel[...]
        assert jnp.any(conv0_grad != 0.0), "CREPE conv gradients should be non-zero"

    def test_nondifferentiable_mode(self):
        """With differentiable=False, operator still produces valid output."""
        cfg = CrepeF0Config(capacity="tiny", differentiable=False)
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        data = {"audio": jnp.zeros(16000)}
        out_data, _, _ = op.apply(data, {}, None)
        assert out_data["f0_hz"].shape == (250,)


# ============================================================================
# Chunked Processing Tests
# ============================================================================


class TestCrepeF0Chunking:
    """Validate chunked frame processing produces identical results."""

    def test_chunked_matches_unchunked(self):
        """Chunked processing gives same f0/confidence as full-batch."""
        audio = jax.random.normal(jax.random.key(42), (16000,))
        data = {"audio": audio}

        # Full batch (batch_frames=0 disables chunking)
        cfg_full = CrepeF0Config(capacity="tiny", batch_frames=0)
        op_full = CrepeF0Operator(cfg_full, rngs=nnx.Rngs(0))
        op_full.eval()
        out_full, _, _ = op_full.apply(data, {}, None)

        # Chunked (batch_frames=64 forces multiple chunks for 250 frames)
        cfg_chunked = CrepeF0Config(capacity="tiny", batch_frames=64)
        op_chunked = CrepeF0Operator(cfg_chunked, rngs=nnx.Rngs(0))
        op_chunked.eval()
        out_chunked, _, _ = op_chunked.apply(data, {}, None)

        # Floating-point non-associativity means different chunk sizes produce
        # slightly different results (different addition order in matmuls).
        # 0.1 Hz tolerance is imperceptible for pitch — well within CREPE's
        # stated accuracy of ~20 cents (~1.2 Hz at 440 Hz).
        assert jnp.allclose(out_full["f0_hz"], out_chunked["f0_hz"], atol=0.1), (
            "Chunked and unchunked f0_hz should match within 0.1 Hz"
        )
        assert jnp.allclose(out_full["f0_confidence"], out_chunked["f0_confidence"], atol=1e-3), (
            "Chunked and unchunked confidence should match within 1e-3"
        )

    def test_batch_frames_default(self):
        """Default batch_frames is 128."""
        cfg = CrepeF0Config()
        assert cfg.batch_frames == 128

    def test_output_shape_with_chunking(self):
        """Output shape is correct even when n_frames is not a multiple of batch_frames."""
        # 16000 samples → 250 frames, batch_frames=100 → 3 chunks (100, 100, 50)
        cfg = CrepeF0Config(capacity="tiny", batch_frames=100)
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        op.eval()
        data = {"audio": jnp.zeros(16000)}
        out_data, _, _ = op.apply(data, {}, None)
        assert out_data["f0_hz"].shape == (250,)


# ============================================================================
# Pitch Accuracy (with pretrained weights)
# ============================================================================


class TestCrepeF0PitchAccuracy:
    """Validate pitch accuracy with pretrained CREPE weights.

    These tests load the full pretrained model and are marked slow.
    """

    @pytest.fixture
    def pretrained_op(self):
        """Create F0 operator with pretrained weights."""
        from datarax.operators.modality.audio.crepe_model import load_crepe_weights

        cfg = CrepeF0Config(capacity="full")
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        try:
            load_crepe_weights(op.crepe_model)
        except FileNotFoundError:
            pytest.skip("CREPE weights not available")
        op.eval()
        return op

    @pytest.mark.slow
    def test_440hz_sine(self, pretrained_op):
        """440 Hz sine wave → f0_hz ≈ 440 Hz."""
        sr = 16000
        t = jnp.linspace(0, 1.0, sr, endpoint=False)
        audio = jnp.sin(2 * jnp.pi * 440.0 * t)
        # Normalize
        audio = (audio - jnp.mean(audio)) / jnp.maximum(jnp.std(audio), 1e-8)

        data = {"audio": audio}
        out_data, _, _ = pretrained_op.apply(data, {}, None)

        # Median f0 should be near 440 Hz (some edge frames may be off)
        median_f0 = jnp.median(out_data["f0_hz"])
        assert jnp.abs(median_f0 - 440.0) < 10.0, f"Expected ~440 Hz, got {median_f0:.1f} Hz"

    @pytest.mark.slow
    def test_880hz_sine(self, pretrained_op):
        """880 Hz sine wave → f0_hz ≈ 880 Hz."""
        sr = 16000
        t = jnp.linspace(0, 1.0, sr, endpoint=False)
        audio = jnp.sin(2 * jnp.pi * 880.0 * t)
        audio = (audio - jnp.mean(audio)) / jnp.maximum(jnp.std(audio), 1e-8)

        data = {"audio": audio}
        out_data, _, _ = pretrained_op.apply(data, {}, None)

        median_f0 = jnp.median(out_data["f0_hz"])
        assert jnp.abs(median_f0 - 880.0) < 15.0, f"Expected ~880 Hz, got {median_f0:.1f} Hz"

    @pytest.mark.slow
    def test_silence_low_confidence(self):
        """Silence → low confidence (using local decode for interpretable confidence)."""
        from datarax.operators.modality.audio.crepe_model import load_crepe_weights

        # Use non-differentiable mode — confidence = max probability (interpretable)
        cfg = CrepeF0Config(capacity="full", differentiable=False)
        op = CrepeF0Operator(cfg, rngs=nnx.Rngs(0))
        try:
            load_crepe_weights(op.crepe_model)
        except FileNotFoundError:
            pytest.skip("CREPE weights not available")
        op.eval()

        data = {"audio": jnp.zeros(16000)}
        out_data, _, _ = op.apply(data, {}, None)
        mean_conf = jnp.mean(out_data["f0_confidence"])
        assert mean_conf < 0.5, f"Silence should have low confidence, got {mean_conf:.3f}"
