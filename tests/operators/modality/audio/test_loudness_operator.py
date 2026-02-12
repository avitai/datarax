"""Tests for LoudnessOperator — pure JAX A-weighted loudness extraction.

TDD RED phase: All tests written before implementation.
The operator computes perceptual loudness from audio using STFT + A-weighting + dB,
with learnable frequency weights and reference level (nnx.Param).

Test categories:
1. A-weighting curve correctness
2. Config validation
3. Output shape and structure
4. Acoustic correctness (sine, silence)
5. vmap/JIT compatibility
6. Learnable parameters and gradient flow
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.element_batch import Batch


# Import under test — will fail in RED phase
try:
    from datarax.operators.modality.audio.loudness_operator import (
        LoudnessConfig,
        LoudnessOperator,
        _a_weighting_jax,
    )
except ImportError:
    LoudnessConfig = None
    LoudnessOperator = None
    _a_weighting_jax = None

pytestmark = pytest.mark.skipif(
    LoudnessOperator is None,
    reason="LoudnessOperator not implemented yet (RED phase)",
)


# ============================================================================
# A-Weighting Curve Tests
# ============================================================================


class TestAWeighting:
    """Validate the IEC 61672 A-weighting curve implementation."""

    def test_a_weighting_zero_at_1khz(self):
        """A-weighting is defined as 0 dB at 1000 Hz (reference frequency)."""
        freqs = jnp.array([1000.0])
        weights = _a_weighting_jax(freqs)
        assert jnp.abs(weights[0]) < 0.5, f"A-weight at 1kHz should be ~0 dB, got {weights[0]}"

    def test_a_weighting_shape(self):
        """Output shape must match input frequency array."""
        freqs = jnp.linspace(20.0, 20000.0, 500)
        weights = _a_weighting_jax(freqs)
        assert weights.shape == freqs.shape

    def test_a_weighting_rolloff_low_freq(self):
        """Low frequencies should be heavily attenuated (< -20 dB at 50 Hz)."""
        freqs = jnp.array([50.0])
        weights = _a_weighting_jax(freqs)
        assert weights[0] < -20.0, f"A-weight at 50 Hz should be < -20 dB, got {weights[0]}"

    def test_a_weighting_rolloff_high_freq(self):
        """High frequencies should also be attenuated (< -5 dB at 16 kHz)."""
        freqs = jnp.array([16000.0])
        weights = _a_weighting_jax(freqs)
        assert weights[0] < -5.0, f"A-weight at 16kHz should be < -5 dB, got {weights[0]}"

    def test_a_weighting_peak_around_2_5khz(self):
        """A-weighting peaks slightly above 0 dB around 2-4 kHz."""
        freqs = jnp.linspace(2000.0, 4000.0, 100)
        weights = _a_weighting_jax(freqs)
        peak = jnp.max(weights)
        assert peak > 0.0, f"A-weighting should peak > 0 dB in 2-4 kHz range, got {peak}"
        assert peak < 2.0, f"A-weighting peak should be < 2 dB, got {peak}"


# ============================================================================
# Config Tests
# ============================================================================


class TestLoudnessConfig:
    """Validate LoudnessConfig defaults and validation."""

    def test_defaults(self):
        """Config defaults match NSynth conventions (16kHz, 250Hz frame rate)."""
        config = LoudnessConfig()
        assert config.sample_rate == 16000
        assert config.frame_rate == 250
        assert config.n_fft == 2048
        assert config.ref_db == 20.7
        assert config.range_db == 120.0

    def test_hop_length_derived(self):
        """hop_length should be sample_rate // frame_rate."""
        config = LoudnessConfig(sample_rate=16000, frame_rate=250)
        # The operator should compute hop_length = 16000 // 250 = 64
        assert config.sample_rate // config.frame_rate == 64

    def test_custom_params(self):
        """Custom parameters should be stored correctly."""
        config = LoudnessConfig(sample_rate=22050, frame_rate=100, n_fft=4096)
        assert config.sample_rate == 22050
        assert config.frame_rate == 100
        assert config.n_fft == 4096


# ============================================================================
# Output Shape and Structure Tests
# ============================================================================


class TestLoudnessOutput:
    """Validate output shapes, keys, and structure declarations."""

    def test_output_shape(self):
        """Input (64000,) audio → output (1000,) loudness for NSynth params.

        64000 samples at 16kHz = 4 seconds.
        At 250 Hz frame rate → 1000 frames.
        """
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        data = {"audio": audio}
        state = {}

        out_data, out_state, out_meta = op.apply(data, state, None)
        assert out_data["loudness"].shape == (1000,), (
            f"Expected (1000,), got {out_data['loudness'].shape}"
        )

    def test_output_key(self):
        """Output data dict must have 'loudness' key alongside 'audio'."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        data = {"audio": audio}

        out_data, _, _ = op.apply(data, {}, None)
        assert "loudness" in out_data
        assert "audio" in out_data, "Original audio key should be preserved"

    def test_get_output_structure(self):
        """get_output_structure() must declare the 'loudness' key."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        sample_data = {"audio": jnp.zeros(64000)}
        sample_state = {}

        out_data_struct, out_state_struct = op.get_output_structure(sample_data, sample_state)
        assert "loudness" in out_data_struct, "Output structure must declare loudness key"
        assert "audio" in out_data_struct, "Output structure must preserve audio key"

    def test_output_shape_different_lengths(self):
        """Operator handles different audio lengths correctly."""
        config = LoudnessConfig(sample_rate=16000, frame_rate=250)
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        # 2 seconds = 32000 samples → 500 frames
        audio = jnp.zeros(32000)
        out_data, _, _ = op.apply({"audio": audio}, {}, None)
        assert out_data["loudness"].shape == (500,)


# ============================================================================
# Acoustic Correctness Tests
# ============================================================================


class TestLoudnessAcoustics:
    """Validate acoustic behavior with known signals."""

    def test_sine_wave(self):
        """440 Hz sine at amplitude 1.0 gives reasonable dB range."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        t = jnp.linspace(0, 4.0, 64000, endpoint=False)
        audio = jnp.sin(2 * jnp.pi * 440.0 * t)

        out_data, _, _ = op.apply({"audio": audio}, {}, None)
        loudness = out_data["loudness"]

        # Loudness should be in a reasonable range (not -inf, not 0)
        assert jnp.all(jnp.isfinite(loudness)), "Loudness must be finite"
        mean_loudness = jnp.mean(loudness)
        assert -100.0 < mean_loudness < 10.0, (
            f"Mean loudness of 440Hz sine should be in [-100, 10] dB, got {mean_loudness}"
        )

    def test_silence(self):
        """Zero audio gives loudness near -range_db (floor)."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        out_data, _, _ = op.apply({"audio": audio}, {}, None)
        loudness = out_data["loudness"]

        # Silence should be at the floor (-range_db)
        assert jnp.all(jnp.isfinite(loudness)), "Loudness must be finite even for silence"
        assert jnp.mean(loudness) < -90.0, (
            f"Silence loudness should be < -90 dB, got {jnp.mean(loudness)}"
        )

    def test_louder_signal_higher_loudness(self):
        """Doubling amplitude should increase loudness by ~6 dB."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        t = jnp.linspace(0, 4.0, 64000, endpoint=False)
        audio_quiet = 0.1 * jnp.sin(2 * jnp.pi * 440.0 * t)
        audio_loud = 0.5 * jnp.sin(2 * jnp.pi * 440.0 * t)

        out_quiet, _, _ = op.apply({"audio": audio_quiet}, {}, None)
        out_loud, _, _ = op.apply({"audio": audio_loud}, {}, None)

        mean_quiet = jnp.mean(out_quiet["loudness"])
        mean_loud = jnp.mean(out_loud["loudness"])
        assert mean_loud > mean_quiet, "Louder signal should have higher loudness"


# ============================================================================
# vmap / JIT / Batch Compatibility Tests
# ============================================================================


class TestLoudnessJaxCompat:
    """Validate vmap, JIT, and batch processing compatibility."""

    def test_batch_vmap(self):
        """apply_batch() handles (B, 64000) → (B, 1000)."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        B = 4
        audio = jnp.zeros((B, 64000))
        batch = Batch.from_parts(
            data={"audio": audio},
            states={"_dummy": jnp.zeros((B,))},
        )

        result = op.apply_batch(batch)
        result_data = result.data.get_value()
        assert result_data["loudness"].shape == (B, 1000)

    def test_jit_compatible(self):
        """jax.jit wrapping around apply works without error."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jitted_apply(op, data, state):
            out_data, out_state, _ = op.apply(data, state, None)
            return out_data, out_state

        audio = jnp.zeros(64000)
        out_data, _ = jitted_apply(op, {"audio": audio}, {})
        assert "loudness" in out_data


# ============================================================================
# Learnable Parameters and Gradient Flow Tests
# ============================================================================


class TestLoudnessLearnableParams:
    """Validate that frequency_weights and ref_db are learnable nnx.Param."""

    def test_learnable_params_exist(self):
        """frequency_weights and ref_db must be nnx.Param."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        # Check that these are nnx.Param (not regular attributes)
        assert hasattr(op, "frequency_weights")
        assert hasattr(op, "ref_db")
        assert isinstance(op.frequency_weights, nnx.Param)
        assert isinstance(op.ref_db, nnx.Param)

    def test_frequency_weights_init(self):
        """frequency_weights initialized from A-weighting curve."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        n_bins = config.n_fft // 2 + 1  # 1025 for n_fft=2048
        weights = op.frequency_weights[...]
        assert weights.shape == (n_bins,), f"Expected ({n_bins},), got {weights.shape}"

        # Compare with A-weighting at corresponding frequencies
        freqs = jnp.linspace(0, config.sample_rate / 2, n_bins)
        expected = _a_weighting_jax(freqs)
        # Should match initialization
        assert jnp.allclose(weights, expected, atol=1e-5)

    def test_ref_db_init(self):
        """ref_db initialized to 20.7 (NSynth convention)."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))
        assert jnp.isclose(op.ref_db[...], 20.7, atol=1e-5)

    def test_gradient_flow(self):
        """nnx.value_and_grad through loudness produces non-zero gradients."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        t = jnp.linspace(0, 4.0, 64000, endpoint=False)
        audio = jnp.sin(2 * jnp.pi * 440.0 * t)

        def loss_fn(op):
            out_data, _, _ = op.apply({"audio": audio}, {}, None)
            return jnp.mean(out_data["loudness"])

        loss, grads = nnx.value_and_grad(loss_fn)(op)

        # Loss should be a finite scalar
        assert jnp.isfinite(loss)

        # Check frequency_weights gradient
        fw_grad = grads.frequency_weights[...]
        assert jnp.any(fw_grad != 0.0), "frequency_weights gradient should be non-zero"

        # Check ref_db gradient
        ref_grad = grads.ref_db[...]
        assert ref_grad != 0.0, "ref_db gradient should be non-zero"

    def test_frequency_weights_shape(self):
        """frequency_weights shape matches n_fft // 2 + 1."""
        config = LoudnessConfig(n_fft=1024)
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))
        assert op.frequency_weights[...].shape == (513,)  # 1024 // 2 + 1
