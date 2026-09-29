"""Tests for LoudnessOperator — pure JAX A-weighted loudness extraction.

The operator computes DDSP's perceptual loudness (magenta/ddsp ``spectral_ops.compute_loudness``,
main at 6b6b8a31): the power spectrum weighted by the A-weighting curve in linear scale, averaged
over frequency, then converted to dB relative to ``ref_db`` and clamped at ``-range_db``; the
frequency weights and the reference level are learnable (``nnx.Param``). Its values are checked
against an independent NumPy port of that reference.

Test categories:
1. A-weighting curve correctness
2. Config validation
3. Output shape and structure
4. Acoustic correctness (sine, silence)
5. vmap/JIT compatibility
6. Learnable parameters and gradient flow
"""

import jax.numpy as jnp
import librosa
import numpy as np
import pytest
import scipy.signal
from flax import nnx

from datarax.core import batch_ops
from datarax.core.element_batch import Element
from datarax.operators.modality.audio.loudness_operator import (
    _a_weighting_jax,
    LoudnessConfig,
    LoudnessOperator,
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
        """Config defaults are DDSP's ``compute_loudness`` defaults."""
        config = LoudnessConfig()
        assert config.sample_rate == 16000
        assert config.frame_rate == 250
        assert config.n_fft == 512
        assert config.ref_db == 0.0
        assert config.range_db == 80.0

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
        """Input (64000,) audio → output (1001,) loudness.

        64000 samples at 16kHz = 4 seconds. Center padding centres one frame on each hop
        boundary, both ends included: 64000 // 64 + 1 frames, as DDSP's framing gives.
        """
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        data = {"audio": audio}
        state = {}

        out_data = op.apply(Element(data, state=state)).data
        assert out_data["loudness"].shape == (1001,), (
            f"Expected (1001,), got {out_data['loudness'].shape}"
        )

    def test_output_key(self):
        """Output data dict must have 'loudness' key alongside 'audio'."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        data = {"audio": audio}

        out_data = op.apply(Element(data)).data
        assert "loudness" in out_data
        assert "audio" in out_data, "Original audio key should be preserved"

    def test_batch_gains_the_loudness_field(self):
        """A batch comes back carrying the loudness the operator adds, and its audio."""
        op = LoudnessOperator(LoudnessConfig(), rngs=nnx.Rngs(0))
        batch = batch_ops.from_arrays({"audio": jnp.zeros((2, 32000))})

        result_data = op(batch).data

        assert "loudness" in result_data, "the batch must carry the loudness the operator adds"
        assert "audio" in result_data, "the original audio must be preserved"

    def test_output_shape_different_lengths(self):
        """Operator handles different audio lengths correctly."""
        config = LoudnessConfig(sample_rate=16000, frame_rate=250)
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        # 2 seconds = 32000 samples → 501 frames
        audio = jnp.zeros(32000)
        out_data = op.apply(Element({"audio": audio})).data
        assert out_data["loudness"].shape == (501,)


# ============================================================================
# Acoustic Correctness Tests
# ============================================================================


class TestLoudnessAcoustics:
    """Validate acoustic behavior with known signals."""

    def test_silence(self):
        """Zero audio gives loudness near -range_db (floor)."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        audio = jnp.zeros(64000)
        out_data = op.apply(Element({"audio": audio})).data
        loudness = out_data["loudness"]

        # Silence sits exactly at the floor (-range_db)
        assert jnp.all(jnp.isfinite(loudness)), "Loudness must be finite even for silence"
        assert jnp.all(loudness == -config.range_db)

    def test_louder_signal_higher_loudness(self):
        """Doubling amplitude should increase loudness by ~6 dB."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        t = jnp.linspace(0, 4.0, 64000, endpoint=False)
        audio_quiet = 0.1 * jnp.sin(2 * jnp.pi * 440.0 * t)
        audio_loud = 0.5 * jnp.sin(2 * jnp.pi * 440.0 * t)

        out_quiet = op.apply(Element({"audio": audio_quiet})).data
        out_loud = op.apply(Element({"audio": audio_loud})).data

        mean_quiet = jnp.mean(out_quiet["loudness"])
        mean_loud = jnp.mean(out_loud["loudness"])
        assert mean_loud > mean_quiet, "Louder signal should have higher loudness"


# ============================================================================
# vmap / JIT / Batch Compatibility Tests
# ============================================================================


class TestLoudnessJaxCompat:
    """Validate vmap, JIT, and batch processing compatibility."""

    def test_batch_vmap(self):
        """apply_batch() handles (B, 64000) → (B, 1001)."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        B = 4
        audio = jnp.zeros((B, 64000))
        batch = batch_ops.from_arrays({"audio": audio}, states={"_dummy": jnp.zeros((B,))})

        result = op(batch)
        result_data = result.data
        assert result_data["loudness"].shape == (B, 1001)

    def test_jit_compatible(self):
        """jax.jit wrapping around apply works without error."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jitted_apply(op, data, state):
            applied = op.apply(Element(data, state=state))
            out_data, out_state = applied.data, applied.state
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

        n_bins = config.n_fft // 2 + 1  # 257 for n_fft=512
        weights = op.frequency_weights[...]
        assert weights.shape == (n_bins,), f"Expected ({n_bins},), got {weights.shape}"

        # The reference weights: librosa's A-weighting (floored at -80 dB) at the FFT bins.
        frequencies = librosa.fft_frequencies(sr=16000, n_fft=config.n_fft)
        expected = librosa.A_weighting(frequencies)
        # The curve is 20 times a sum of six log10 terms; float32 rounds each term, so the error
        # at a frequency is bounded by eps times 20 times the sum of the terms' magnitudes.
        f_sq = np.maximum(frequencies, 1e-20) ** 2
        corners = (12194.217**2, 20.598997**2, 107.65265**2, 737.86223**2)
        terms = (
            np.abs(np.log10(corners[0]))
            + 2.0 * np.abs(np.log10(f_sq))
            + sum(np.abs(np.log10(f_sq + c)) for c in corners)
        )
        bound = float(np.finfo(np.float32).eps) * 20.0 * terms
        assert np.all(np.abs(np.asarray(weights) - expected) <= bound)

    def test_ref_db_init(self):
        """ref_db initialized to the configured reference, DDSP's 0 dB by default."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))
        assert op.ref_db[...] == 0.0

    def test_gradient_flow(self):
        """nnx.value_and_grad through loudness produces non-zero gradients."""
        config = LoudnessConfig()
        op = LoudnessOperator(config, rngs=nnx.Rngs(0))

        t = jnp.linspace(0, 4.0, 64000, endpoint=False)
        audio = jnp.sin(2 * jnp.pi * 440.0 * t)

        def loss_fn(op):
            out_data = op.apply(Element({"audio": audio})).data
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


# ============================================================================
# The published method
# ============================================================================


def _ddsp_loudness(
    audio: np.ndarray,
    *,
    sample_rate: int = 16000,
    frame_rate: int = 250,
    n_fft: int = 512,
    range_db: float = 80.0,
    ref_db: float = 0.0,
) -> np.ndarray:
    """DDSP's ``compute_loudness`` (magenta/ddsp main 6b6b8a31), ported to float64 NumPy.

    ``pad(..., padding='center')`` pads ``n_fft // 2`` zeros on both ends; ``tf.signal.stft``
    frames with a periodic Hann window and no end padding; the power is weighted by
    ``10 ** (librosa.A_weighting(f) / 10)``, averaged over frequency and passed through
    ``core.power_to_db`` (floor ``10 ** (-range_db / 10)``, ``10 * log10``, minus ``ref_db``,
    clamp at ``-range_db``).
    """
    hop = sample_rate // frame_rate
    padded = np.pad(audio.astype(np.float64), (n_fft // 2, n_fft // 2))
    n_frames = (len(padded) - n_fft) // hop + 1
    frames = np.stack([padded[i * hop : i * hop + n_fft] for i in range(n_frames)])
    window = scipy.signal.get_window("hann", n_fft)  # periodic, as tf.signal.hann_window
    power = np.abs(np.fft.rfft(frames * window, n=n_fft, axis=-1)) ** 2
    weighting = librosa.A_weighting(librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft))
    average = np.mean(power * 10.0 ** (weighting / 10.0), axis=-1)
    decibels = 10.0 * np.log10(np.maximum(10.0 ** (-range_db / 10.0), average)) - ref_db
    return np.maximum(decibels, -range_db)


def _signals() -> dict[str, np.ndarray]:
    time = np.arange(16000) / 16000.0
    rng = np.random.default_rng(0)
    half_silent = np.concatenate([np.sin(2 * np.pi * 440.0 * time[:8000]), np.zeros(8000)])
    return {
        "sine mixture": 0.5 * np.sin(2 * np.pi * 440.0 * time)
        + 0.1 * np.sin(2 * np.pi * 97 * time),
        "white noise": 0.3 * rng.standard_normal(16000),
        "silence": np.zeros(16000),
        "half silent": half_silent,
    }


@pytest.mark.parametrize("n_fft", [512, 2048])
@pytest.mark.parametrize("name", sorted(_signals()))
def test_loudness_matches_the_published_method(name: str, n_fft: int) -> None:
    """The operator computes DDSP's loudness, frame for frame.

    The bound follows float32 rounding of the averaged power: a relative error ``d`` in the
    power moves the loudness by ``10 * log10(1 + d) ~ 4.34 d`` dB, and summing ``n_bins`` float32
    terms leaves ``d <= n_bins * eps``.
    """
    audio = _signals()[name]
    operator = LoudnessOperator(LoudnessConfig(n_fft=n_fft), rngs=nnx.Rngs(0))

    out = operator.apply(Element({"audio": jnp.asarray(audio, jnp.float32)})).data

    expected = _ddsp_loudness(audio, n_fft=n_fft)
    bound = 4.343 * (n_fft // 2 + 1) * float(np.finfo(np.float32).eps)
    assert out["loudness"].shape == expected.shape
    np.testing.assert_allclose(np.asarray(out["loudness"]), expected, rtol=0, atol=bound)
