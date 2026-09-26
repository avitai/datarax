"""LoudnessOperator — differentiable A-weighted loudness extraction.

Computes DDSP's perceptual loudness (Engel et al., "DDSP: Differentiable Digital Signal
Processing", ICLR 2020; magenta/ddsp ``spectral_ops.compute_loudness`` and ``core.power_to_db``):
the power spectrum is weighted by the A-weighting curve in linear scale, averaged over frequency,
and converted to dB relative to a reference level, clamped at ``-range_db``. The frequency weights
are initialized from the IEC 61672 A-weighting curve, as librosa computes it, and stored as
``nnx.Param`` with the reference level, so both are learnable during end-to-end training.

All operations are pure JAX — fully vmap/JIT/grad compatible.
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


logger = logging.getLogger(__name__)


# librosa.A_weighting's floor, which DDSP's loudness inherits: the curve never falls below it.
A_WEIGHTING_MIN_DB = -80.0


def _a_weighting_jax(frequencies: jax.Array) -> jax.Array:
    """IEC 61672 A-weighting curve in dB, floored at ``A_WEIGHTING_MIN_DB``, ported to JAX.

    Standard formula: 0 dB at 1000 Hz, heavy low-frequency rolloff.
    Matches librosa.A_weighting (default ``min_db=-80``) within floating-point tolerance.

    Args:
        frequencies: Array of frequencies in Hz.

    Returns:
        A-weighting values in dB, same shape as input.
    """
    f_sq = frequencies**2
    # IEC 61672 corner frequencies squared
    c1 = 12194.217**2
    c2 = 20.598997**2
    c3 = 107.65265**2
    c4 = 737.86223**2

    # Numerically safe frequency (avoid log(0))
    f_safe = jnp.maximum(frequencies, 1e-20)

    weights = 2.0 + 20.0 * (
        jnp.log10(c1)
        + 4.0 * jnp.log10(f_safe)
        - jnp.log10(f_sq + c1)
        - jnp.log10(f_sq + c2)
        - 0.5 * jnp.log10(f_sq + c3)
        - 0.5 * jnp.log10(f_sq + c4)
    )
    return jnp.maximum(weights, A_WEIGHTING_MIN_DB)


@dataclass(frozen=True)
class LoudnessConfig(OperatorConfig):
    """Configuration for LoudnessOperator.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        frame_rate: Output frame rate in Hz (loudness frames per second).
        n_fft: FFT window size for STFT.
        ref_db: Initial reference level in dB (learnable); 0 dB is amplitude 1.0.
        range_db: Dynamic range in dB: the loudness floor is ``-range_db``.

    The defaults are DDSP's ``compute_loudness`` defaults.
    """

    sample_rate: int = 16000
    frame_rate: int = 250
    n_fft: int = 512
    ref_db: float = 0.0
    range_db: float = 80.0


class LoudnessOperator(OperatorModule):
    """Differentiable A-weighted loudness extraction operator.

    Computes per-frame loudness from raw audio as DDSP does:
    1. STFT framing: center padding by ``n_fft // 2``, a periodic Hann window,
       ``n_samples // hop + 1`` frames
    2. Power spectrum
    3. Learned frequency weighting in linear scale, ``power * 10 ** (weights / 10)``
    4. Mean power over frequency, floored at ``10 ** (-range_db / 10)``
    5. dB relative to the learnable reference level, clamped at ``-range_db``

    Learnable parameters (nnx.Param):
        frequency_weights: Per-bin weighting in dB, initialized from IEC 61672 A-weighting.
        ref_db: Reference level in dB, initialized to ``config.ref_db``.

    Input:  data["audio"] shape (n_samples,)
    Output: data["audio"] preserved + data["loudness"] shape (n_frames,)
    """

    def __init__(
        self,
        config: LoudnessConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the loudness extraction operator."""
        super().__init__(config, rngs=rngs, name=name)
        self.config: LoudnessConfig = config

        # Derived constants
        self._hop_length = config.sample_rate // config.frame_rate
        self._n_bins = config.n_fft // 2 + 1

        # Learnable frequency weights — initialized from A-weighting at the FFT bin frequencies
        freqs = jnp.linspace(0, config.sample_rate / 2, self._n_bins)
        a_weights = _a_weighting_jax(freqs)
        self.frequency_weights = nnx.Param(a_weights)

        # Learnable reference level
        self.ref_db = nnx.Param(jnp.array(config.ref_db))

        # Periodic Hann window, as tf.signal.stft uses: the symmetric window one sample longer,
        # without its last sample.
        self._window = jnp.hanning(config.n_fft + 1)[:-1]

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Compute loudness from audio.

        Args:
            data: Must contain "audio" key with shape (n_samples,).
            state: Passed through unchanged.
            metadata: Passed through unchanged.
            key: Unused (deterministic operator).
            stats: Unused.

        Returns:
            (data_with_loudness, state, metadata) where data_with_loudness
            has original keys plus "loudness" with shape (n_frames,).
        """
        del key, stats
        audio = data["audio"]
        loudness = self._compute_loudness(audio)
        out_data = {**data, "loudness": loudness}
        return out_data, state, metadata

    def _compute_loudness(self, audio: jax.Array) -> jax.Array:
        """DDSP's loudness: A-weighted mean power per frame, in dB.

        Center padding (``n_fft // 2`` on each side) gives ``n_samples // hop + 1`` frames.
        All steps are differentiable JAX operations.
        """
        n_fft = self.config.n_fft
        hop = self._hop_length
        n_samples = audio.shape[0]

        # Center-pad audio so first and last frames are centered on audio boundaries
        pad = n_fft // 2
        audio_padded = jnp.pad(audio, (pad, pad), mode="constant")

        # Frames of the padded signal: (n_samples + n_fft - n_fft) // hop + 1
        n_frames = n_samples // hop + 1

        # Frame audio into overlapping windows: (n_frames, n_fft)
        indices = jnp.arange(n_fft)[None, :] + (jnp.arange(n_frames) * hop)[:, None]
        frames = audio_padded[indices]

        # Apply Hann window
        windowed = frames * self._window

        # FFT → power spectrum
        spectrum = jnp.fft.rfft(windowed, n=n_fft, axis=-1)
        power = jnp.real(spectrum * jnp.conj(spectrum))

        # Weight in linear scale and average the power over frequency
        weighting = 10.0 ** (self.frequency_weights[...] / 10.0)
        average = jnp.mean(power * weighting, axis=-1)

        # dB relative to the reference, within the dynamic range (DDSP's core.power_to_db)
        range_db = self.config.range_db
        average = jnp.maximum(average, 10.0 ** (-range_db / 10.0))
        loudness = 10.0 * jnp.log10(average) - self.ref_db[...]
        return jnp.maximum(loudness, -range_db)
