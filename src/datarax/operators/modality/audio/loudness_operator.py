"""LoudnessOperator — differentiable A-weighted loudness extraction.

Computes perceptual loudness from audio using STFT + frequency weighting + dB.
The frequency weights are initialized from the IEC 61672 A-weighting curve but
stored as nnx.Param, making them learnable during end-to-end training.

All operations are pure JAX — fully vmap/JIT/grad compatible.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


def _a_weighting_jax(frequencies: jax.Array) -> jax.Array:
    """IEC 61672 A-weighting curve in dB, ported to JAX.

    Standard formula: 0 dB at 1000 Hz, heavy low-frequency rolloff.
    Matches librosa.A_weighting within floating-point tolerance.

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
    return weights


@dataclass
class LoudnessConfig(OperatorConfig):
    """Configuration for LoudnessOperator.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        frame_rate: Output frame rate in Hz (loudness frames per second).
        n_fft: FFT window size for STFT.
        ref_db: Initial reference level in dB (learnable, matches NSynth).
        range_db: Dynamic range floor in dB (silence threshold).
    """

    sample_rate: int = 16000
    frame_rate: int = 250
    n_fft: int = 2048
    ref_db: float = 20.7
    range_db: float = 120.0


class LoudnessOperator(OperatorModule):
    """Differentiable A-weighted loudness extraction operator.

    Computes per-frame loudness from raw audio via:
    1. STFT framing (overlapping windows)
    2. Power spectrum
    3. Learned frequency weighting (initialized from A-weighting)
    4. dB conversion with learnable reference level

    Learnable parameters (nnx.Param):
        frequency_weights: Per-bin frequency weighting, initialized from IEC 61672.
        ref_db: Reference level, initialized to 20.7 (NSynth convention).

    Input:  data["audio"] shape (n_samples,)
    Output: data["audio"] preserved + data["loudness"] shape (n_frames,)
    """

    def __init__(
        self,
        config: LoudnessConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.config: LoudnessConfig = config

        # Derived constants
        self._hop_length = config.sample_rate // config.frame_rate
        self._n_bins = config.n_fft // 2 + 1

        # Learnable frequency weights — initialized from A-weighting
        freqs = jnp.linspace(0, config.sample_rate / 2, self._n_bins)
        a_weights = _a_weighting_jax(freqs)
        self.frequency_weights = nnx.Param(a_weights)

        # Learnable reference level
        self.ref_db = nnx.Param(jnp.array(config.ref_db))

        # Hann window (not learnable, used for STFT)
        self._window = jnp.hanning(config.n_fft)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Compute loudness from audio.

        Args:
            data: Must contain "audio" key with shape (n_samples,).
            state: Passed through unchanged.
            metadata: Passed through unchanged.
            random_params: Unused (deterministic operator).
            stats: Unused.

        Returns:
            (data_with_loudness, state, metadata) where data_with_loudness
            has original keys plus "loudness" with shape (n_frames,).
        """
        audio = data["audio"]
        loudness = self._compute_loudness(audio)
        out_data = {**data, "loudness": loudness}
        return out_data, state, metadata

    def _compute_loudness(self, audio: jax.Array) -> jax.Array:
        """STFT-based A-weighted loudness computation.

        Uses center-padding (n_fft//2 on each side) so that n_frames = n_samples // hop.
        All steps are differentiable JAX operations.
        """
        n_fft = self.config.n_fft
        hop = self._hop_length
        n_samples = audio.shape[0]

        # Center-pad audio so first and last frames are centered on audio boundaries
        pad = n_fft // 2
        audio_padded = jnp.pad(audio, (pad, pad), mode="constant")

        # Number of frames: n_samples // hop (exact for standard configs)
        n_frames = n_samples // hop

        # Frame audio into overlapping windows: (n_frames, n_fft)
        indices = jnp.arange(n_fft)[None, :] + (jnp.arange(n_frames) * hop)[:, None]
        frames = audio_padded[indices]

        # Apply Hann window
        windowed = frames * self._window

        # FFT → power spectrum
        spectrum = jnp.fft.rfft(windowed, n=n_fft, axis=-1)
        power = jnp.real(spectrum * jnp.conj(spectrum))

        # Convert to dB (with floor to avoid log(0))
        power_db = 10.0 * jnp.log10(jnp.maximum(power, 1e-20))

        # Apply learned frequency weighting
        weighted_db = power_db + self.frequency_weights[...]

        # Subtract learnable reference and average across frequency bins
        loudness_per_frame = jnp.mean(weighted_db, axis=-1) - self.ref_db[...]

        # Floor at -range_db
        loudness_per_frame = jnp.maximum(loudness_per_frame, -self.config.range_db)

        return loudness_per_frame

    def get_output_structure(
        self,
        sample_data: PyTree,
        sample_state: PyTree,
    ) -> tuple[PyTree, PyTree]:
        """Declare output structure with added 'loudness' key."""
        out_data = {**{k: 0 for k in sample_data}, "loudness": 0}
        out_state = jax.tree.map(lambda _: 0, sample_state) if sample_state else {}
        return out_data, out_state
