"""CrepeF0Operator — pitch (f0) extraction via CREPE CNN.

Wraps the Flax NNX CrepeModel as a standard OperatorModule. Handles audio
framing, per-frame normalization, and dual-mode pitch decoding (differentiable
for training, local weighted average for inference).

All operations are pure JAX/Flax NNX — fully vmap/JIT/grad compatible.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators.modality.audio.crepe_model import (
    CrepeModel,
    decode_pitch_differentiable,
    decode_pitch_local,
)


@dataclass
class CrepeF0Config(OperatorConfig):
    """Configuration for CrepeF0Operator.

    Attributes:
        capacity: CREPE model size ("tiny", "small", "medium", "large", "full").
        sample_rate: Audio sample rate in Hz.
        frame_rate: Output frame rate in Hz (f0 values per second).
        frame_size: CREPE input window size in samples (fixed at 1024).
        differentiable: If True, use differentiable softmax-based pitch decoding
            (for end-to-end training). If False, use local weighted average with
            argmax (higher accuracy for inference).
        decode_temperature: Temperature for differentiable softmax decoding
            (lower = sharper, closer to argmax). Only used when differentiable=True.
        batch_frames: Maximum number of frames to process in a single CREPE forward
            pass. Limits GPU memory usage for long audio. Set to 0 to process all
            frames at once (may OOM on full model with long audio).
        batch_strategy: Inherited from OperatorConfig (default "vmap"). CREPE is a
            6-layer CNN with large intermediates (layer 1 produces (B, 256, 1024)),
            so vmap across batch elements materializes all activations simultaneously.
            Set to "scan" to process batch elements sequentially (O(1) memory).
    """

    capacity: str = "full"
    sample_rate: int = 16000
    frame_rate: int = 250
    frame_size: int = 1024
    differentiable: bool = True
    decode_temperature: float = 0.05
    batch_frames: int = 128


class CrepeF0Operator(OperatorModule):
    """Pitch (f0) extraction via CREPE CNN (Flax NNX port).

    Extracts per-frame fundamental frequency from raw audio:
    1. Pad audio to fill integer number of frames
    2. Slice into overlapping 1024-sample windows
    3. Normalize each frame (zero-mean, unit-variance)
    4. Run through CREPE CNN → 360-bin probability distribution
    5. Decode pitch (differentiable or local mode)

    All CREPE weights are nnx.Param — learnable for fine-tuning.
    Use op.train()/op.eval() to switch modes (propagates to inner CrepeModel).

    Input:  data["audio"] shape (n_samples,)
    Output: data["audio"] preserved + data["f0_hz"] (n_frames,)
                                    + data["f0_confidence"] (n_frames,)
    """

    def __init__(
        self,
        config: CrepeF0Config,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.config: CrepeF0Config = config

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.crepe_model = CrepeModel(capacity=config.capacity, rngs=rngs)

        # Derived constants
        self._hop_length = config.sample_rate // config.frame_rate

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Extract f0 and confidence from audio.

        Args:
            data: Must contain "audio" key with shape (n_samples,).
            state: Passed through unchanged.
            metadata: Passed through unchanged.
            random_params: Unused (deterministic operator in eval mode).
            stats: Unused.

        Returns:
            (data_with_f0, state, metadata) where data_with_f0 has original
            keys plus "f0_hz" and "f0_confidence" with shape (n_frames,).
        """
        audio = data["audio"]
        f0_hz, confidence = self._extract_f0(audio)
        out_data = {**data, "f0_hz": f0_hz, "f0_confidence": confidence}
        return out_data, state, metadata

    def _extract_f0(self, audio: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Frame, normalize, run CREPE, and decode pitch.

        Args:
            audio: Raw audio shape (n_samples,).

        Returns:
            (f0_hz, confidence) each shape (n_frames,).
        """
        hop = self._hop_length
        frame_size = self.config.frame_size
        n_samples = audio.shape[0]
        n_frames = n_samples // hop

        # Pad audio so we can extract n_frames windows of frame_size
        # Need: last frame start = (n_frames - 1) * hop, end = start + frame_size
        needed_length = (n_frames - 1) * hop + frame_size
        pad_amount = jnp.maximum(needed_length - n_samples, 0)
        audio_padded = jnp.pad(audio, (0, pad_amount), mode="constant")

        # Frame audio: extract 1024-sample windows at hop stride
        starts = jnp.arange(n_frames) * hop
        indices = starts[:, None] + jnp.arange(frame_size)[None, :]
        frames = audio_padded[indices]  # (n_frames, 1024)

        # Per-frame normalization (CREPE requires zero-mean, unit-variance)
        frame_means = jnp.mean(frames, axis=-1, keepdims=True)
        frame_stds = jnp.maximum(jnp.std(frames, axis=-1, keepdims=True), 1e-8)
        frames_normed = (frames - frame_means) / frame_stds

        # Add channel dim for CREPE: (n_frames, 1024) → (n_frames, 1024, 1)
        frames_input = frames_normed[:, :, None]

        # Run CREPE model in chunks to limit GPU memory usage.
        # Full CREPE layer 1 produces (B, 256, 1024) intermediates — at B=1000
        # that's ~1 GB per layer, causing XLA OOM during compilation.
        # Uses jax.lax.scan to trace the CREPE forward pass once (O(1) graph
        # size) instead of a Python for-loop which unrolls N copies.
        batch_frames = self.config.batch_frames
        if batch_frames <= 0 or n_frames <= batch_frames:
            probs = self.crepe_model(frames_input)
        else:
            # Pad frames to multiple of batch_frames and reshape into chunks
            n_chunks = -(-n_frames // batch_frames)  # ceil division
            total_padded = n_chunks * batch_frames
            frames_padded = jnp.pad(frames_input, ((0, total_padded - n_frames), (0, 0), (0, 0)))
            frames_chunked = frames_padded.reshape(n_chunks, batch_frames, frame_size, 1)

            def _crepe_scan_fn(carry, chunk):
                return carry, self.crepe_model(chunk)

            _, probs_chunked = jax.lax.scan(_crepe_scan_fn, None, frames_chunked)
            probs = probs_chunked.reshape(-1, 360)[:n_frames]

        # Decode pitch per frame
        if self.config.differentiable:
            # Vectorize over frames — decode_pitch_differentiable works on (360,)
            def _decode_diff(p):
                return decode_pitch_differentiable(p, self.config.decode_temperature)

            f0_hz, confidence = jax.vmap(_decode_diff)(probs)
        else:

            def _decode_local(p):
                return decode_pitch_local(p)

            f0_hz, confidence = jax.vmap(_decode_local)(probs)

        return f0_hz, confidence

    def get_output_structure(
        self,
        sample_data: PyTree,
        sample_state: PyTree,
    ) -> tuple[PyTree, PyTree]:
        """Declare output structure with added 'f0_hz' and 'f0_confidence' keys."""
        out_data = {**{k: 0 for k in sample_data}, "f0_hz": 0, "f0_confidence": 0}
        out_state = jax.tree.map(lambda _: 0, sample_state) if sample_state else {}
        return out_data, out_state
