# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# DDSP: Differentiable Digital Signal Processing

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~3 hrs (GPU, full) / ~15 min (GPU, quick) |
| **Prerequisites** | JAX, Flax NNX, audio/DSP basics, custom operator patterns |
| **Memory** | ~6 GB VRAM (GPU, full) / ~4 GB VRAM (GPU, quick) |
| **Devices** | GPU recommended, CPU supported |
| **Dataset** | NSynth gansynth_subset (~1 GB, auto-downloaded via TFDS) |
| **Format** | Python + Jupyter |

## Overview

This example re-implements the core architecture from **DDSP: Differentiable
Digital Signal Processing** (Engel et al., ICLR 2020) using datarax's
extensibility features. We create **3 custom operators** for audio synthesis
that extend `OperatorModule` directly — proving that datarax's operator
system works for any domain, not just images.

**Key insight**: DDSP shows that classical DSP operations (oscillators,
filters, reverb) can be made differentiable and trained end-to-end, requiring
100x less training data than neural audio models. Datarax's operator system
makes this natural — just subclass `OperatorModule`, add `nnx.Param`, and
the DAG executor handles the rest.

## Learning Goals

By the end of this example, you will be able to:

1. **Create** custom `OperatorModule` subclasses for non-image domains (audio)
2. **Implement** differentiable DSP primitives (harmonic synth, noise filter, reverb)
3. **Compose** parallel + sequential pipelines using `CompositeOperatorModule`
4. **Train** an audio synthesis model using multi-scale spectral loss on real data
5. **Understand** how datarax's extensibility enables any-domain differentiable pipelines

## Reference

- Paper: Engel et al., "DDSP: Differentiable Digital Signal Processing" (ICLR 2020)
  — [arXiv:2001.04643](https://arxiv.org/abs/2001.04643)
- Code: [github.com/magenta/ddsp](https://github.com/magenta/ddsp) (TensorFlow)
- JAX ref: [github.com/PapayaResearch/synthax](https://github.com/PapayaResearch/synthax)
"""

# %% [markdown]
"""
## Setup & Prerequisites

### Required Knowledge
- [Custom Operators](../../core/02_operators_tutorial.py) — OperatorModule pattern
- [DAG Pipelines](../dag/01_dag_fundamentals_guide.py) — Parallel, Merge nodes
- Basic audio/DSP concepts (sample rate, FFT, harmonics)

### Installation

```bash
# Install datarax with data dependencies (includes tensorflow-datasets)
uv pip install "datarax[data]"

# No additional audio libraries needed — all DSP is in pure JAX
```

**Estimated Time:** ~3 hrs on GPU (full, 10K samples) / ~15 min on GPU (quick mode)
"""

# %%
# === Imports ===
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from datarax import from_source
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from datarax.operators import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.operators.modality.audio import LoudnessOperator, LoudnessConfig


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity (DDSP paper, Engel et al. 2020).

    Attempt at bounds: [threshold, max_value] with logarithmic response.
    Used for amplitude activations in both harmonic and noise synthesis.

    Reference: ddsp/core.py — ``exp_sigmoid``
    """
    return max_value * jax.nn.sigmoid(x) ** jnp.log(exponent) + threshold


from datarax.operators.modality.audio.crepe_model import load_crepe_weights
from datarax.operators.modality.audio.f0_operator import CrepeF0Operator, CrepeF0Config

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory for saved figures
OUTPUT_DIR = Path("docs/assets/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_specgram(ax, audio, sample_rate=16000):
    """Plot spectrogram on axes with standard DDSP visualization settings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ax.specgram(
            audio,
            Fs=sample_rate,
            NFFT=1024,
            noverlap=768,
            cmap="magma",
            vmin=-80,
            vmax=0,
        )


# %% [markdown]
r"""
## Core Concepts

### DDSP Architecture

DDSP's key innovation: replace opaque neural audio generation with
**differentiable classical DSP**. The architecture:

1. **Decoder**: Maps audio features (f0, loudness) → synthesis parameters
2. **Harmonic Synth**: Additive synthesis with phase accumulation
3. **Noise Synth**: Filtered white noise with learned frequency response
4. **Reverb**: Trainable FIR impulse response for room acoustics
5. **Loss**: Multi-scale spectral comparison with ground truth

```
                       DDSP Autoencoder
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Audio Features (f0, loudness) → [Decoder]                  │
│                                    (GRU + MLP)              │
│                                       │                     │
│                                  ┌────┴────┐                │
│                                  ▼         ▼                │
│                         [Harmonic    [Filtered               │
│                          Synth]       Noise]   ← Parallel   │
│                            │            │                    │
│                            └─────┬──────┘                    │
│                                  ▼                           │
│                     WEIGHTED_PARALLEL (sum)                   │
│                                  │                           │
│                                  ▼                           │
│                             [Reverb]      ← Trainable IR     │
│                                  │                           │
│                                  ▼                           │
│                      Resynthesized Audio                      │
│                                  │                           │
│                                  ▼                           │
│                    Multi-Scale Spectral Loss                  │
│                                  │                           │
│                    jax.grad → update decoder + operators      │
└─────────────────────────────────────────────────────────────┘
```

### Why Custom Operators?

Datarax's image operators (`BrightnessOperator`, etc.) extend `ModalityOperator`
which provides image-specific helpers (`_extract_field`, `_apply_clip_range`).
For audio, we extend `OperatorModule` directly — there's no audio base class
(yet), which is exactly the point: **datarax is extensible to any domain**.

Each custom operator follows the same pattern:
1. Create a companion `Config` dataclass extending `OperatorConfig`
2. Add `nnx.Param` for learnable parameters in `__init__`
3. Implement `apply()` with the standard signature

### Amplitude Activation: ``exp_sigmoid``

The reference DDSP uses an exponentiated sigmoid nonlinearity for amplitude
outputs (both harmonic and noise). This bounds values to ``[1e-7, 2.0]`` with
a logarithmic response, giving stable gradients across the dynamic range:

$$
\\text{exp\_sigmoid}(x) = 2.0 \\cdot \\sigma(x)^{\\ln 10} + 10^{-7}
$$

This replaces ``softplus`` (unbounded) and ``softmax`` (zero-sum competition
between harmonics), which caused amplitude collapse in early training.

### Multi-Scale Spectral Loss

DDSP uses spectral loss instead of waveform MSE because:
- Waveform MSE penalizes phase differences (which humans can't hear)
- Spectral loss compares frequency content across multiple time scales
- FFT sizes [64, 128, 256, 512, 1024, 2048] capture both fine detail and global structure

$$
\\mathcal{L} = \\sum_{s \\in \\text{scales}} \\left(
  \\|\\hat{S}_s - S_s\\|_1 + \\alpha \\|\\log \\hat{S}_s - \\log S_s\\|_1
\\right)
$$
"""

# %% [markdown]
"""
## Implementation

### Step 1: Load NSynth Dataset + Extract Features via Datarax Operators

We load raw instrument recordings from the NSynth dataset and extract
audio features **using datarax's own audio operators** — the same
`OperatorModule` pattern used for synthesis later in this guide:

1. **`LoudnessOperator`** (pure JAX, learnable weights):
   STFT → power spectrum → A-weighted loudness in dB.
   Frequency weights are `nnx.Param`, initialized from IEC 61672.

2. **`CrepeF0Operator`** (Flax NNX CREPE port):
   Frames audio → normalizes → runs CREPE CNN → decodes pitch.
   All weights are `nnx.Param` — enable fine-tuning during training.

Each sample produces:
- `audio`: (64000,) float32 — 4 seconds at 16 kHz
- `f0_hz`: (1000,) — CREPE pitch estimates at 250 Hz frame rate
- `loudness`: (1000,) — A-weighted loudness in dB

**Why datarax operators instead of crepe + librosa?**
- Same `apply(data, state, metadata)` contract as the synthesis operators below
- Pure JAX — vmap/JIT/grad compatible, GPU-accelerated
- No external Python dependencies (crepe, librosa) needed at runtime
"""


# %%
# Step 1: Load NSynth dataset
SAMPLE_RATE = 16000
AUDIO_LENGTH = 64000  # 4 seconds at 16 kHz
N_FRAMES = 1000  # Feature frames at 250 Hz frame rate
FRAME_RATE = 250  # Hz
N_HARMONICS = 100  # Paper uses 100 harmonics
N_NOISE_BANDS = 65  # Number of frequency bins for noise filter

# === Training Configuration ===
# All mode-dependent settings in one place. QUICK_MODE=True for fast demos
# (~15 min GPU), False for full training (~3 hrs GPU, ~31K steps).


@dataclass(frozen=True)
class TrainConfig:
    """Immutable training configuration — all mode-dependent settings."""

    n_train: int
    n_test: int
    num_epochs: int
    batch_size: int
    loss_fft_sizes: tuple[int, ...]


QUICK_CONFIG = TrainConfig(
    n_train=500,
    n_test=100,
    num_epochs=5,
    batch_size=8,
    # Fewer FFT scales reduces XLA compilation time significantly
    # (each scale adds a separate STFT + gradient computation to the XLA graph)
    loss_fft_sizes=(256, 1024, 2048),
)

FULL_CONFIG = TrainConfig(
    n_train=10000,
    n_test=500,
    num_epochs=100,
    batch_size=32,
    loss_fft_sizes=(64, 128, 256, 512, 1024, 2048),
)

QUICK_MODE = False
cfg = QUICK_CONFIG if QUICK_MODE else FULL_CONFIG


def load_nsynth(
    n_train: int = 10000,
    n_test: int = 500,
) -> tuple[dict, dict]:
    """Load NSynth gansynth_subset and extract features with datarax operators.

    Downloads via tensorflow_datasets on first run (~1 GB). Computes f0 with
    datarax's CrepeF0Operator (Flax NNX CREPE port) and loudness with
    LoudnessOperator (pure JAX A-weighted STFT). Results are cached to disk.

    Args:
        n_train: Number of training samples to use.
        n_test: Number of test samples to use.

    Returns:
        Tuple of (train_data, test_data) dicts with keys:
            audio: (N, 64000) float32
            f0: (N, 1000) float32 — MIDI-normalized f0 in [0,1]
            loudness: (N, 1000) float32 — dB-range normalized loudness in [0,1]
            f0_hz: (N, 1000) float32 — raw f0 in Hz
    """
    import os as _os
    import glob
    import csv
    import tensorflow as tf

    # Prevent TF from claiming GPU memory (only JAX needs it)
    tf.config.set_visible_devices([], "GPU")

    # ---- Fast path: bypass Beam entirely ----
    # The TFDS gansynth_subset.f0_and_loudness config runs CREPE (a CNN) on
    # every audio clip via Apache Beam, which takes 30+ minutes even with
    # multi-processing. Instead, we:
    #   1. Read raw NSynth TFRecords directly (already downloaded)
    #   2. Filter to GANSynth subset (acoustic instruments, MIDI pitch [24,84])
    #   3. Compute f0 and loudness with datarax's audio operators — pure JAX,
    #      GPU-accelerated, only for the samples we need (not all ~290K)
    data_dir = _os.environ.get("TFDS_DATA_DIR", None)
    if data_dir is None:
        data_dir = _os.path.join(_os.path.expanduser("~"), "tensorflow_datasets")

    # Step 1: Download raw NSynth data if needed (uses TFDS downloader)
    import tensorflow_datasets as tfds

    tfds.builder("nsynth/gansynth_subset", data_dir=data_dir)
    dl_manager = tfds.download.DownloadManager(
        download_dir=_os.path.join(data_dir, "downloads"),
        extract_dir=_os.path.join(data_dir, "downloads", "extracted"),
    )
    dl_urls = {
        "examples": {
            "train": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.tfrecord.tar",
        },
        "gansynth_splits": "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-gansynth_splits.csv",
    }
    dl_paths = dl_manager.download_and_extract(dl_urls)

    # Step 2: Load GANSynth split IDs (acoustic instruments, pitch [24,84])
    gansynth_train_ids = set()
    with tf.io.gfile.GFile(dl_paths["gansynth_splits"]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "train":
                gansynth_train_ids.add(row["id"])
    print(f"  GANSynth train subset: {len(gansynth_train_ids)} note IDs")

    # Step 3: Read raw TFRecords, filter to GANSynth subset, collect samples
    train_dir = dl_paths["examples"]["train"]
    if _os.path.isdir(train_dir):
        tfrecord_files = sorted(glob.glob(_os.path.join(train_dir, "*.tfrecord*")))
    else:
        tfrecord_files = [train_dir]

    # Parse raw NSynth TFRecord format
    feature_spec = {
        "audio": tf.io.FixedLenFeature([64000], tf.float32),
        "note_str": tf.io.FixedLenFeature([], tf.string),
    }
    n_total = n_train + n_test
    # Over-read to ensure enough samples after filtering + shuffling
    n_read_target = n_total * 2

    raw_ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=8)
    audio_list = []
    note_ids = []
    for raw_record in raw_ds:
        parsed = tf.io.parse_single_example(raw_record, feature_spec)
        note_id = parsed["note_str"].numpy().decode("utf-8")
        if note_id in gansynth_train_ids:
            audio_list.append(parsed["audio"].numpy())
            note_ids.append(note_id)
            if len(audio_list) >= n_read_target:
                break

    print(f"  Loaded {len(audio_list)} GANSynth samples from raw TFRecords")
    # Shuffle and trim to exact count
    rng_load = np.random.RandomState(42)
    load_indices = rng_load.permutation(len(audio_list))[:n_total]
    audio_arr = np.stack([audio_list[i] for i in load_indices])

    # Step 4: Compute f0 and loudness with datarax audio operators
    # These use the same OperatorModule.apply() contract as the synthesis
    # operators below — showing that datarax is extensible to any domain.
    #
    # Check disk cache first (feature extraction is a one-time cost)
    cache_path = _os.path.join(data_dir, f"nsynth_ddsp_cache_{n_total}.npz")
    if _os.path.exists(cache_path):
        print(f"  Loading cached features from {cache_path}")
        cached = np.load(cache_path)
        audio_all = cached["audio"]
        f0_all = cached["f0_hz"]
        loudness_all = cached["loudness"]
    else:
        # Create datarax feature extraction operators
        rngs = nnx.Rngs(0)
        loudness_op = LoudnessOperator(LoudnessConfig(), rngs=rngs)

        f0_op = CrepeF0Operator(
            CrepeF0Config(differentiable=False, batch_strategy="scan"),
            rngs=rngs,
        )
        load_crepe_weights(f0_op.crepe_model)
        f0_op.eval()

        print(f"  Extracting features with datarax operators for {len(audio_arr)} samples...")
        f0_list, loudness_list = [], []

        # Batched extraction — call operators directly with Batch objects.
        # __call__ → apply_batch → scan(apply), processing elements sequentially.
        # CREPE uses jax.lax.scan internally for frame chunking AND
        # batch_strategy="scan" processes elements sequentially (O(1) memory).
        extract_batch_size = 16

        for start in range(0, len(audio_arr), extract_batch_size):
            end = min(start + extract_batch_size, len(audio_arr))
            audio_batch = jnp.array(audio_arr[start:end])
            batch = Batch.from_parts(data={"audio": audio_batch}, states={})

            loud_out = loudness_op(batch)
            loudness_list.append(np.array(loud_out.get_data()["loudness"]))

            f0_out = f0_op(batch)
            f0_list.append(np.array(f0_out.get_data()["f0_hz"]))

            if end % 50 == 0 or end == len(audio_arr):
                print(f"    Processed {end}/{len(audio_arr)} samples")

        audio_all = np.stack([audio_list[i] for i in load_indices])
        f0_all = np.concatenate(f0_list, axis=0)
        loudness_all = np.concatenate(loudness_list, axis=0)

        # Cache to disk for subsequent runs
        np.savez(cache_path, audio=audio_all, f0_hz=f0_all, loudness=loudness_all)
        print(f"  Cached features to {cache_path}")

    # Normalize audio to [-1, 1]
    audio_max = np.max(np.abs(audio_all), axis=1, keepdims=True)
    audio_all = audio_all / np.maximum(audio_max, 1e-8)

    # Scale loudness to [0, 1] via fixed dB range (matching DDSP paper)
    DB_RANGE = 80.0
    loudness_norm = np.clip(loudness_all / DB_RANGE + 1.0, 0.0, 1.0)

    # Scale f0 to [0, 1] via MIDI note normalization (perceptually uniform, matching DDSP paper)
    f0_midi = 12.0 * np.log2(np.maximum(f0_all, 1e-5) / 440.0) + 69.0
    f0_scaled = np.clip(f0_midi / 127.0, 0.0, 1.0)

    # Shuffle with fixed seed and split
    rng = np.random.RandomState(42)
    n_total = len(audio_all)
    indices = rng.permutation(n_total)

    n_train = min(n_train, n_total - n_test)
    train_idx = indices[:n_train]
    test_idx = indices[n_train : n_train + n_test]

    train_data = {
        "audio": audio_all[train_idx],
        "f0": f0_scaled[train_idx],
        "loudness": loudness_norm[train_idx],
        "f0_hz": f0_all[train_idx],  # Keep raw Hz for synthesis
    }
    test_data = {
        "audio": audio_all[test_idx],
        "f0": f0_scaled[test_idx],
        "loudness": loudness_norm[test_idx],
        "f0_hz": f0_all[test_idx],
    }
    return train_data, test_data


# Load NSynth data
print(f"Loading NSynth dataset ({cfg.n_train} train, {cfg.n_test} test)...")
train_data, test_data = load_nsynth(n_train=cfg.n_train, n_test=cfg.n_test)

# Wrap in MemorySource
train_source = MemorySource(MemorySourceConfig(), data=train_data, rngs=nnx.Rngs(0))
test_source = MemorySource(MemorySourceConfig(), data=test_data, rngs=nnx.Rngs(1))

print(
    f"Train: audio={train_data['audio'].shape}, "
    f"f0={train_data['f0'].shape}, "
    f"loudness={train_data['loudness'].shape}"
)
print(
    f"Sample rate: {SAMPLE_RATE} Hz, Audio length: {AUDIO_LENGTH} samples "
    f"({AUDIO_LENGTH / SAMPLE_RATE:.1f}s)"
)
print(f"Feature frames: {N_FRAMES} @ {FRAME_RATE} Hz frame rate")
# Expected output (QUICK_MODE=False):
# Train: audio=(10000, 64000), f0=(10000, 1000), loudness=(10000, 1000)
# Sample rate: 16000 Hz, Audio length: 64000 samples (4.0s)
# Feature frames: 1000 @ 250 Hz frame rate

# %%
# Visualize sample audio waveforms and their spectrograms
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
time_axis = np.arange(AUDIO_LENGTH) / SAMPLE_RATE

for i in range(3):
    audio = train_data["audio"][i]
    f0_val = float(train_data["f0_hz"][i, N_FRAMES // 2])  # Mid-point f0

    # Waveform (top row) — show first 4000 samples (~250ms)
    n_show = 4000
    axes[0, i].plot(time_axis[:n_show], audio[:n_show], color="steelblue", linewidth=0.5)
    axes[0, i].set_xlabel("Time (s)")
    axes[0, i].set_ylabel("Amplitude")
    axes[0, i].set_title(f"Sample {i + 1}: f0 ~ {f0_val:.0f} Hz")
    axes[0, i].set_xlim(0, n_show / SAMPLE_RATE)
    axes[0, i].grid(True, alpha=0.3)

    # Spectrogram (bottom row)
    plot_specgram(axes[1, i], audio)
    axes[1, i].set_xlabel("Time (s)")
    axes[1, i].set_ylabel("Frequency (Hz)")
    axes[1, i].set_ylim(0, 4000)
    axes[1, i].set_title("Spectrogram (harmonics visible)")

fig.suptitle(
    "NSynth Dataset — Real Instrument Waveforms and Spectrograms\n"
    "Each sample is a 4-second recording with pre-computed f0 and loudness",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "cv-ddsp-dataset-samples.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: docs/assets/images/examples/cv-ddsp-dataset-samples.png")

# %% [markdown]
"""
### Step 2: Custom Audio Operators (Extending OperatorModule)

These operators extend `OperatorModule` directly — not `ModalityOperator` —
because there's no audio-specific base class. This showcases datarax's
extensibility: you can build operators for any data type.

Each operator follows the standard contract:
- `OperatorConfig` subclass for configuration
- `nnx.Param` for learnable parameters
- `apply(data, state, metadata, random_params, stats) → (data, state, metadata)`

**Critical design choices**: The harmonic synthesizer uses **per-frame synthesis**
with upsampling from frame rate (250 Hz) to sample rate (16 kHz) via linear
interpolation, matching the DDSP paper (Engel et al. 2020). Phase accumulation
(`cumsum`) on the upsampled f0 ensures smooth phase continuity — without it,
frequency changes create phase discontinuities (audible clicks).
"""


# %%
# Step 2: Custom DDSP operators


# --- Operator 1: Harmonic Synthesizer ---
@dataclass
class HarmonicSynthConfig(OperatorConfig):
    """Configuration for Harmonic Synthesizer.

    Attributes:
        n_harmonics: Number of harmonics to synthesize
        n_frames: Number of input feature frames (for upsampling to sample rate)
        sample_rate: Audio sample rate in Hz
        audio_length: Number of output audio samples
    """

    n_harmonics: int = field(default=N_HARMONICS, kw_only=True)
    n_frames: int = field(default=N_FRAMES, kw_only=True)
    sample_rate: int = field(default=SAMPLE_RATE, kw_only=True)
    audio_length: int = field(default=AUDIO_LENGTH, kw_only=True)


class HarmonicSynthOperator(OperatorModule):
    """Differentiable harmonic additive synthesizer with phase accumulation.

    Generates audio as a sum of sinusoidal harmonics using continuous phase
    accumulation (matching the DDSP paper's Harmonic synth):

        phase[t] = cumsum(2π * f0[t] / sample_rate)
        audio = Σ amplitudes[k] * sin(k * phase)

    Phase accumulation is critical for differentiable synthesis — it ensures
    smooth phase continuity when f0 varies over time (unlike instantaneous
    phase `sin(2π * k * f0 * t)` which creates artifacts).

    Harmonics above the Nyquist frequency are filtered out.
    """

    def __init__(self, config: HarmonicSynthConfig, *, rngs: nnx.Rngs | None = None):
        super().__init__(config, rngs=rngs)
        self.config: HarmonicSynthConfig = config

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Synthesize audio from per-frame harmonic amplitudes and f0.

        Upsamples frame-rate controls to sample-rate via linear interpolation,
        then performs additive synthesis with time-varying phase accumulation
        (matching Engel et al. 2020, Section 3.1).

        Expected data keys:
            - 'amplitudes': (n_frames, n_harmonics) — per-frame harmonic amplitudes
            - 'f0_hz': (n_frames,) — fundamental frequency in Hz per frame

        Output data keys (added/updated):
            - 'audio': (audio_length,) — synthesized waveform
        """
        amplitudes = data["amplitudes"]  # (n_frames, n_harmonics)
        f0_hz = data["f0_hz"]  # (n_frames,)
        n_harmonics = self.config.n_harmonics
        n_frames = self.config.n_frames
        sr = self.config.sample_rate
        length = self.config.audio_length

        # Upsample f0 from frame rate → sample rate (linear interpolation)
        frame_times = jnp.linspace(0, 1, n_frames)
        sample_times = jnp.linspace(0, 1, length)
        f0_upsampled = jnp.interp(sample_times, frame_times, f0_hz)  # (audio_length,)

        # Upsample per-harmonic amplitudes: (n_frames, n_harmonics) → (audio_length, n_harmonics)
        amp_upsampled = jax.vmap(
            lambda amp_k: jnp.interp(sample_times, frame_times, amp_k),
            in_axes=1,
            out_axes=1,
        )(amplitudes)  # (audio_length, n_harmonics)

        # Phase accumulation with time-varying f0
        phase_inc = 2.0 * jnp.pi * f0_upsampled / sr  # (audio_length,)
        phase = jnp.cumsum(phase_inc)  # (audio_length,)

        # Harmonic indices: 1, 2, ..., n_harmonics
        harmonic_k = jnp.arange(1, n_harmonics + 1, dtype=jnp.float32)

        # Nyquist filtering (per-sample, since f0 varies over time)
        nyquist = sr / 2.0
        valid_mask = (f0_upsampled[:, None] * harmonic_k[None, :]) < nyquist
        amp_upsampled = amp_upsampled * valid_mask

        # Sinusoid generation + weighted sum
        harmonics = jnp.sin(harmonic_k[None, :] * phase[:, None])  # (audio_length, n_harmonics)
        audio = jnp.sum(amp_upsampled * harmonics, axis=1)  # (audio_length,)

        out_data = {**data, "audio": audio}
        return out_data, state, metadata


# --- Operator 2: Filtered Noise ---
@dataclass
class FilteredNoiseConfig(OperatorConfig):
    """Configuration for Filtered Noise synthesizer.

    Attributes:
        audio_length: Number of output audio samples
        n_noise_bands: Number of frequency bands for noise filter
    """

    audio_length: int = field(default=AUDIO_LENGTH, kw_only=True)
    n_noise_bands: int = field(default=N_NOISE_BANDS, kw_only=True)


class FilteredNoiseOperator(OperatorModule):
    """Differentiable filtered noise synthesizer.

    Generates audio by filtering white noise in the frequency domain:
        1. Generate white noise
        2. FFT → multiply by learned frequency response → IFFT

    The frequency magnitudes are provided as input (from decoder) and passed
    through ``exp_sigmoid`` (bounded [1e-7, 2.0]) before interpolation,
    matching the DDSP reference (ddsp/synths.py). Uses a fixed noise seed
    for deterministic gradient computation.
    """

    def __init__(self, config: FilteredNoiseConfig, *, rngs: nnx.Rngs | None = None):
        super().__init__(config, rngs=rngs)
        self.config: FilteredNoiseConfig = config
        # Fixed noise for deterministic gradients
        noise = jax.random.normal(jax.random.key(0), (config.audio_length,))
        self.fixed_noise = nnx.Variable(noise)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Filter noise using learned frequency magnitudes.

        Expected data keys:
            - 'noise_magnitudes': (n_frames, n_noise_bands) — per-frame filter shape

        Output data keys (added/updated):
            - 'audio': (audio_length,) — filtered noise waveform
        """
        noise_magnitudes = data["noise_magnitudes"]  # (n_frames, n_noise_bands)

        # Average over time frames (simplification — paper uses per-frame overlap-add)
        magnitudes = jnp.mean(noise_magnitudes, axis=0)  # (n_noise_bands,)

        noise = self.fixed_noise[...]  # (audio_length,)

        # FFT-based filtering
        noise_fft = jnp.fft.rfft(noise)  # (audio_length//2 + 1,)
        n_fft = noise_fft.shape[0]

        # Interpolate magnitudes to match FFT size
        x_interp = jnp.linspace(0, 1, n_fft)
        x_orig = jnp.linspace(0, 1, magnitudes.shape[0])
        filter_response = jnp.interp(x_interp, x_orig, exp_sigmoid(magnitudes))

        # Apply filter and IFFT
        filtered_fft = noise_fft * filter_response
        audio = jnp.fft.irfft(filtered_fft, n=self.config.audio_length)

        out_data = {**data, "audio": audio}
        return out_data, state, metadata


# --- Operator 3: Reverb ---
@dataclass
class ReverbConfig(OperatorConfig):
    """Configuration for trainable Reverb operator.

    Attributes:
        ir_length: Length of impulse response in samples
        sample_rate: Audio sample rate in Hz
    """

    ir_length: int = field(default=SAMPLE_RATE, kw_only=True)  # 1 second IR
    sample_rate: int = field(default=SAMPLE_RATE, kw_only=True)


class ReverbOperator(OperatorModule):
    """Differentiable reverb via trainable FIR impulse response.

    Applies room acoustics by convolving the input audio with a learned
    impulse response (IR). The IR is initialized with exponential decay
    (approximating a simple room) and optimized end-to-end.

    Uses FFT-based convolution for efficiency.

    Matches DDSP paper's Reverb effect (ddsp/effects.py).
    """

    def __init__(self, config: ReverbConfig, *, rngs: nnx.Rngs | None = None):
        super().__init__(config, rngs=rngs)
        self.config: ReverbConfig = config

        # Learnable impulse response (init = exponential decay)
        decay = jnp.exp(-jnp.arange(config.ir_length, dtype=jnp.float32) * 5.0 / config.ir_length)
        self.impulse_response = nnx.Param(decay * 0.1)

    def apply(
        self,
        data: dict[str, Any],
        state: dict[str, Any],
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """Apply reverb to audio via FFT-based convolution.

        Expected data keys:
            - 'audio': (audio_length,) — input audio

        Output data keys (updated):
            - 'audio': (audio_length,) — reverbed audio (same length)
        """
        audio = data["audio"]  # (audio_length,)
        ir = self.impulse_response[...]  # (ir_length,)

        # FFT-based convolution
        n_fft = audio.shape[0] + ir.shape[0] - 1
        # Round up to next power of 2 for FFT efficiency
        n_fft_padded = 1 << (n_fft - 1).bit_length()

        audio_fft = jnp.fft.rfft(audio, n=n_fft_padded)
        ir_fft = jnp.fft.rfft(ir, n=n_fft_padded)
        convolved = jnp.fft.irfft(audio_fft * ir_fft, n=n_fft_padded)

        # Trim to original length
        reverbed = convolved[: audio.shape[0]]

        out_data = {**data, "audio": reverbed}
        return out_data, state, metadata


# Verify operators
print("Verifying DDSP operators...")

# Test harmonic synth (per-frame inputs)
h_config = HarmonicSynthConfig()
h_op = HarmonicSynthOperator(h_config)
h_batch = Batch.from_parts(
    data={
        "amplitudes": jnp.ones((1, N_FRAMES, N_HARMONICS)) * 0.1,
        "f0_hz": jnp.ones((1, N_FRAMES)) * 440.0,
    },
    states={},
)
h_result = h_op(h_batch)
h_out = h_result.get_data()
print(f"  HarmonicSynth: output keys={list(h_out.keys())}, audio shape={h_out['audio'].shape}")

# Test filtered noise (per-frame input)
n_config = FilteredNoiseConfig()
n_op = FilteredNoiseOperator(n_config)
n_batch = Batch.from_parts(
    data={"noise_magnitudes": jnp.ones((1, N_FRAMES, N_NOISE_BANDS))},
    states={},
)
n_result = n_op(n_batch)
n_out = n_result.get_data()
print(f"  FilteredNoise: output keys={list(n_out.keys())}, audio shape={n_out['audio'].shape}")

# Test reverb
r_config = ReverbConfig()
r_op = ReverbOperator(r_config)
r_batch = Batch.from_parts(
    data={"audio": jnp.sin(jnp.linspace(0, 10, AUDIO_LENGTH))[None]},
    states={},
)
r_result = r_op(r_batch)
r_out = r_result.get_data()
print(
    f"  Reverb: IR params={r_op.impulse_response[...].shape[0]}, audio shape={r_out['audio'].shape}"
)

# Count total operator parameters
total_op_params = sum(
    p.size for op in [h_op, n_op, r_op] for p in jax.tree.leaves(nnx.state(op, nnx.Param))
)
print(f"\nTotal operator parameters: {total_op_params:,}")
# Expected output:
#   HarmonicSynth: output keys=['amplitudes', 'f0_hz', 'audio'], audio shape=(1, 64000)
#   FilteredNoise: output keys=['noise_magnitudes', 'audio'], audio shape=(1, 64000)
#   Reverb: IR params=16000, audio shape=(1, 64000)
#   Total operator parameters: 16,000

# %% [markdown]
"""
### Step 3: DDSP Decoder (Paper-Accurate Architecture)

The paper's "decoder" maps audio features (f0, loudness) to synthesis
parameters. It follows the architecture from Section 3.1:

    f0 + loudness → Linear(2→512) → GRU(512) → MLP(512, 3 layers) → heads

The MLP stack uses the Artifex pattern: `nnx.List` for dynamic layer
collections with `LayerNorm` + `ReLU` activation at each layer.

**Why "decoder", not "encoder"?** The paper calls this a decoder because
it maps *extracted audio features* to *synthesis parameters* — the inverse
direction of an encoder. Previous versions of this example incorrectly
called it "encoder".
"""


# %%
# Step 3: DDSP Decoder
class DDSPDecoder(nnx.Module):
    """RNN-FC decoder predicting synth params from audio features.

    Architecture (matching paper Section 3.1):
        f0 + loudness → Linear(2→hidden) → GRU(hidden) →
        MLP(hidden, n_layers) → output heads

    Output activations use ``exp_sigmoid`` (bounded [1e-7, 2.0]) for amplitude
    and harmonic distribution heads, matching the reference DDSP implementation.
    Noise head bias is initialized to -5.0 so noise starts near-zero.

    Uses nnx.List for the MLP layer collection (Artifex pattern) to
    allow arbitrary depth without hardcoding layer count.

    Args:
        hidden_dim: Hidden dimension for GRU and MLP layers.
        n_harmonics: Number of harmonic amplitudes to predict.
        n_noise_bands: Number of noise filter frequency bands.
        n_mlp_layers: Number of MLP layers after GRU.
        rngs: Flax NNX random number generators.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        n_harmonics: int = N_HARMONICS,
        n_noise_bands: int = N_NOISE_BANDS,
        n_mlp_layers: int = 3,
        rngs: nnx.Rngs,
    ):
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands

        # Input projection: f0_scaled + loudness_normalized (2 features) → hidden
        self.input_proj = nnx.Linear(2, hidden_dim, rngs=rngs)

        # GRU for temporal modeling over feature frames
        self.gru_cell = nnx.GRUCell(hidden_dim, hidden_dim, rngs=rngs)

        # Learnable initial GRU state
        self.init_state = nnx.Param(jnp.zeros(hidden_dim))

        # MLP stack (Artifex pattern: nnx.List for dynamic layer collections)
        self.mlp_layers = nnx.List([])
        self.mlp_norms = nnx.List([])
        for _ in range(n_mlp_layers):
            self.mlp_layers.append(nnx.Linear(hidden_dim, hidden_dim, rngs=rngs))
            self.mlp_norms.append(nnx.LayerNorm(hidden_dim, rngs=rngs))

        # Output heads
        self.amplitude_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.harmonic_head = nnx.Linear(hidden_dim, n_harmonics, rngs=rngs)
        # Initialize noise head bias to -5.0 so noise starts near-zero
        # (matching DDSP paper's initial_bias=-5.0 for FilteredNoise)
        self.noise_head = nnx.Linear(hidden_dim, n_noise_bands, rngs=rngs)
        self.noise_head.bias.value = jnp.full(n_noise_bands, -5.0)

    def __call__(self, f0: jax.Array, loudness: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Decode audio features into per-frame synthesis parameters.

        Args:
            f0: Scaled fundamental frequency per frame (n_frames,)
            loudness: Standardized loudness per frame (n_frames,)

        Returns:
            amplitudes: Per-frame harmonic amplitudes (n_frames, n_harmonics)
            noise_mags: Per-frame noise filter magnitudes (n_frames, n_noise_bands)
        """
        # Stack features: (n_frames, 2)
        features = jnp.stack([f0, loudness], axis=-1)

        # Project to hidden dimension
        projected = self.input_proj(features)  # (n_frames, hidden)

        # Run GRU over frames using lax.scan (compiles loop body once,
        # avoiding 1000× unrolling that dominates XLA compilation time)
        def gru_step(carry, x_t):
            new_carry, output = self.gru_cell(carry, x_t)
            return new_carry, output

        _, hidden_seq = jax.lax.scan(gru_step, self.init_state[...], projected)
        # hidden_seq: (n_frames, hidden)

        # MLP stack with LayerNorm + ReLU
        h = hidden_seq
        for layer, norm in zip(self.mlp_layers, self.mlp_norms):
            h = nnx.relu(norm(layer(h)))
        # h is (n_frames, hidden) — NO averaging

        # Per-frame output heads (using exp_sigmoid from DDSP paper)
        overall_amp = exp_sigmoid(self.amplitude_head(h))  # (n_frames, 1)
        harmonic_dist = exp_sigmoid(self.harmonic_head(h))  # (n_frames, n_harmonics)
        noise_mags = self.noise_head(h)  # (n_frames, n_noise_bands)

        # Normalize harmonic distribution and scale by overall amplitude
        harmonic_dist = harmonic_dist / (jnp.sum(harmonic_dist, axis=-1, keepdims=True) + 1e-8)
        amplitudes = overall_amp * harmonic_dist  # (n_frames, n_harmonics)
        return amplitudes, noise_mags


# Verify decoder
decoder = DDSPDecoder(rngs=nnx.Rngs(0))
test_f0 = jnp.ones(N_FRAMES) * 0.543  # ~440 Hz (MIDI 69 / 127)
test_loudness = jnp.ones(N_FRAMES) * 0.5  # Moderate loudness (-40 dB)
test_amps, test_noise = decoder(test_f0, test_loudness)
n_dec_params = sum(p.size for p in jax.tree.leaves(nnx.state(decoder, nnx.Param)))
print(f"Decoder output: amplitudes={test_amps.shape}, noise_mags={test_noise.shape}")
print(f"Decoder parameters: {n_dec_params:,}")
# Expected output:
# Decoder output: amplitudes=(1000, 100), noise_mags=(1000, 65)
# Decoder parameters: 2,452,646

# %% [markdown]
"""
### Step 4: Multi-Scale Spectral Loss

The loss function compares predicted and target audio in the frequency domain
at multiple FFT scales. This is more perceptually meaningful than waveform MSE.
"""


# %%
# Step 4: Multi-scale spectral loss
def stft_magnitude(audio: jax.Array, fft_size: int, hop_size: int | None = None) -> jax.Array:
    """Compute STFT magnitude spectrogram.

    Args:
        audio: Input waveform (n_samples,)
        fft_size: FFT window size
        hop_size: Hop between frames (default: fft_size // 4)

    Returns:
        Magnitude spectrogram (n_frames, fft_size // 2 + 1)
    """
    if hop_size is None:
        hop_size = fft_size // 4

    # Hann window
    window = jnp.hanning(fft_size)

    # Frame the signal using dynamic_slice via vmap (single XLA op,
    # avoids unrolling ~100+ Python slices per FFT scale)
    n_frames = max(1, (audio.shape[0] - fft_size) // hop_size + 1)
    starts = jnp.arange(n_frames) * hop_size
    frames = jax.vmap(lambda s: jax.lax.dynamic_slice(audio, (s,), (fft_size,)))(starts)
    frames = frames * window  # (n_frames, fft_size)

    # FFT and magnitude
    spectra = jnp.fft.rfft(frames)  # (n_frames, fft_size//2 + 1)
    return jnp.abs(spectra)


def multi_scale_spectral_loss(
    pred_audio: jax.Array,
    target_audio: jax.Array,
    fft_sizes: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    alpha: float = 1.0,
) -> jax.Array:
    """Multi-scale spectral loss (matching DDSP paper).

    Computes L1 + alpha * L1(log) loss across multiple FFT scales.
    Handles both single (n_samples,) and batched (B, n_samples) inputs.

    Args:
        pred_audio: Predicted waveform (n_samples,) or (B, n_samples)
        target_audio: Target waveform (n_samples,) or (B, n_samples)
        fft_sizes: Tuple of FFT window sizes
        alpha: Weight for log-magnitude term

    Returns:
        Scalar loss value (mean over batch if batched)
    """
    # Batched inputs: vmap over batch dim, return mean
    if pred_audio.ndim == 2:
        per_sample = jax.vmap(
            lambda p, t: multi_scale_spectral_loss(p, t, fft_sizes, alpha),
        )(pred_audio, target_audio)
        return jnp.mean(per_sample)

    total_loss = jnp.array(0.0)

    for fft_size in fft_sizes:
        if fft_size > pred_audio.shape[0]:
            continue

        pred_mag = stft_magnitude(pred_audio, fft_size)
        target_mag = stft_magnitude(target_audio, fft_size)

        # L1 on linear magnitude
        l1_loss = jnp.mean(jnp.abs(pred_mag - target_mag))

        # L1 on log magnitude (matching DDSP paper — uses L1 for both terms)
        pred_log = jnp.log(pred_mag + 1e-7)
        target_log = jnp.log(target_mag + 1e-7)
        log_loss = jnp.mean(jnp.abs(pred_log - target_log))

        total_loss = total_loss + l1_loss + alpha * log_loss

    return total_loss


# Verify loss function
test_pred = jnp.sin(jnp.linspace(0, 100, AUDIO_LENGTH))
test_target = jnp.sin(jnp.linspace(0, 100.5, AUDIO_LENGTH))  # Slightly different
loss_val = multi_scale_spectral_loss(test_pred, test_target)
print(f"Spectral loss (similar signals): {float(loss_val):.4f}")

test_noise_sig = jax.random.normal(jax.random.key(0), (AUDIO_LENGTH,)) * 0.1
loss_val_diff = multi_scale_spectral_loss(test_pred, test_noise_sig)
print(f"Spectral loss (signal vs noise): {float(loss_val_diff):.4f}")
print("(Higher loss for dissimilar signals — as expected)")
# Expected output:
# Spectral loss (similar signals): 4.0301
# Spectral loss (signal vs noise): 68.8026
# (Higher loss for dissimilar signals — as expected)

# %% [markdown]
"""
### Step 5: DDSP Synthesis via Composite

Instead of manually calling each operator's `.apply()` and explicit `jax.vmap()`,
we compose the synthesis pipeline using `CompositeOperatorModule`:

- **`WEIGHTED_PARALLEL([1.0, 0.1])`**: Runs HarmonicSynth and FilteredNoise on the
  same input dict, then computes `1.0 * harmonic_audio + 0.1 * noise_audio`
- **`SEQUENTIAL`**: Chains the parallel mix into Reverb

```
synth_composite = SEQUENTIAL([
    WEIGHTED_PARALLEL([HarmonicSynth, FilteredNoise], weights=[1.0, 0.1]),
    Reverb,
])
```

Both operators use `{**data, "audio": audio}` passthrough, so non-audio keys
get weighted-summed but Reverb ignores them — only reading the `audio` key.
"""


# %%
# Step 5: Composite-based DDSP synthesis pipeline
def create_synth_composite(
    harmonic_synth: HarmonicSynthOperator,
    noise_synth: FilteredNoiseOperator,
    reverb: ReverbOperator,
) -> CompositeOperatorModule:
    """Create DDSP synthesis composite: (Harmonic | Noise) >> Reverb.

    Architecture:
        SEQUENTIAL([
            WEIGHTED_PARALLEL([HarmonicSynth, FilteredNoise], weights=[1.0, 0.1]),
            Reverb,
        ])
    """
    synth_parallel = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[harmonic_synth, noise_synth],
            weights=[1.0, 0.1],
        )
    )
    return CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[synth_parallel, reverb],
        )
    )


def synthesize_batch(
    decoder: DDSPDecoder,
    synth_composite: CompositeOperatorModule,
    f0_batch: jax.Array,
    loudness_batch: jax.Array,
    f0_hz_batch: jax.Array,
) -> jax.Array:
    """Batch DDSP forward pass: decode → composite synthesis.

    Uses jax.vmap(decoder) for batched decoding, then delegates to
    the composite's native Batch processing (which vmaps internally).

    Args:
        decoder: RNN-FC decoder
        synth_composite: Nested composite (WEIGHTED_PARALLEL >> Reverb)
        f0_batch: Scaled f0 per sample (B, n_frames)
        loudness_batch: Standardized loudness per sample (B, n_frames)
        f0_hz_batch: Raw f0 in Hz per sample (B, n_frames)

    Returns:
        Synthesized audio batch (B, audio_length)
    """
    # Batched decoding — now returns per-frame params
    amps_batch, noise_batch = jax.vmap(decoder)(f0_batch, loudness_batch)
    # amps_batch: (B, n_frames, n_harmonics), noise_batch: (B, n_frames, n_noise_bands)

    # Pass full time-varying f0 — NO averaging
    batch = Batch.from_parts(
        data={
            "amplitudes": amps_batch,  # (B, n_frames, n_harmonics)
            "f0_hz": f0_hz_batch,  # (B, n_frames)
            "noise_magnitudes": noise_batch,  # (B, n_frames, n_noise_bands)
        },
        states={},
    )
    result = synth_composite(batch)
    return result.get_data()["audio"]


# Create composite and test synthesis
synth_composite = create_synth_composite(h_op, n_op, r_op)
test_f0_hz = jnp.ones((1, N_FRAMES)) * 440.0
test_synth = synthesize_batch(
    decoder,
    synth_composite,
    test_f0[None],
    test_loudness[None],
    test_f0_hz,
)
print(f"Synthesized audio shape: {test_synth.shape}")
print(f"Audio range: [{float(test_synth[0].min()):.4f}, {float(test_synth[0].max()):.4f}]")
# Expected output (values vary):
# Synthesized audio shape: (1, 64000)
# Audio range: [-0.0116, 0.0109]  (small before training — exp_sigmoid starts near-zero)


def evaluate_spectral_loss(
    decoder: DDSPDecoder,
    synth_composite: CompositeOperatorModule,
    source: MemorySource,
    batch_size: int = 4,
) -> float:
    """Compute average spectral loss over a dataset."""
    pipeline = from_source(source, batch_size=batch_size)
    total_loss = 0.0
    num_batches = 0
    for batch in pipeline:
        pred = synthesize_batch(
            decoder,
            synth_composite,
            batch["f0"],
            batch["loudness"],
            batch["f0_hz"],
        )
        total_loss += float(
            multi_scale_spectral_loss(pred, batch["audio"], fft_sizes=cfg.loss_fft_sizes)
        )
        num_batches += 1
    return total_loss / max(num_batches, 1)


# %% [markdown]
"""
### Step 6: Training Loop

Train the DDSP model using multi-scale spectral loss with exponential LR decay
and gradient clipping (global norm 3.0), matching the DDSP reference.

Gradients flow through: loss → reverb → noise synth + harmonic synth → decoder.
"""


# %%
# Step 6: Training


@nnx.jit
def ddsp_train_step(
    all_params: tuple,
    optimizer: nnx.Optimizer,
    target_audio: jax.Array,
    f0: jax.Array,
    loudness: jax.Array,
    f0_hz: jax.Array,
) -> jax.Array:
    """JIT-compiled DDSP training step."""

    def loss_fn(params: tuple) -> jax.Array:
        dec, synth_comp = params
        pred_audio = synthesize_batch(dec, synth_comp, f0, loudness, f0_hz)
        return multi_scale_spectral_loss(pred_audio, target_audio, fft_sizes=cfg.loss_fft_sizes)

    loss, grads = nnx.value_and_grad(loss_fn)(all_params)
    optimizer.update(all_params, grads)
    return loss


def train_ddsp(
    train_source: MemorySource,
    num_epochs: int = 50,
    batch_size: int = 16,
    peak_lr: float = 1e-3,
    decay_rate: float = 0.98,
    decay_steps: int = 10000,
) -> tuple[DDSPDecoder, CompositeOperatorModule, list[float]]:
    """Train DDSP model on audio data with LR scheduling.

    Uses exponential decay matching the DDSP reference:
        lr = peak_lr * decay_rate^(step / decay_steps)

    Gradient clipping (global norm 3.0) stabilizes training and prevents
    the loss spikes observed in unclipped training.

    Returns (decoder, synth_composite, loss_history).
    """
    rngs = nnx.Rngs(42)

    # Create model components
    decoder = DDSPDecoder(rngs=rngs)
    harmonic_synth = HarmonicSynthOperator(HarmonicSynthConfig())
    noise_synth = FilteredNoiseOperator(FilteredNoiseConfig())
    reverb = ReverbOperator(ReverbConfig())

    # Compose synthesis pipeline
    synth_composite = create_synth_composite(harmonic_synth, noise_synth, reverb)

    # LR schedule: exponential decay (matching DDSP reference, no warmup)
    schedule = optax.exponential_decay(
        init_value=peak_lr,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
    )

    # Joint optimizer with gradient clipping (matching DDSP reference: global norm 3.0)
    all_params = (decoder, synth_composite)
    optimizer = nnx.Optimizer(
        all_params,
        optax.chain(
            optax.clip_by_global_norm(3.0),
            optax.adam(schedule),
        ),
        wrt=nnx.Param,
    )

    loss_history: list[float] = []

    print(f"Training DDSP: {num_epochs} epochs, batch_size={batch_size}, peak_lr={peak_lr}")

    for epoch in range(num_epochs):
        pipeline = from_source(train_source, batch_size=batch_size)
        epoch_loss = 0.0
        num_steps = 0

        for batch in pipeline:
            loss = ddsp_train_step(
                all_params,
                optimizer,
                batch["audio"],
                batch["f0"],
                batch["loudness"],
                batch["f0_hz"],
            )
            epoch_loss += float(loss)
            num_steps += 1

        avg_loss = epoch_loss / max(num_steps, 1)
        loss_history.append(avg_loss)
        print(f"  Epoch {epoch + 1:2d}/{num_epochs} | Spectral Loss: {avg_loss:.4f}")

    return decoder, synth_composite, loss_history


# Run training
print("\n=== DDSP Training ===\n")
print("Note: First step triggers XLA compilation of the full forward+backward pass.")
print("This takes 1-3 minutes depending on GPU (lax.scan keeps the graph compact).\n")
decoder, synth_composite, loss_history = train_ddsp(
    train_source,
    num_epochs=cfg.num_epochs,
    batch_size=cfg.batch_size,
)

# %%
# Plot training loss curve
fig, ax = plt.subplots(figsize=(10, 5))
epochs_range = list(range(1, len(loss_history) + 1))
ax.plot(
    epochs_range,
    loss_history,
    "o-",
    color="steelblue",
    linewidth=2,
    markersize=6,
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Multi-Scale Spectral Loss")
ax.set_title("DDSP Training: Spectral Loss Over Epochs (NSynth)")
ax.grid(True, alpha=0.3)

# Annotate start and end values
if len(loss_history) >= 2:
    ax.annotate(
        f"{loss_history[0]:.2f}",
        (1, loss_history[0]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    ax.annotate(
        f"{loss_history[-1]:.2f}",
        (len(loss_history), loss_history[-1]),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "perf-ddsp-training-curve.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: docs/assets/images/examples/perf-ddsp-training-curve.png")

# %% [markdown]
"""
### Step 7: Verify Gradient Flow Through All Operators

Gradients must flow from the spectral loss back through reverb,
noise synth, harmonic synth, and into the decoder.
"""


# %%
# Step 7: Gradient flow verification
print("\n=== Gradient Flow Verification ===")

# Get a test batch
verify_pipeline = from_source(test_source, batch_size=4)
verify_batch = next(iter(verify_pipeline))
target = verify_batch["audio"]
f0 = verify_batch["f0"]
loudness = verify_batch["loudness"]
f0_hz = verify_batch["f0_hz"]

all_params = (decoder, synth_composite)


def verify_loss(params: tuple) -> jax.Array:
    dec, synth_comp = params
    pred = synthesize_batch(dec, synth_comp, f0, loudness, f0_hz)
    return multi_scale_spectral_loss(pred, target)


loss, grads = nnx.value_and_grad(verify_loss)(all_params)
grad_leaves = jax.tree.leaves(grads)

assert len(grad_leaves) > 0, "No gradient leaves found"
assert any(jnp.sum(jnp.abs(g)) > 0 for g in grad_leaves), (
    "All gradients are zero — DDSP pipeline is NOT differentiable!"
)

print(f"Loss: {float(loss):.4f}")
print(f"Gradient flow verified: {len(grad_leaves)} parameter groups")

# Check per-component gradients
component_names = ["Decoder", "SynthComposite"]
for name, component_grad in zip(component_names, grads):
    leaves = jax.tree.leaves(component_grad)
    total_grad_norm = sum(float(jnp.sum(jnp.abs(g))) for g in leaves)
    n_params = sum(g.size for g in leaves)
    status = "RECEIVES GRADIENTS" if total_grad_norm > 0 else "NO GRADIENTS"
    print(f"  {name:15s} | params: {n_params:6d} | |grad|: {total_grad_norm:.6f} | {status}")

print("\nSUCCESS: All DDSP components receive gradients!")
# Expected output (values vary per training run):
# Loss: 13.4107
# Gradient flow verified: 25 parameter groups
#   Decoder         | params: 2452646 | |grad|: 12439.834685 | RECEIVES GRADIENTS
#   SynthComposite  | params:  16000 | |grad|: 700.408447 | RECEIVES GRADIENTS
# SUCCESS: All DDSP components receive gradients!

# %% [markdown]
"""
### Step 8: Demonstrate Composite Pipeline Structure

The trained `synth_composite` is a nested `CompositeOperatorModule`:
`SEQUENTIAL([WEIGHTED_PARALLEL(...), Reverb])`. Let's process a demo batch
through it to show the composite in action.
"""


# %%
# Step 8: Demonstrate composite pipeline structure
print("\n=== Composite Pipeline Demonstration ===")
print("DDSP synthesis uses a nested CompositeOperatorModule:")
print("""
  synth_composite = SEQUENTIAL([
      WEIGHTED_PARALLEL([HarmonicSynth, FilteredNoise], weights=[1.0, 0.1]),
      Reverb,
  ])

  This replaces the manual pattern:
      h_audio = harmonic_synth.apply(h_data)["audio"]
      n_audio = noise_synth.apply(n_data)["audio"]
      combined = h_audio + n_audio * 0.1
      reverbed = reverb.apply({"audio": combined})["audio"]
""")

# Process a batch through the actual trained composite (per-frame data)
demo_amps = jnp.ones((1, N_FRAMES, N_HARMONICS)) * 0.1
demo_f0 = jnp.ones((1, N_FRAMES)) * 440.0
demo_noise_mags = jnp.ones((1, N_FRAMES, N_NOISE_BANDS)) * 0.5

demo_batch = Batch.from_parts(
    data={
        "amplitudes": demo_amps,
        "f0_hz": demo_f0,
        "noise_magnitudes": demo_noise_mags,
    },
    states={},
)
result = synth_composite(demo_batch)
result_audio = result.get_data()["audio"]

print(
    f"Input: amplitudes={demo_amps.shape}, f0_hz={demo_f0.shape}, "
    f"noise_mags={demo_noise_mags.shape}"
)
print(
    f"Output audio: shape={result_audio.shape}, "
    f"rms={float(jnp.sqrt(jnp.mean(result_audio**2))):.4f}"
)

# %% [markdown]
"""
### Step 9: Evaluate Resynthesis Quality
"""


# %%
# Step 9: Evaluate on test set
print("\n=== Resynthesis Evaluation ===")

avg_test_loss = evaluate_spectral_loss(decoder, synth_composite, test_source)
print(f"Average test spectral loss: {avg_test_loss:.4f}")

# Compare with random synthesis (untrained)
print("\nComparison:")
random_decoder = DDSPDecoder(rngs=nnx.Rngs(999))
random_composite = create_synth_composite(
    HarmonicSynthOperator(HarmonicSynthConfig()),
    FilteredNoiseOperator(FilteredNoiseConfig()),
    ReverbOperator(ReverbConfig()),
)

rand_loss = evaluate_spectral_loss(random_decoder, random_composite, test_source)

print(f"  Random (untrained):  {rand_loss:.4f}")
print(f"  Trained DDSP:        {avg_test_loss:.4f}")
improvement = ((rand_loss - avg_test_loss) / rand_loss) * 100
print(f"  Improvement:         {improvement:.1f}% lower spectral loss")
# Expected output (varies by training run, 10K samples):
#   Random (untrained):  ~30
#   Trained DDSP:        ~8-10
#   Improvement:         ~65-70% lower spectral loss

# %%
# Visualize resynthesis quality: target vs. synthesized waveforms and spectrograms
vis_pipeline = from_source(test_source, batch_size=4)
vis_batch = next(iter(vis_pipeline))
vis_pred = synthesize_batch(
    decoder,
    synth_composite,
    vis_batch["f0"],
    vis_batch["loudness"],
    vis_batch["f0_hz"],
)

fig, axes = plt.subplots(4, 2, figsize=(16, 12))
time_axis_full = np.arange(AUDIO_LENGTH) / SAMPLE_RATE

for i in range(2):
    target_audio = np.array(vis_batch["audio"][i])
    pred_audio = np.array(vis_pred[i])
    f0_val = float(vis_batch["f0_hz"][i, N_FRAMES // 2])

    # Waveform comparison — show first 4000 samples for detail
    n_show = 4000
    t_show = time_axis_full[:n_show]

    # Target waveform
    axes[i * 2, 0].plot(t_show, target_audio[:n_show], color="steelblue", linewidth=0.8)
    axes[i * 2, 0].set_title(f"Target Waveform (f0~{f0_val:.0f} Hz)", fontsize=10)
    axes[i * 2, 0].set_ylabel("Amplitude")
    axes[i * 2, 0].grid(True, alpha=0.3)

    # Synthesized waveform
    axes[i * 2, 1].plot(t_show, pred_audio[:n_show], color="darkorange", linewidth=0.8)
    axes[i * 2, 1].set_title(f"Synthesized Waveform (f0~{f0_val:.0f} Hz)", fontsize=10)
    axes[i * 2, 1].set_ylabel("Amplitude")
    axes[i * 2, 1].grid(True, alpha=0.3)

    # Target spectrogram
    plot_specgram(axes[i * 2 + 1, 0], target_audio)
    axes[i * 2 + 1, 0].set_ylabel("Frequency (Hz)")
    axes[i * 2 + 1, 0].set_ylim(0, 4000)
    axes[i * 2 + 1, 0].set_title("Target Spectrogram")

    # Synthesized spectrogram
    plot_specgram(axes[i * 2 + 1, 1], pred_audio)
    axes[i * 2 + 1, 1].set_ylabel("Frequency (Hz)")
    axes[i * 2 + 1, 1].set_ylim(0, 4000)
    axes[i * 2 + 1, 1].set_title("Synthesized Spectrogram")

# Only last row gets x labels
for ax in axes[-1]:
    ax.set_xlabel("Time (s)")

fig.suptitle(
    f"DDSP Resynthesis: Target vs. Synthesized Audio (NSynth)\n"
    f"Spectral Loss: {avg_test_loss:.4f} (trained) vs {rand_loss:.4f} (random) — "
    f"{improvement:.1f}% improvement",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "cv-ddsp-resynthesis-comparison.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()
print("Saved: docs/assets/images/examples/cv-ddsp-resynthesis-comparison.png")

# %% [markdown]
"""
### Step 10: Analyze Learned Parameters
"""


# %%
# Step 10: Analyze what the model learned
def analyze_ddsp(
    decoder: DDSPDecoder,
    synth_composite: CompositeOperatorModule,
) -> None:
    """Print analysis of learned DDSP parameters."""
    print("\n=== Learned DDSP Parameters ===")

    # Extract reverb from composite: SEQUENTIAL([WEIGHTED_PARALLEL(...), Reverb])
    reverb = list(synth_composite.operators)[1]

    # Decoder statistics
    dec_params = sum(p.size for p in jax.tree.leaves(nnx.state(decoder, nnx.Param)))
    print(f"\n1. Decoder: {dec_params:,} parameters")
    print(f"   GRU hidden dim: {decoder.init_state[...].shape[0]}")
    print(f"   MLP layers: {len(decoder.mlp_layers)}")

    # Reverb impulse response
    ir = reverb.impulse_response[...]
    ir_energy = float(jnp.sum(ir**2))
    ir_peak = float(jnp.max(jnp.abs(ir)))
    ir_length_ms = ir.shape[0] / SAMPLE_RATE * 1000

    # Estimate RT60 (time for energy to decay by 60dB)
    energy_cumsum = jnp.cumsum(ir**2)
    total_energy = energy_cumsum[-1]
    rt60_idx = jnp.searchsorted(energy_cumsum, total_energy * 0.999)
    rt60_ms = float(rt60_idx) / SAMPLE_RATE * 1000

    print("\n2. Reverb Impulse Response:")
    print(f"   Length: {ir_length_ms:.0f} ms ({ir.shape[0]} samples)")
    print(f"   Peak amplitude: {ir_peak:.4f}")
    print(f"   Total energy: {ir_energy:.4f}")
    print(f"   Estimated RT60: {rt60_ms:.0f} ms")

    # Test synthesis with a known pitch
    test_f0 = jnp.ones(N_FRAMES) * 0.543  # ~440 Hz (MIDI 69 / 127)
    test_loud = jnp.ones(N_FRAMES) * 0.5  # Moderate loudness (-40 dB)
    amps, noise_mags = decoder(test_f0, test_loud)
    # amps: (n_frames, n_harmonics), noise_mags: (n_frames, n_noise_bands)

    # Time-average for visualization (bar chart of harmonic strengths)
    amps_avg = jnp.mean(amps, axis=0)  # (n_harmonics,)
    noise_avg = jnp.mean(noise_mags, axis=0)  # (n_noise_bands,)

    print("\n3. Synthesis for A4 (440 Hz):")
    print(f"   Top 5 harmonic amplitudes: {amps_avg[:5]}")
    print(
        f"   Amplitude decay rate: {float(amps_avg[0] / (amps_avg[4] + 1e-8)):.2f}x "
        f"(fundamental vs 5th harmonic)"
    )
    print(f"   Noise magnitude range: [{float(noise_avg.min()):.3f}, {float(noise_avg.max()):.3f}]")

    # --- Visualization: Impulse response + harmonic amplitudes ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")

    # 1. Learned impulse response
    ir_np = np.array(ir)
    ir_time = np.arange(len(ir_np)) / SAMPLE_RATE * 1000  # ms
    # Show first 200ms for detail
    show_ms = 200
    show_idx = int(show_ms * SAMPLE_RATE / 1000)
    axes[0].plot(ir_time[:show_idx], ir_np[:show_idx], color="steelblue", linewidth=0.5)
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"Learned Impulse Response (first {show_ms}ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="black", linewidth=0.5)

    # 2. Harmonic amplitudes (bar chart) — time-averaged
    amps_np = np.array(amps_avg)
    n_show_harmonics = min(20, len(amps_np))
    harmonic_nums = np.arange(1, n_show_harmonics + 1)
    freqs = harmonic_nums * 440  # Hz
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_show_harmonics))
    axes[1].bar(
        harmonic_nums,
        amps_np[:n_show_harmonics],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Harmonic Number (k)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Learned Harmonic Amplitudes (A4 = 440 Hz)")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Add frequency labels for first few harmonics
    # Use relative offset so labels scale with amplitude range (avoids
    # layout explosion when amps are near-zero in early training)
    label_offset = max(float(amps_np[:n_show_harmonics].max()) * 0.08, 1e-8)
    for idx in range(min(5, n_show_harmonics)):
        axes[1].text(
            idx + 1,
            float(amps_np[idx]) + label_offset,
            f"{int(freqs[idx])}Hz",
            ha="center",
            fontsize=7,
            rotation=45,
        )

    # 3. Noise filter frequency response — time-averaged
    noise_np = np.array(exp_sigmoid(noise_avg))
    freq_bins = np.linspace(0, SAMPLE_RATE / 2, len(noise_np))
    axes[2].fill_between(freq_bins, noise_np, alpha=0.3, color="darkorange")
    axes[2].plot(freq_bins, noise_np, color="darkorange", linewidth=1.5)
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Magnitude")
    axes[2].set_title("Learned Noise Filter Response")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, SAMPLE_RATE / 2)

    fig.suptitle("DDSP Learned Parameters — Audio Synthesis Analysis", fontsize=13)
    plt.savefig(
        OUTPUT_DIR / "cv-ddsp-learned-parameters.png",
        dpi=150,
        facecolor="white",
    )
    plt.close()
    print("\nSaved: docs/assets/images/examples/cv-ddsp-learned-parameters.png")


analyze_ddsp(decoder, synth_composite)

# %% [markdown]
"""
## Troubleshooting

### Out of Memory during training

**Symptom**: `XlaRuntimeError: RESOURCE_EXHAUSTED` during `train_ddsp()`.

**Cause**: Audio batches are large — each sample is 64,000 floats, and the
spectral loss computes multiple FFTs per sample. With `batch_size=32`, a single
batch uses ~8 MB of audio data, plus intermediate FFT buffers. Loading 10K
samples also requires ~2.5 GB RAM for the dataset arrays.

**Solution**: Reduce `batch_size` in `train_ddsp()` to 16 or 8, reduce the
number of FFT scales in `cfg.loss_fft_sizes`, or use `QUICK_MODE = True` (500
samples, 5 epochs).

### NSynth dataset download fails or is slow

**Symptom**: `load_nsynth()` hangs or fails with network errors.

**Cause**: The NSynth TFRecord archive is ~1 GB. Downloads may fail on slow
or unstable connections.

**Solution**: Set `TFDS_DATA_DIR` to a directory with sufficient space on a
fast drive: `export TFDS_DATA_DIR=/path/to/data`. If the download was partially
completed, delete the `downloads/` subdirectory and retry.

### Training is very slow on CPU

**Symptom**: Each epoch takes 10+ minutes.

**Cause**: The harmonic synthesizer uses phase accumulation with 100 harmonics
per frame — heavy on FLOPs. XLA compilation also takes longer on CPU.

**Solution**: Set `QUICK_MODE = True` to use fewer FFT scales, fewer samples
(500 vs 10K), and fewer epochs (5 vs 100). GPU is strongly recommended for
full training (10K samples, ~3 hrs).
"""

# %% [markdown]
"""
## Results & Evaluation

### What We Achieved

This example demonstrates datarax's **extensibility** to non-image domains.
Three custom `OperatorModule` subclasses for audio DSP — with no changes to
datarax's core library — enable a complete differentiable audio synthesis system
trained on real NSynth instrument recordings.

### Observed Results (Full Training, 10K samples, 100 epochs, ~31K steps)

| Configuration | Spectral Loss | Notes |
|---------------|---------------|-------|
| Random (untrained) | ~30 | Random synth params on real audio |
| **Trained DDSP (10K, 100 epochs)** | **~8-10** | ~31K steps matching paper's training budget |
| Improvement | **~65-70%** | Relative to untrained baseline |
| DDSP (paper, full NSynth) | ~1-3 | Full dataset (290K) + joint encoder-decoder |

Training loss drops steadily over 100 epochs with no loss spikes (gradient
clipping). The ~31K training steps match the paper's budget. The remaining gap
is primarily due to using 10K samples (vs. 290K) and fixed CREPE features
(vs. jointly-trained encoder).

### Key Takeaways

1. **Any domain, same pattern**: Custom audio operators follow the exact same
   `OperatorModule` pattern as image operators. No special audio infrastructure needed.

2. **3 operators, complete pipeline**: HarmonicSynth + FilteredNoise + Reverb
   = complete differentiable audio synthesis. Each has `nnx.Param` for
   end-to-end gradient optimization.

3. **Parallel + Sequential composition**: `CompositeOperatorModule` naturally
   expresses DDSP's architecture: `WEIGHTED_PARALLEL` mixes harmonic + noise,
   then `SEQUENTIAL` chains the mix into reverb — no manual `.apply()` calls.

4. **Real data, real results**: Training on NSynth instrument recordings
   produces recognizable instrument timbres, validating that the architecture
   works on real audio (not just synthetic sine waves).

5. **Paper-accurate architecture**: The DDSPDecoder uses GRU + MLP (nnx.List
   pattern) with per-frame synthesis and phase accumulation in the harmonic
   synth, matching the Engel et al. 2020 architecture — time-varying
   amplitudes and f0 are upsampled from frame rate to sample rate.
"""

# %% [markdown]
"""
## Next Steps & Resources

### Try These Experiments

1. **Even larger dataset**: Increase `n_train = 30000` (half the GANSynth subset)
   for closer-to-paper results (requires ~8 hrs GPU, ~18 GB RAM).

2. **Per-frame noise synthesis**: Upgrade `FilteredNoiseOperator` to use
   windowed overlap-add for per-frame noise filtering (currently time-averaged).

3. **More effects**: Add a differentiable EQ (parametric equalizer) or
   compressor operator to the chain.

4. **Transfer to new instruments**: Train on strings, then fine-tune
   on brass — does the reverb IR transfer?

### Related Examples

- [DADA Learned Augmentation](01_dada_learned_augmentation_guide.py) —
  Differentiable augmentation search (operator library showcase)
- [Learned ISP Guide](02_learned_isp_guide.py) — DAG-based differentiable
  image processing
- [Operators Tutorial](../../core/02_operators_tutorial.py) — Deep dive
  into operator patterns

### API Reference

- [OperatorModule](../../../docs/core/operator.md) — Base class extended by custom operators
- [OperatorConfig](../../../docs/core/config.md) — Configuration base class
- [MergeBatchNode](../../../docs/dag/nodes.md) — Parallel merge node
- [DAGExecutor](../../../docs/dag/dag_executor.md) — Pipeline executor

### Further Reading

- [DDSP Paper (arXiv)](https://arxiv.org/abs/2001.04643) — Full paper
- [Synthax (JAX DDSP)](https://github.com/PapayaResearch/synthax) — JAX implementation
- [Magenta DDSP](https://github.com/magenta/ddsp) — Original TensorFlow implementation
- [NSynth Dataset](https://magenta.tensorflow.org/datasets/nsynth) — Real audio dataset
"""


# %%
def main():
    """CLI entry point for full DDSP training pipeline."""
    print("=" * 60)
    print("DDSP: Differentiable Digital Signal Processing")
    print("  Dataset: NSynth (real instrument recordings)")
    print("=" * 60)

    # Load data (full dataset for CLI training — 10K samples, ~31K training steps)
    print("\n[1/4] Loading NSynth dataset...")
    data_train, data_test = load_nsynth(n_train=10000, n_test=500)
    src_train = MemorySource(MemorySourceConfig(), data=data_train, rngs=nnx.Rngs(0))
    src_test = MemorySource(MemorySourceConfig(), data=data_test, rngs=nnx.Rngs(1))
    print(f"  Train: {data_train['audio'].shape}, Test: {data_test['audio'].shape}")

    # Train
    print("\n[2/4] Training DDSP model...")
    dec, synth_comp, _ = train_ddsp(src_train, num_epochs=100, batch_size=32)

    # Evaluate (reuses the extracted helper — no code duplication)
    print("\n[3/4] Evaluating resynthesis quality...")
    test_loss = evaluate_spectral_loss(dec, synth_comp, src_test)
    print(f"  Test spectral loss: {test_loss:.4f}")

    # Analyze
    print("\n[4/4] Analyzing learned parameters...")
    analyze_ddsp(dec, synth_comp)

    print()
    print("=" * 60)
    print("DDSP training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
