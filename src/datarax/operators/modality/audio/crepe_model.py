"""CrepeModel — Flax NNX port of the CREPE pitch detection CNN.

CREPE (Convolutional Representation for Pitch Estimation) is a deep CNN that
estimates pitch from raw audio. This module provides a pure JAX/Flax NNX
reimplementation with pretrained Keras weight loading.

All weights are nnx.Param, enabling fine-tuning during end-to-end training.
Mode switching via model.train() / model.eval() for BatchNorm + Dropout.

Architecture: 6 × (Conv1D + BatchNorm + ReLU + Dropout + MaxPool) + Dense + Sigmoid
Output: 360-bin pitch probability distribution (20 cents/bin, ~32 Hz to ~1975 Hz)
"""

import pathlib

import jax
import jax.numpy as jnp
from flax import nnx

# CREPE pitch bins: 360 bins spanning 1997.38 cents to 9177.38 cents
# Each bin is 20 cents wide. Cents = 1200 * log2(f / f_ref)
CENTS_PER_BIN = 20
CENTS_MAPPING = jnp.linspace(0, 7180, 360) + 1997.3794084376191

# Model capacity multipliers (scale channel counts)
_CAPACITY_MULTIPLIERS = {
    "tiny": 4,
    "small": 8,
    "medium": 16,
    "large": 24,
    "full": 32,
}

_CREPE_CACHE_DIR = pathlib.Path.home() / ".cache" / "datarax" / "crepe"


def max_pool_1d(x: jax.Array, window: int = 2, stride: int = 2) -> jax.Array:
    """1D max pooling via reduce_window (VALID padding, no overflow).

    Uses jax.lax.reduce_window with lax.max computation.

    Args:
        x: Input array shape (batch, length, channels).
        window: Pooling window size.
        stride: Pooling stride.

    Returns:
        Pooled array shape (batch, length // stride, channels).
    """
    return jax.lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=(1, window, 1),
        window_strides=(1, stride, 1),
        padding="VALID",
    )


# torchcrepe padding specs (matching their F.pad calls exactly):
# Layer 1: pad (254, 254) on height dim — kernel=512, stride=4
# Layers 2-6: pad (31, 32) on height dim — kernel=64, stride=1 (asymmetric!)
_LAYER_PADDINGS = [
    (254, 254),  # Layer 1
    (31, 32),  # Layer 2
    (31, 32),  # Layer 3
    (31, 32),  # Layer 4
    (31, 32),  # Layer 5
    (31, 32),  # Layer 6
]


class CrepeModel(nnx.Module):
    """CREPE pitch detection CNN ported to Flax NNX.

    Faithful port of torchcrepe's architecture:
    - 6 conv blocks: pad → Conv → ReLU → BatchNorm → MaxPool
    - No dropout in forward pass (torchcrepe omits it)
    - Explicit asymmetric padding (not SAME)
    - All weights are nnx.Param (learnable for fine-tuning)

    Use model.train()/model.eval() to switch BatchNorm modes.

    Args:
        capacity: Model size variant ("tiny", "small", "medium", "large", "full").
        rngs: Flax NNX random number generators.
    """

    def __init__(self, capacity: str = "full", *, rngs: nnx.Rngs):
        super().__init__()

        if capacity not in _CAPACITY_MULTIPLIERS:
            raise ValueError(
                f"capacity must be one of {list(_CAPACITY_MULTIPLIERS)}, got {capacity!r}"
            )

        mult = _CAPACITY_MULTIPLIERS[capacity]

        # Channel configuration (full model: [1024, 128, 128, 128, 256, 512])
        out_channels = [mult * 32, mult * 4, mult * 4, mult * 4, mult * 8, mult * 16]
        in_channels = [1, *out_channels[:-1]]
        kernel_sizes = [512, 64, 64, 64, 64, 64]
        strides = [4, 1, 1, 1, 1, 1]

        # nnx.List for dynamic layer collections
        self.conv_layers = nnx.List([])
        self.batch_norms = nnx.List([])
        # Keep dropouts for API compatibility (train/eval tests) but they're inactive
        self.dropouts = nnx.List([])

        for i in range(6):
            # VALID padding — we handle padding explicitly in __call__
            self.conv_layers.append(
                nnx.Conv(
                    in_features=in_channels[i],
                    out_features=out_channels[i],
                    kernel_size=(kernel_sizes[i],),
                    strides=(strides[i],),
                    padding="VALID",
                    rngs=rngs,
                )
            )
            # BatchNorm: eps and momentum match torchcrepe exactly
            self.batch_norms.append(
                nnx.BatchNorm(
                    num_features=out_channels[i],
                    momentum=0.0,
                    epsilon=0.0010000000474974513,
                    use_running_average=False,
                    rngs=rngs,
                )
            )
            self.dropouts.append(
                nnx.Dropout(
                    rate=0.25,
                    deterministic=False,
                    rngs=rngs,
                )
            )

        # Dimension trace with torchcrepe-exact padding (VALID conv, explicit pad):
        # Input: (B, 1024, 1)
        # Pad(254,254): (B, 1532, 1)  Conv1(k512,s4,VALID): (B, 256, 1024)  Pool: (B, 128, 1024)
        # Pad(31,32):   (B, 191, 1024) Conv2(k64,s1,VALID): (B, 128, 128)    Pool: (B, 64, 128)
        # Pad(31,32):   (B, 127, 128)  Conv3(k64,s1,VALID): (B, 64, 128)     Pool: (B, 32, 128)
        # Pad(31,32):   (B, 95, 128)   Conv4(k64,s1,VALID): (B, 32, 128)     Pool: (B, 16, 128)
        # Pad(31,32):   (B, 79, 128)   Conv5(k64,s1,VALID): (B, 16, 256)     Pool: (B, 8, 256)
        # Pad(31,32):   (B, 71, 256)   Conv6(k64,s1,VALID): (B, 8, 512)      Pool: (B, 4, 512)
        # Flatten: B × 4 × 512 = B × 2048
        flatten_dim = 4 * out_channels[-1]

        self.classifier = nnx.Linear(
            in_features=flatten_dim,
            out_features=360,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through CREPE.

        Matches torchcrepe exactly: pad → conv(VALID) → ReLU → BN → maxpool.

        Args:
            x: Audio frames shape (batch, 1024, 1).

        Returns:
            Pitch probability distribution shape (batch, 360), values in [0, 1].
        """
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # Explicit padding matching torchcrepe F.pad
            pad_left, pad_right = _LAYER_PADDINGS[i]
            x = jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)))
            x = conv(x)
            x = nnx.relu(x)
            x = bn(x)
            x = max_pool_1d(x, window=2, stride=2)

        x = x.reshape(x.shape[0], -1)  # Flatten → (B, flatten_dim)
        return nnx.sigmoid(self.classifier(x))  # (B, 360)


# ============================================================================
# Pitch Decoding
# ============================================================================


def decode_pitch_local(probs: jax.Array, window: int = 9) -> tuple[jax.Array, jax.Array]:
    """Decode pitch using local weighted average around the peak.

    Non-differentiable (uses argmax). Higher accuracy for standalone inference.

    Args:
        probs: Probability distribution shape (360,).
        window: Window size for local averaging around peak.

    Returns:
        (f0_hz, confidence) — pitch in Hz and max probability.
    """
    center = jnp.argmax(probs)
    confidence = jnp.max(probs)

    # Local weighted average around peak
    half_w = window // 2
    # Clamp indices to valid range
    start = jnp.maximum(center - half_w, 0)
    end = jnp.minimum(center + half_w + 1, 360)

    # Create mask for valid bins
    indices = jnp.arange(360)
    mask = (indices >= start) & (indices < end)
    local_probs = jnp.where(mask, probs, 0.0)

    # Weighted average of cents values
    total_weight = jnp.sum(local_probs) + 1e-8
    f0_cents = jnp.sum(local_probs * CENTS_MAPPING) / total_weight

    # Convert cents to Hz
    f0_hz = 10.0 * 2.0 ** (f0_cents / 1200.0)
    return f0_hz, confidence


def decode_pitch_differentiable(
    probs: jax.Array,
    temperature: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Decode pitch using global weighted average (fully differentiable).

    Uses temperature-scaled softmax as a continuous relaxation of argmax.
    As temperature → 0, converges to argmax behavior.

    Args:
        probs: Probability distribution shape (360,).
        temperature: Temperature for softmax sharpening (lower = sharper).

    Returns:
        (f0_hz, confidence) — pitch in Hz and entropy-based confidence.
    """
    # Sharpen distribution with temperature
    sharpened = jax.nn.softmax(jnp.log(probs + 1e-8) / temperature, axis=-1)

    # Global weighted average
    f0_cents = jnp.sum(sharpened * CENTS_MAPPING, axis=-1)
    f0_hz = 10.0 * 2.0 ** (f0_cents / 1200.0)

    # Entropy-based confidence (low entropy = high confidence)
    entropy = -jnp.sum(sharpened * jnp.log(sharpened + 1e-8), axis=-1)
    max_entropy = jnp.log(360.0)
    confidence = 1.0 - entropy / max_entropy

    return f0_hz, confidence


# ============================================================================
# Weight Loading
# ============================================================================


def _find_torchcrepe_weights(capacity: str = "full") -> pathlib.Path | None:
    """Find torchcrepe .pth weight files from the installed package."""
    try:
        import torchcrepe

        pth = pathlib.Path(torchcrepe.__path__[0]) / "assets" / f"{capacity}.pth"
        return pth if pth.exists() else None
    except ImportError:
        return None


def _load_from_pth(model: CrepeModel, pth_path: pathlib.Path) -> None:
    """Load weights from a torchcrepe .pth file into a CrepeModel.

    PyTorch conv weights are (out, in, kernel, 1) — channels-first with trailing dim.
    Flax NNX conv weights are (kernel, in, out) — channels-last.
    Transpose: pth[out, in, k, 1] → squeeze → [out, in, k] → transpose → [k, in, out]
    """
    import torch

    state = torch.load(str(pth_path), map_location="cpu", weights_only=True)

    for i in range(6):
        layer_idx = i + 1
        # Conv: PyTorch (out, in, k, 1) → Flax (k, in, out)
        w = state[f"conv{layer_idx}.weight"].numpy()  # (out, in, k, 1)
        w = w.squeeze(-1)  # (out, in, k)
        w = w.transpose(2, 1, 0)  # (k, in, out)
        model.conv_layers[i].kernel[...] = jnp.array(w)
        model.conv_layers[i].bias[...] = jnp.array(state[f"conv{layer_idx}.bias"].numpy())

        # BatchNorm
        model.batch_norms[i].scale[...] = jnp.array(state[f"conv{layer_idx}_BN.weight"].numpy())
        model.batch_norms[i].bias[...] = jnp.array(state[f"conv{layer_idx}_BN.bias"].numpy())
        model.batch_norms[i].mean[...] = jnp.array(
            state[f"conv{layer_idx}_BN.running_mean"].numpy()
        )
        model.batch_norms[i].var[...] = jnp.array(state[f"conv{layer_idx}_BN.running_var"].numpy())

    # Classifier: PyTorch (360, 2048) → Flax (2048, 360)
    model.classifier.kernel[...] = jnp.array(state["classifier.weight"].numpy().T)
    model.classifier.bias[...] = jnp.array(state["classifier.bias"].numpy())


def _load_from_h5(model: CrepeModel, h5_path: pathlib.Path) -> None:
    """Load weights from a Keras .h5 file into a CrepeModel.

    Keras and Flax both use channels-last: conv kernel (k, in, out) — no transpose.
    """
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        for i in range(6):
            conv_name = f"conv1d{'_' + str(i) if i > 0 else ''}"
            bn_name = f"batch_normalization{'_' + str(i) if i > 0 else ''}"

            conv_group = f[conv_name][conv_name]
            model.conv_layers[i].kernel[...] = jnp.array(conv_group["kernel:0"][:])
            model.conv_layers[i].bias[...] = jnp.array(conv_group["bias:0"][:])

            bn_group = f[bn_name][bn_name]
            model.batch_norms[i].scale[...] = jnp.array(bn_group["gamma:0"][:])
            model.batch_norms[i].bias[...] = jnp.array(bn_group["beta:0"][:])
            model.batch_norms[i].mean[...] = jnp.array(bn_group["moving_mean:0"][:])
            model.batch_norms[i].var[...] = jnp.array(bn_group["moving_variance:0"][:])

        dense_group = f["dense"]["dense"]
        model.classifier.kernel[...] = jnp.array(dense_group["kernel:0"][:])
        model.classifier.bias[...] = jnp.array(dense_group["bias:0"][:])


def load_crepe_weights(
    model: CrepeModel,
    weights_path: str | pathlib.Path | None = None,
    capacity: str = "full",
) -> None:
    """Load pretrained CREPE weights into a CrepeModel.

    Supports three weight sources (tried in order):
    1. Explicit path (.h5 or .pth file)
    2. torchcrepe package .pth files (if installed)
    3. Cached .h5 download (future: when Keras weights URL is available)

    Args:
        model: CrepeModel instance to load weights into.
        weights_path: Explicit path to .h5 or .pth file. If None, auto-detects.
        capacity: Model capacity (must be "full" for pretrained).
    """
    if weights_path is not None:
        weights_path = pathlib.Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"CREPE weights not found at {weights_path}")
        if weights_path.suffix == ".pth":
            _load_from_pth(model, weights_path)
        else:
            _load_from_h5(model, weights_path)
        model.eval()
        return

    # Auto-detect: try torchcrepe first
    pth_path = _find_torchcrepe_weights(capacity)
    if pth_path is not None:
        _load_from_pth(model, pth_path)
        model.eval()
        return

    # Try cached .h5
    h5_cache = _CREPE_CACHE_DIR / f"model-{capacity}.h5"
    if h5_cache.exists():
        _load_from_h5(model, h5_cache)
        model.eval()
        return

    raise FileNotFoundError(
        f"CREPE weights not found. Install torchcrepe (`pip install torchcrepe`) "
        f"or place a .h5/.pth file at {_CREPE_CACHE_DIR}/model-{capacity}.*"
    )
