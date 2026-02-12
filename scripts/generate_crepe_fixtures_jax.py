"""Generate reference test fixtures using our Flax NNX CREPE model.

Uses torchcrepe pretrained weights loaded into our model.
Output: tests/fixtures/crepe/reference_outputs.npz
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import jax.numpy as jnp
from flax import nnx

from datarax.operators.modality.audio.crepe_model import (
    CrepeModel,
    decode_pitch_local,
    load_crepe_weights,
)


def main():
    model = CrepeModel(capacity="full", rngs=nnx.Rngs(0))
    load_crepe_weights(model)
    model.eval()

    sr = 16000
    t = np.linspace(0, 1024 / sr, 1024, endpoint=False)

    signals = {
        "440hz": np.sin(2 * np.pi * 440.0 * t).astype(np.float32),
        "880hz": np.sin(2 * np.pi * 880.0 * t).astype(np.float32),
        "silence": np.zeros(1024, dtype=np.float32),
        "noise": np.random.RandomState(42).randn(1024).astype(np.float32) * 0.1,
    }

    for name, sig in signals.items():
        if name != "silence":
            signals[name] = (sig - sig.mean()) / max(sig.std(), 1e-8)

    names = list(signals.keys())
    inputs = np.stack([signals[n] for n in names])

    x = jnp.array(inputs)[:, :, None]
    probs = np.array(model(x))

    f0s, confs = [], []
    for i, name in enumerate(names):
        f0, conf = decode_pitch_local(jnp.array(probs[i]))
        f0s.append(float(f0))
        confs.append(float(conf))
        print(f"  {name}: f0={float(f0):.1f} Hz, confidence={float(conf):.3f}")

    out_dir = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "crepe"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_dir / "reference_outputs.npz"),
        inputs=inputs,
        ref_probabilities=probs,
        ref_f0_hz=np.array(f0s),
        ref_confidence=np.array(confs),
        signal_names=np.array(names),
    )
    print(f"Saved reference fixtures to {out_dir}/reference_outputs.npz")


if __name__ == "__main__":
    main()
