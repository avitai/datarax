"""Generate reference test fixtures from the original TF CREPE model.

One-time utility — run this to generate tests/fixtures/crepe/reference_outputs.npz.
Requires: pip install crepe (TensorFlow-based reference implementation).

Usage:
    python scripts/generate_crepe_fixtures.py

Output:
    tests/fixtures/crepe/reference_outputs.npz containing:
        - inputs: (N, 1024) audio frames
        - ref_probabilities: (N, 360) CREPE probability distributions
        - ref_f0_hz: (N,) decoded f0 values
        - ref_confidence: (N,) decoded confidence values
"""

import pathlib

import numpy as np

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "crepe"
SAMPLE_RATE = 16000


def generate_test_signals():
    """Generate test audio signals: 440 Hz, 880 Hz, silence, white noise."""
    t = np.linspace(0, 1024 / SAMPLE_RATE, 1024, endpoint=False)

    signals = {
        "440hz": np.sin(2 * np.pi * 440.0 * t).astype(np.float32),
        "880hz": np.sin(2 * np.pi * 880.0 * t).astype(np.float32),
        "silence": np.zeros(1024, dtype=np.float32),
        "noise": np.random.RandomState(42).randn(1024).astype(np.float32) * 0.1,
    }

    # Normalize each signal (CREPE convention)
    for name, sig in signals.items():
        if name != "silence":
            mean = np.mean(sig)
            std = max(np.std(sig), 1e-8)
            signals[name] = (sig - mean) / std

    return signals


def main():
    try:
        import crepe
    except ImportError:
        print("ERROR: 'crepe' package not installed.")
        print("Install with: pip install crepe")
        print("This is only needed for generating reference fixtures.")
        return

    import tensorflow as tf  # noqa: F401 — side-effect import; CREPE requires TF initialized

    signals = generate_test_signals()
    names = list(signals.keys())

    inputs = np.stack([signals[n] for n in names])  # (N, 1024)

    # Run through CREPE model
    # crepe.predict returns (time, frequency, confidence, activation)
    # For single-frame inputs, we can use the model directly
    model = crepe.core.build_and_load_model("full")

    all_probs = []
    all_f0 = []
    all_conf = []

    for name in names:
        frame = signals[name].reshape(1, 1024, 1)
        probs = model.predict(frame, verbose=0)  # (1, 360)
        all_probs.append(probs[0])

        # Decode using CREPE's own decoding
        cents_mapping = np.linspace(0, 7180, 360) + 1997.3794084376191
        center = np.argmax(probs[0])
        conf = np.max(probs[0])

        # Local weighted average (matching CREPE's decode)
        start = max(0, center - 4)
        end = min(360, center + 5)
        local_probs = probs[0][start:end]
        local_cents = cents_mapping[start:end]
        f0_cents = np.sum(local_probs * local_cents) / (np.sum(local_probs) + 1e-8)
        f0_hz = 10.0 * 2.0 ** (f0_cents / 1200.0)

        all_f0.append(f0_hz)
        all_conf.append(conf)

        print(f"  {name}: f0={f0_hz:.1f} Hz, confidence={conf:.3f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "reference_outputs.npz"

    np.savez(
        str(output_path),
        inputs=inputs,
        ref_probabilities=np.stack(all_probs),
        ref_f0_hz=np.array(all_f0),
        ref_confidence=np.array(all_conf),
        signal_names=np.array(names),
    )
    print(f"\nSaved reference fixtures to {output_path}")
    print(f"  inputs: {inputs.shape}")
    print(f"  ref_probabilities: {np.stack(all_probs).shape}")


if __name__ == "__main__":
    main()
