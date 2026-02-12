"""Audio modality operators for Datarax.

Provides reusable, differentiable audio feature extraction operators:

- LoudnessOperator: A-weighted loudness via STFT (pure JAX, learnable weights)
- CrepeF0Operator: Pitch (f0) extraction via CREPE CNN (Flax NNX port)

Both operators follow the standard OperatorModule contract (apply/apply_batch)
and are fully vmap/JIT/grad compatible.
"""

from datarax.operators.modality.audio.loudness_operator import (
    LoudnessConfig,
    LoudnessOperator,
)

__all__ = [
    "LoudnessConfig",
    "LoudnessOperator",
]


def __getattr__(name: str):
    """Lazy imports for CREPE-related classes (heavy weight loading)."""
    if name in ("CrepeF0Operator", "CrepeF0Config"):
        from datarax.operators.modality.audio.f0_operator import (
            CrepeF0Config,
            CrepeF0Operator,
        )

        return {"CrepeF0Operator": CrepeF0Operator, "CrepeF0Config": CrepeF0Config}[name]
    if name in ("CrepeModel", "load_crepe_weights"):
        from datarax.operators.modality.audio.crepe_model import (
            CrepeModel,
            load_crepe_weights,
        )

        return {"CrepeModel": CrepeModel, "load_crepe_weights": load_crepe_weights}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
