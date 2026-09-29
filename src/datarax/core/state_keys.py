"""Names of the ``state`` and ``batch_state`` entries datarax itself writes.

Every writer and reader of one of these entries uses the constant, so a name is spelled once.
"""

WEIGHT = "weight"
"""A record's float weight in losses and metrics; 0 for a padding row."""

MIX_PARTNER = "mix_partner"
"""int32, per record: the row a mixing operator (MixUp, CutMix) mixed this record with."""

MIX_LAMBDA = "mix_lambda"
"""float, one per batch (``batch_state``): the fraction of each record a mixing operator kept.

A loss for mixed records reads both:
``lam * loss(y) + (1 - lam) * loss(y[partner])``.
"""

__all__ = ["MIX_LAMBDA", "MIX_PARTNER", "WEIGHT"]
