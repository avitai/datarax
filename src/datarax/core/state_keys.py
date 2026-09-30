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

MASKED = "masked"
"""bool per record, one entry per data field (``state[MASKED][field]``): the value is present and
an operator hides it from the model, keeping it in ``data`` as the target (masked pretraining).
The shape is the field's own mask: ``(B,)`` for the whole value, or finer (one per token)."""

IMPUTED = "imputed"
"""bool per record, one entry per data field (``state[IMPUTED][field]``): the value was missing
and an operator filled it, setting the field's ``Maybe.present``. A loss reads it to tell observed
values from filled ones."""

__all__ = ["IMPUTED", "MASKED", "MIX_LAMBDA", "MIX_PARTNER", "WEIGHT"]
