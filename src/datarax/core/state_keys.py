"""Names of the per-record ``state`` entries datarax itself writes.

Every writer and reader of one of these entries uses the constant, so a name is spelled once.
"""

WEIGHT = "weight"
"""A record's float weight in losses and metrics; 0 for a padding row."""

__all__ = ["WEIGHT"]
