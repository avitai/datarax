"""Performance modules for Datarax: goodput tracking.

JAX process settings (XLA flags, the compilation cache, accelerator memory) live in
``substrax.runtime``. Roofline analysis and compilation profiling live in calibrax
(``calibrax.profiling``).
"""

from datarax.performance.goodput import GoodputMetrics, GoodputTracker


__all__ = [
    "GoodputTracker",
    "GoodputMetrics",
]
