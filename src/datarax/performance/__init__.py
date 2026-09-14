"""Performance modules for Datarax: goodput tracking and host/device synchronization.

JAX process settings (XLA flags, the compilation cache, accelerator memory) live in
``substrax.runtime``. Roofline analysis and compilation profiling live in calibrax
(``calibrax.profiling``).
"""

from datarax.performance.goodput import GoodputMetrics, GoodputTracker
from datarax.performance.synchronization import (
    block_until_ready_tree,
    copy_to_host_async_tree,
)


__all__ = [
    "GoodputTracker",
    "GoodputMetrics",
    "block_until_ready_tree",
    "copy_to_host_async_tree",
]
