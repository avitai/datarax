"""Positive controls for jax's transfer guards: whether a guard fires on this backend.

A test asserting that no transfer happens inside a guard first shows the guard fires on a
transfer it must see; on a backend where it does not (device memory is host memory on CPU), the
assertion would pass without measuring anything.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def device_to_host_raises() -> bool:
    """The control: reading a device array back is a device-to-host transfer."""
    value = jax.block_until_ready(jnp.ones(2) * 3)
    with jax.transfer_guard_device_to_host("disallow"):
        try:
            np.asarray(value)
        except RuntimeError:
            return True
    return False


def implicit_upload_raises() -> bool:
    """The control: a NumPy argument to a jitted function is an implicit host-to-device transfer."""
    double = jax.jit(lambda x: x * 2)
    jax.block_until_ready(double(jnp.ones(2)))
    with jax.transfer_guard_host_to_device("disallow"):
        try:
            double(np.ones(2))
        except RuntimeError:
            return True
    return False
