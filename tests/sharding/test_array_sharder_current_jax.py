"""Current JAX sharding API contracts for ArraySharder."""

import jax
import jax.numpy as jnp
import numpy as np

from datarax.sharding.array_sharder import ArraySharder


def test_existing_jax_array_uses_reshard(monkeypatch) -> None:
    array = jnp.ones((2, 2))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    sentinel = object()
    calls = []

    def fake_reshard(value, out_sharding):
        calls.append((value, out_sharding))
        return sentinel

    monkeypatch.setattr(jax, "reshard", fake_reshard)

    result = ArraySharder._shard_array_static(array, sharding)

    assert result is sentinel
    assert calls == [(array, sharding)]


def test_host_array_uses_device_put(monkeypatch) -> None:
    array = np.ones((2, 2))
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    sentinel = object()
    calls = []

    def fake_device_put(value, out_sharding):
        calls.append((value, out_sharding))
        return sentinel

    monkeypatch.setattr(jax, "device_put", fake_device_put)

    result = ArraySharder._shard_array_static(array, sharding)

    assert result is sentinel
    assert calls[0][1] == sharding
