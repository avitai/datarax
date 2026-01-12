"""Metadata module with proper pytree registration for JAX compatibility.

This module provides metadata for tracking experiment state with proper
separation of static and dynamic fields to avoid JIT recompilation.
"""

from __future__ import annotations
from typing import Any
from dataclasses import dataclass
import dataclasses

import jax
import flax.nnx as nnx
from jax.tree_util import register_pytree_node


@dataclass
class RecordMetadata:
    """Metadata for a record in the data pipeline.

    This dataclass tracks metadata about data records as they
    flow through the pipeline. It's designed to be PyTree-compatible for
    JAX transformations.

    Attributes:
        index: Monotonically increasing index for checkpointing
        record_key: Reference to the actual record (file index, offset, etc.)
        rng_key: JAX random key for stateless random operations
        epoch: Current epoch number
        global_step: Global step across all epochs
        batch_idx: Optional batch index within the current epoch
        shard_id: Optional shard identifier for distributed processing
        source_info: Optional dictionary for source-specific metadata
    """

    index: int
    record_key: Any
    rng_key: jax.Array | None = None
    epoch: int = 0
    global_step: int = 0
    batch_idx: int | None = None
    shard_id: int | None = None
    source_info: dict[str, Any] | None = None

    def split_rng(self, num: int = 2) -> list[jax.Array | None]:
        """Split RNG key into multiple keys."""
        if self.rng_key is None:
            return [None] * num
        return list(jax.random.split(self.rng_key, num))


@dataclass
class Metadata:
    """Metadata for tracking experiment state.

    Uses custom pytree registration to exclude static fields from tracing.
    This prevents JIT recompilation when only static fields change.

    Dynamic fields (traced):
        - index, epoch, global_step, batch_idx, shard_id, rng_key: Numeric tracking fields
        - _encoded_key: Byte-encoded key for JAX compatibility

    Static fields (not traced):
        - source_info: Arbitrary metadata dictionary
    """

    # Dynamic fields - included in pytree
    index: int = 0
    epoch: int = 0
    global_step: int = 0
    batch_idx: int | None = None
    shard_id: int | None = None
    rng_key: jax.Array | None = None

    # Internal storage for JIT-compatible key
    _encoded_key: jax.Array | None = dataclasses.field(default=None)

    # Static fields - excluded from pytree
    source_info: dict[str, Any] | None = None

    # Init-only field for backward compatibility
    key: dataclasses.InitVar[str | None] = None

    def __post_init__(self, key: str | None):
        """Initialize encoded key from string if provided."""
        if self._encoded_key is None and key is not None:
            self._encoded_key = _encode_key(key)

    @property
    def record_key(self) -> str | None:
        """Get record key as string (decodes on demand)."""
        return _decode_key(self._encoded_key)

    def replace(self, **kwargs) -> Metadata:
        """Create a new Metadata instance with updated fields."""
        # Handle key/record_key specifically to ensure it updates _encoded_key
        key_arg = kwargs.pop("key", kwargs.pop("record_key", None))
        encoded_key_arg = kwargs.pop("_encoded_key", self._encoded_key)

        if key_arg is not None:
            encoded_key_arg = _encode_key(key_arg)

        current = {
            "index": self.index,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_idx": self.batch_idx,
            "shard_id": self.shard_id,
            "rng_key": self.rng_key,
            "_encoded_key": encoded_key_arg,
            "source_info": self.source_info,
        }
        current.update(kwargs)
        # pass explicitly to init vars (key=None because we set _encoded_key directly)
        return Metadata(**current, key=None)

    def split_rng(self, num: int = 2) -> list[jax.Array | None]:
        """Split RNG key into multiple keys."""
        if self.rng_key is None:
            return [None] * num
        keys = jax.random.split(self.rng_key, num)
        return list(keys)

    def next_rng(self) -> Metadata:
        """Get next RNG state."""
        if self.rng_key is None:
            return self
        _, next_key = jax.random.split(self.rng_key, 2)
        return self.replace(rng_key=next_key)

    def increment_step(self) -> Metadata:
        """Increment global step."""
        return self.replace(global_step=self.global_step + 1)

    def increment_epoch(self) -> Metadata:
        """Increment epoch and reset batch index."""
        return self.replace(epoch=self.epoch + 1, batch_idx=0)

    def increment_batch(self) -> Metadata:
        """Increment batch index."""
        new_idx = 0 if self.batch_idx is None else self.batch_idx + 1
        return self.replace(batch_idx=new_idx)

    def with_shard(self, shard_id: int) -> Metadata:
        """Set shard ID."""
        return self.replace(shard_id=shard_id)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "batch_idx": self.batch_idx,
            "shard_id": self.shard_id,
            "rng_key": self.rng_key,
            "record_key": self.record_key,
            "source_info": self.source_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metadata:
        """Create from dictionary."""
        # Map 'record_key' from dict to 'key' arg
        key = data.pop("record_key", data.pop("key", None))
        return cls(**data, key=key)

    def merge(self, other: Metadata | None) -> Metadata:
        """Merge with another metadata, other takes precedence for non-zero/non-None values."""
        if other is None:
            return self

        return self.replace(
            index=other.index if other.index != 0 else self.index,
            epoch=other.epoch if other.epoch != 0 else self.epoch,
            global_step=other.global_step if other.global_step != 0 else self.global_step,
            batch_idx=other.batch_idx if other.batch_idx is not None else self.batch_idx,
            shard_id=other.shard_id if other.shard_id is not None else self.shard_id,
            _encoded_key=other._encoded_key
            if other._encoded_key is not None
            else self._encoded_key,
            source_info=other.source_info or self.source_info,
            rng_key=other.rng_key if other.rng_key is not None else self.rng_key,
        )


# Key Encoding/Decoding Constants
MAX_KEY_LENGTH = 128  # Fixed length for JIT compatibility
NULL_BYTE = 0


def _encode_key(key: str | None) -> jax.Array | None:
    """Encode string key to fixed-length uint8 array."""
    if key is None:
        return None

    # Convert to bytes
    key_bytes = key.encode("utf-8")
    if len(key_bytes) > MAX_KEY_LENGTH:
        # Truncate
        key_bytes = key_bytes[:MAX_KEY_LENGTH]

    # Create padded array using jax.numpy
    import jax.numpy as jnp

    # Create zero array
    arr = jnp.zeros(MAX_KEY_LENGTH, dtype=jnp.uint8)
    # Update with key bytes (eagerly)
    byte_list = list(key_bytes)
    # Use index update
    arr = arr.at[: len(byte_list)].set(jnp.array(byte_list, dtype=jnp.uint8))
    return arr


def _decode_key(arr: jax.Array | None) -> str | None:
    """Decode fixed-length uint8 array to string."""
    if arr is None:
        return None

    try:
        # If traced, we cannot decode
        if isinstance(arr, jax.core.Tracer):
            return "<TRACED_KEY>"

        # Extract bytes until null (0)
        # Convert to numpy/list for byte processing
        if hasattr(arr, "tolist"):
            bytes_list = arr.tolist()
        else:
            bytes_list = arr

        # Find first null byte or take all
        valid_bytes = []
        for b in bytes_list:
            if b == 0:
                break
            valid_bytes.append(b)

        return bytes(valid_bytes).decode("utf-8")
    except Exception:
        return "<DECODE_ERROR>"


# Custom pytree registration
def _metadata_flatten(metadata):
    """Flatten Metadata for pytree, excluding static fields."""
    # Dynamic values become leaves
    dynamic = (
        metadata.index,
        metadata.epoch,
        metadata.global_step,
        metadata.batch_idx,
        metadata.shard_id,
        metadata.rng_key,
        metadata._encoded_key,  # Now dynamic!
    )
    # Static values go in treedef
    static = (metadata.source_info,)
    return dynamic, static


def _metadata_unflatten(static, dynamic):
    """Reconstruct Metadata from pytree."""
    (source_info,) = static
    index, epoch, global_step, batch_idx, shard_id, rng_key, encoded_key = dynamic
    return Metadata(
        index=index,
        epoch=epoch,
        global_step=global_step,
        batch_idx=batch_idx,
        shard_id=shard_id,
        rng_key=rng_key,
        _encoded_key=encoded_key,
        source_info=source_info,
        key=None,
    )


# Register as pytree
register_pytree_node(Metadata, _metadata_flatten, _metadata_unflatten)


# Helper functions


def create_metadata(
    index: int = 0,
    epoch: int = 0,
    global_step: int = 0,
    batch_idx: int | None = None,
    shard_id: int | None = None,
    record_key: str | None = None,
    source_info: dict[str, Any] | None = None,
    rng_key: jax.Array | None = None,
    seed: int | None = None,
) -> Metadata:
    """Create metadata with optional RNG initialization."""
    if rng_key is None and seed is not None:
        rng_key = jax.random.key(seed)

    return Metadata(
        index=index,
        epoch=epoch,
        global_step=global_step,
        batch_idx=batch_idx,
        shard_id=shard_id,
        rng_key=rng_key,
        key=record_key,
        source_info=source_info,
    )


def split_rng_tree(metadata: Metadata, num: int = 2) -> dict[str, jax.Array | None]:
    """Split RNG key into named dictionary."""
    if metadata.rng_key is None:
        return {f"key_{i}": None for i in range(num)}

    keys = jax.random.split(metadata.rng_key, num)
    result = {}

    # Special names for first two keys
    if num > 0:
        result["a"] = keys[0]
    if num > 1:
        result["b"] = keys[1]

    # Generic names for additional keys
    for i in range(2, num):
        result[f"key_{i}"] = keys[i]

    return result


def batch_metadata(metadata_list: list[Metadata]) -> Metadata:
    """Combine multiple metadata into batch metadata."""
    if not metadata_list:
        return Metadata()

    first = metadata_list[0]

    # Take max of numeric fields
    max_step = max(m.global_step for m in metadata_list)
    max_epoch = max(m.epoch for m in metadata_list)

    # Find first non-None optional fields
    batch_idx = next((m.batch_idx for m in metadata_list if m.batch_idx is not None), None)
    shard_id = next((m.shard_id for m in metadata_list if m.shard_id is not None), None)

    return Metadata(
        index=first.index,
        epoch=max_epoch,
        global_step=max_step,
        batch_idx=batch_idx,
        shard_id=shard_id,
        rng_key=first.rng_key,
        key=first.record_key,
        source_info=first.source_info,
    )


@jax.jit
def update_metadata_batch(metadata: Metadata, batch_size: int) -> Metadata:
    """Update metadata after processing a batch."""
    new_step = metadata.global_step + batch_size
    new_batch_idx = 1 if metadata.batch_idx is None else metadata.batch_idx + 1
    return metadata.replace(global_step=new_step, batch_idx=new_batch_idx)


class MetadataManager(nnx.Module):
    """Utility module for managing metadata state in data sources.

    This is a composable utility that data sources can optionally include
    to track and manage metadata like epochs, global steps, and indices.
    It's designed to be used via composition rather than inheritance.

    Examples:
        Example usage:

        ```python
        class MySource(DataSourceModule):
            def __init__(self, track_metadata=False, *, rngs: nnx.Rngs):
                super().__init__(rngs=rngs)
                if track_metadata:
                    self.metadata_manager = MetadataManager(rngs=rngs)

            def __next__(self):
                data = self.get_data()
                if hasattr(self, 'metadata_manager'):
                    metadata = self.metadata_manager.create_metadata(record_key=...)
                    return data, metadata
                return data
        ```
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs | None = None,
        initial_epoch: int = 0,
        initial_step: int = 0,
        track_batches: bool = False,
        shard_id: int | None = None,
    ):
        """Initialize the MetadataManager.

        Args:
            rngs: Optional Rngs for generating RNG keys in metadata
            initial_epoch: Starting epoch number
            initial_step: Starting global step count
            track_batches: Whether to track batch indices
            shard_id: Optional shard identifier for distributed processing
        """
        super().__init__()
        self.rngs = rngs
        self.shard_id = shard_id
        self.track_batches = track_batches

        # Mutable state variables
        self.state = nnx.Variable(
            {
                "global_step": initial_step,
                "epoch": initial_epoch,
                "index": 0,
                "batch_idx": 0 if track_batches else None,
            }
        )

    def create_metadata(
        self,
        record_key: Any,
        source_info: dict[str, Any] | None = None,
    ) -> RecordMetadata:
        """Create metadata for a record and update internal state.

        Args:
            record_key: Reference to the actual record
            source_info: Optional source-specific metadata

        Returns:
            RecordMetadata instance with current state
        """
        # Get RNG key if available
        rng_key: jax.Array | None = None
        if self.rngs is not None and "metadata" in self.rngs:
            rng_key = self.rngs.metadata()

        # Get current state
        current_state = self.state.get_value()

        # Extract state values with type assertions
        index = current_state["index"]
        epoch = current_state["epoch"]
        global_step = current_state["global_step"]
        batch_idx = current_state["batch_idx"]

        assert isinstance(index, int), "index must be int"
        assert isinstance(epoch, int), "epoch must be int"
        assert isinstance(global_step, int), "global_step must be int"

        # Create metadata with current state
        metadata = RecordMetadata(
            index=index,
            record_key=record_key,
            rng_key=rng_key,
            epoch=epoch,
            global_step=global_step,
            batch_idx=batch_idx,
            shard_id=self.shard_id,
            source_info=source_info,
        )

        # Update state
        current_state["index"] = index + 1
        current_state["global_step"] = global_step + 1
        self.state.set_value(current_state)

        return metadata

    def next_epoch(self) -> None:
        """Advance to the next epoch."""
        current_state = self.state.get_value()
        epoch = current_state["epoch"]
        assert isinstance(epoch, int), "epoch must be int"
        current_state["epoch"] = epoch + 1
        current_state["index"] = 0
        self.state.set_value(current_state)

    def next_batch(self) -> None:
        """Advance to the next batch."""
        if self.track_batches:
            current_state = self.state.get_value()
            batch_idx = current_state["batch_idx"]
            assert isinstance(batch_idx, int), "batch_idx must be int when track_batches is True"
            current_state["batch_idx"] = batch_idx + 1
            self.state.set_value(current_state)

    def reset(self) -> None:
        """Reset all counters to initial values."""
        current_state = self.state.get_value()
        current_state["global_step"] = 0
        current_state["epoch"] = 0
        current_state["index"] = 0
        if self.track_batches:
            current_state["batch_idx"] = 0
        self.state.set_value(current_state)
