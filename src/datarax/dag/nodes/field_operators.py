from __future__ import annotations
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Any

from datarax.dag.nodes.base import Node
from datarax.typing import Batch


class SplitFields(Node):
    """Split dictionary input and apply different transforms to different fields.

    Useful for multimodal data processing where different fields
    need different preprocessing.

    Examples:
        Split dictionary fields:

        ```python
        from datarax.dag.nodes import SplitFields, Identity
        split = SplitFields({
            'image': Identity(name="img_op"),
            'text': Identity(name="txt_op"),
            'label': Identity(name="lbl_op")
        })
        ```
    """

    def __init__(self, field_transforms: dict[str, Node]):
        """Initialize field splitter.

        Args:
            field_transforms: Dictionary mapping field names to transforms
        """
        super().__init__()
        # Use nnx.Dict for module containers (required since Flax 0.12.0)
        self.field_transforms = nnx.Dict(field_transforms)

    def __call__(self, batch: Batch, *, key: jax.Array | None = None) -> Batch:
        """Apply transforms to respective fields.

        Args:
            batch: Batch with dict data
            key: Optional RNG key (split across fields)

        Returns:
            Batch with transformed data fields
        """
        data = batch.data.get_value()
        states = batch.states.get_value()

        if not isinstance(data, dict):
            raise TypeError(f"SplitFields expects Batch with dict data, got {type(data)}")

        # Split RNG key if provided
        if key is not None:
            keys = jax.random.split(key, len(self.field_transforms))
            key_dict = dict(zip(self.field_transforms.keys(), keys))
        else:
            key_dict = {k: None for k in self.field_transforms.keys()}

        result_data = {}
        result_states = {}
        for field, transform in self.field_transforms.items():
            if field in data:
                # Create a mini-Batch for this field
                field_state = states.get(field, jnp.zeros((data[field].shape[0],)))
                mini_batch = Batch.from_parts(
                    data={field: data[field]},
                    states={field: field_state},
                    metadata_list=batch._metadata_list,
                    validate=False,
                )
                # Apply transform (expects Batch, returns Batch)
                transformed = transform(mini_batch, key=key_dict[field])
                # Extract result
                result_data[field] = transformed.data.get_value()[field]
                result_states[field] = transformed.states.get_value()[field]

        # Include any fields not mentioned in field_transforms
        for field in data:
            if field not in result_data:
                result_data[field] = data[field]
                result_states[field] = states.get(field, jnp.zeros((data[field].shape[0],)))

        return Batch.from_parts(
            data=result_data,
            states=result_states,
            metadata_list=batch._metadata_list,
            batch_metadata=batch._batch_metadata,
            batch_state=batch.batch_state.get_value(),
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        fields = list(self.field_transforms.keys())
        return f"SplitFields(fields={fields})"


class SplitField(Node):
    """Split specific fields for parallel processing.

    Useful for separating features and labels for different
    processing paths.
    """

    def __init__(self, fields: list[str], name: str | None = None):
        """Initialize field splitter.

        Args:
            fields: List of field names to extract
            name: Optional name
        """
        super().__init__(name=name or "SplitField")
        self.fields = fields

    def __call__(self, data: dict[str, Any], *, key: jax.Array | None = None) -> dict[str, Any]:
        """Extract specified fields.

        Args:
            data: Dictionary batch
            key: Optional RNG key (unused)

        Returns:
            Dictionary with only specified fields
        """
        if not isinstance(data, dict):
            raise ValueError("SplitField requires dictionary input")

        result = {}
        for field in self.fields:
            if field in data:
                result[field] = data[field]

        return result if result else data
