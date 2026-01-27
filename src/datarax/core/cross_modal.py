"""Cross-modal operator base classes.

This module provides base classes for operators that work across multiple modalities
(reading from multiple fields and producing combined outputs). CrossModalOperator
enables fusion, cross-attention, contrastive learning, and alignment operations.

Key Features:

- Multi-field input/output transformations
- Learnable fusion and attention mechanisms
- Compatible with JAX transformations (jit, vmap, grad)
- End-to-end differentiable cross-modal pipelines
- Support for contrastive learning and alignment

Examples:
    Deterministic fusion operator:

    ```python
    config = CrossModalOperatorConfig(
        input_fields=["image_embedding", "text_embedding"],
        output_fields=["fused_embedding"],
        operation="fusion"
    )
    operator = FusionOperator(config, rngs=nnx.Rngs(0))
    ```

    Stochastic contrastive operator:

    ```python
    config = CrossModalOperatorConfig(
        input_fields=["anchor_emb", "positive_emb"],
        output_fields=["similarity"],
        operation="contrastive",
        stochastic=True,
        stream_name="contrastive"
    )
    operator = ContrastiveOperator(config, rngs=nnx.Rngs(0, contrastive=1))
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
class CrossModalOperatorConfig(OperatorConfig):
    """Configuration for cross-modal operators.

    Cross-modal operators read from MULTIPLE fields and may produce
    new combined fields. They can have learnable parameters and must
    be compatible with JAX transformations.

    Use Cases:

        - **Fusion**: Combine embeddings from different modalities
          Examples: [image_emb, text_emb] → joint_emb
        - **Cross-attention**: Attend from one modality to another
          Examples: [image_features, text_features] → attended_features
        - **Contrastive**: Compute similarity across modalities
          Examples: [image_emb, text_emb] → similarity_score
        - **Alignment**: Enforce cross-modal consistency
          Examples: [image_features, text_features] → alignment_loss

    Attributes:
        input_fields: List of input field names to read from
                     Examples: ["image_embedding", "text_embedding"]
        output_fields: List of output field names to write to
                      Examples: ["fused_embedding", "similarity_score"]
        operation: Cross-modal operation type
                  Common values: "fusion", "cross_attention", "contrastive", "alignment"
        validate_alignment: Whether to validate input field alignment
                          (e.g., check batch dimensions match)

    Validation Rules:

        - input_fields must be non-empty list of non-empty strings
        - output_fields must be non-empty list of non-empty strings
        - Inherits stochastic validation from OperatorConfig
        - Inherits statistics validation from DataraxModuleConfig

    Examples:
        Simple fusion:

        ```python
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb"],
            output_fields=["fused_emb"]
        )
        ```

        Cross-attention with multiple outputs:

        ```python
        config = CrossModalOperatorConfig(
            input_fields=["query", "key", "value"],
            output_fields=["attended_output", "attention_weights"],
            operation="cross_attention"
        )
        ```

        Stochastic contrastive learning:

        ```python
        config = CrossModalOperatorConfig(
            input_fields=["anchor", "positive", "negative"],
            output_fields=["similarity"],
            operation="contrastive",
            stochastic=True,
            stream_name="contrastive"
        )
        ```
    """

    # Input fields (multiple modalities)
    # Use kw_only to allow required field after parent's optional fields
    input_fields: list[str] = field(kw_only=True)

    # Output fields (may produce multiple outputs)
    output_fields: list[str] = field(kw_only=True)

    # Cross-modal operation type
    operation: str = field(default="fusion", kw_only=True)

    # Whether to validate input field alignment
    validate_alignment: bool = field(default=True, kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters.

        Validates:
        1. Parent configuration (OperatorConfig stochastic/statistics rules)
        2. input_fields is non-empty list of non-empty strings
        3. output_fields is non-empty list of non-empty strings

        Raises:
            ValueError: If configuration is invalid
        """
        # Call parent validation first (statistics, stochastic rules)
        super().__post_init__()

        # Validate input_fields is non-empty
        if not self.input_fields:
            raise ValueError("input_fields must be a non-empty list")

        # Validate all input fields are non-empty strings
        for field_name in self.input_fields:
            if not isinstance(field_name, str) or not field_name:
                raise ValueError(f"All input_fields must be non-empty strings, got {field_name}")

        # Validate output_fields is non-empty
        if not self.output_fields:
            raise ValueError("output_fields must be a non-empty list")

        # Validate all output fields are non-empty strings
        for field_name in self.output_fields:
            if not isinstance(field_name, str) or not field_name:
                raise ValueError(f"All output_fields must be non-empty strings, got {field_name}")


class CrossModalOperator(OperatorModule):
    """Base class for cross-modal operators with learnable parameters.

    Operates across multiple fields within an Element, enabling:

        - Multi-modal fusion
        - Cross-modal attention
        - Contrastive learning
        - Cross-modal alignment

    Key Features:

        - Compatible with nnx.jit, jax.vmap, jax.grad
        - Supports learnable parameters via nnx.Param
        - End-to-end differentiable
        - Can be optimized jointly with model
        - Operates on Batch[Element] (inherited from OperatorModule)

    Inherited Features from OperatorModule:

        - **apply_batch()**: Automatically handles batched operations by calling apply()
      on each element. Override only if you need custom batch-level logic (e.g.,
      batch-level contrastive loss, cross-element attention). Default is sufficient
      for most element-wise cross-modal operations.

        - **Statistics system**: Optionally collect and use batch statistics via stats
          parameter in apply(). Useful for adaptive cross-modal operations (e.g.,
          batch-aware normalization of fused embeddings).

        - **Caching system**: Results can be cached based on operator configuration
      and input characteristics. Inherited from base OperatorModule, helps avoid
      redundant computation for deterministic cross-modal operations.

    Subclass Implementation Pattern:
        ```python
        class FusionOperator(CrossModalOperator):
            def __init__(self, config: CrossModalOperatorConfig, *, rngs: nnx.Rngs | None = None):
                super().__init__(config, rngs=rngs)
                # Add learnable fusion parameters
                self.fusion_weights = nnx.Param(jnp.ones(len(config.input_fields)))

            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Extract inputs
                inputs = self._extract_inputs(data)

                # Fuse with learnable weights
                fused = sum(w * emb for w, emb in zip(self.fusion_weights.get_value(), inputs))

                # Store outputs
                outputs = [fused]
                result = self._store_outputs(data, outputs)

                return result, state, metadata

            def generate_random_params(self, rng, data_shapes):
                # For stochastic operators only
                batch_size = data_shapes[self.config.input_fields[0]][0]
                return jax.random.normal(rng, (batch_size,))
        ```

    Subclasses provide specific cross-modal operations:

        - **FusionOperator**: Learned combination of embeddings
        - **CrossAttentionOperator**: Learnable query/key/value projections
        - **ContrastiveOperator**: Learnable projection heads and temperature

    Examples:
        Deterministic fusion:

        ```python
        fusion_op = FusionOperator(config, rngs=nnx.Rngs(0))
        ```

        Learnable cross-attention:

        ```python
        class LearnedCrossAttention(CrossModalOperator):
            def __init__(self, config, *, rngs, dim, num_heads):
                super().__init__(config, rngs=rngs)
                self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
                self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
                self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
                self.num_heads = num_heads
        ```

        Batch-level contrastive operator:

        ```python
        class BatchContrastiveOperator(CrossModalOperator):
            def apply_batch(self, batch, stats=None):
                # Override for batch-level contrastive loss
                # Compute pairwise similarities across entire batch
                # Call apply() for final per-element outputs
                pass
        ```
    """

    def __init__(
        self,
        config: CrossModalOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize CrossModalOperator.

        Args:
            config: Cross-modal operator configuration (already validated)
            rngs: Random number generators (required if stochastic=True)
            name: Optional operator name

        Raises:
            ValueError: If stochastic=True but rngs is None
        """
        super().__init__(config, rngs=rngs, name=name)
        self.config: CrossModalOperatorConfig = config
        # Subclasses add learnable parameters here

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply cross-modal operation to element.

        MUST be implemented by subclasses to provide cross-modal behavior.

        This is a PURE FUNCTION that transforms a single data element.
        It should not access self.rngs or generate random numbers.
        All randomness comes through random_params argument.

        Args:
            data: Element data PyTree (contains fields specified by config.input_fields)
                 Typically dict[str, Array] with no batch dimension
            state: Element state PyTree (typically dict[str, Any])
            metadata: Element metadata dict
            random_params: Random parameters for this element (from generate_random_params)
            stats: Optional batch statistics (from get_statistics() or passed explicitly)

        Returns:
            Tuple of (transformed_data, new_state, new_metadata)
            - transformed_data: PyTree with original fields + new output fields
            - new_state: Updated state PyTree
            - new_metadata: Updated metadata dict

        Implementation Pattern:
            ```python
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # 1. Extract input fields
                inputs = self._extract_inputs(data)

                # 2. Perform cross-modal operation
                outputs = self._cross_modal_transform(inputs, random_params, stats)

                # 3. Store outputs in data
                result = self._store_outputs(data, outputs)

                return result, state, metadata
            ```

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement apply()")

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> PyTree:
        """Generate random parameters for stochastic cross-modal operations.

        MUST be implemented by stochastic operators (config.stochastic=True).
        Deterministic operators can use default implementation (returns None).

        Generates PyTree of random parameters for cross-modal operations. For example,
        contrastive learning might generate per-element noise for augmentation.

        This method is impure (uses RNG) and called once per batch. The generated
        parameters are then passed to apply() for each element via vmap.

        Args:
            rng: JAX random key for this batch
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image_emb": (batch_size, dim), "text_emb": (batch_size, dim)}

        Returns:
            PyTree of random parameters for this batch.
            Structure depends on operator needs.
            For deterministic operators, returns None.

        Examples:
            ```python
            # Stochastic contrastive operator with noise augmentation
            def generate_random_params(self, rng, data_shapes):
                batch_size = data_shapes[self.config.input_fields[0]][0]
                # Generate per-element noise scales
                return jax.random.uniform(rng, (batch_size,), minval=0.0, maxval=0.1)
            ```

        Raises:
            NotImplementedError: If stochastic=True but not implemented
        """
        # Default implementation for deterministic operators
        return super().generate_random_params(rng, data_shapes)

    def _extract_inputs(self, data: dict) -> list[Any]:
        """Extract all input fields from data.

        Helper method for subclasses to safely extract multiple input fields.

        Args:
            data: Data dictionary containing fields

        Returns:
            List of input field values in order specified by config.input_fields

        Raises:
            KeyError: If any input field not found in data

        Examples:
            ```python
            inputs = self._extract_inputs(data)
            image_emb, text_emb = inputs[0], inputs[1]  # Two inputs
            image_emb, text_emb, audio_emb = self._extract_inputs(data)  # Three inputs
            ```
        """
        return [data[field] for field in self.config.input_fields]

    def _store_outputs(self, data: dict, outputs: list[Any]) -> dict:
        """Store output values in target fields.

        Helper method to store transformation results in output fields.
        Preserves all original data fields and adds new output fields.

        Args:
            data: Original data dictionary
            outputs: List of output values in order specified by config.output_fields

        Returns:
            New data dictionary with original fields preserved and output fields added

        Examples:
            ```python
            fused_emb = self._fuse(inputs)
            result = self._store_outputs(data, [fused_emb])  # Adds "fused_emb" field
            fused, similarity, alignment = self._process(inputs)
            result = self._store_outputs(data, [fused, similarity, alignment])
            ```
        """
        result = dict(data)
        for field_name, value in zip(self.config.output_fields, outputs):
            result[field_name] = value
        return result
