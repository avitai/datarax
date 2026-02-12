"""NoiseOperator - Operator for image noise augmentation.

This operator extends ModalityOperator to provide three types of noise:

- Gaussian: Additive Gaussian noise
- Salt & Pepper: Impulse noise (random pixels to min/max)
- Poisson: Shot noise (photon noise simulation)

Key Features:

- Three noise types via 'mode' parameter
- Stochastic mode with pre-generated noise
- Deterministic mode for reproducible noise patterns
- Full JAX compatibility with JIT compilation

Examples:
    Basic usage:

    ```python
    config = NoiseOperatorConfig(
        field_key="image",
        mode="gaussian",
        noise_std=0.05,
        noise_mean=0.0
    )
    op = NoiseOperator(config, rngs=rngs)
    ```
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.modality import ModalityOperator, ModalityOperatorConfig
from datarax.operators.modality.image._validation import validate_field_key_shape


@dataclass
class NoiseOperatorConfig(ModalityOperatorConfig):
    """Configuration for NoiseOperator.

    Extends ModalityOperatorConfig with noise-specific parameters.

    Attributes:
        mode: Type of noise to apply:

            - "gaussian": Additive Gaussian noise
            - "salt_pepper": Impulse noise (random min/max pixels)
            - "poisson": Shot noise (photon counting noise)

            Default: "gaussian"

        # Gaussian mode parameters:
        noise_std: Standard deviation for Gaussian noise. Default: 0.05
        noise_mean: Mean for Gaussian noise. Default: 0.0

        # Salt & Pepper mode parameters:
        salt_prob: Probability of salt (max value) pixels. Default: 0.01
        pepper_prob: Probability of pepper (min value) pixels. Default: 0.01
        salt_value: Value for salt pixels (None=auto-detect). Default: None
        pepper_value: Value for pepper pixels (None=auto-detect). Default: None

        # Poisson mode parameters:
        lam_scale: Scale factor for Poisson lambda. Higher=more noise. Default: 1.0

        # Common parameters:
        clip_range: Range for clipping output values. Default: (0.0, 1.0)
                   Set to None for no clipping.

    Note:

        Different noise types use different parameters:

        - mode="gaussian": Uses noise_std and noise_mean
        - mode="salt_pepper": Uses salt_prob, pepper_prob, salt_value, pepper_value
        - mode="poisson": Uses lam_scale
    """

    mode: Literal["gaussian", "salt_pepper", "poisson"] = field(default="gaussian", kw_only=True)

    # Gaussian parameters
    noise_std: float = field(default=0.05, kw_only=True)
    noise_mean: float = field(default=0.0, kw_only=True)

    # Salt & Pepper parameters
    salt_prob: float = field(default=0.01, kw_only=True)
    pepper_prob: float = field(default=0.01, kw_only=True)
    salt_value: float | None = field(default=None, kw_only=True)
    pepper_value: float | None = field(default=None, kw_only=True)

    # Poisson parameters
    lam_scale: float = field(default=1.0, kw_only=True)

    # Override default clip_range to (0.0, 1.0)
    clip_range: tuple[float, float] | None = field(default=(0.0, 1.0), kw_only=True)

    def __post_init__(self):
        """Validate configuration parameters."""
        super().__post_init__()

        # Validate mode
        if self.mode not in ("gaussian", "salt_pepper", "poisson"):
            raise ValueError(
                f"mode must be 'gaussian', 'salt_pepper', or 'poisson', got '{self.mode}'"
            )

        # Validate Gaussian parameters
        if self.mode == "gaussian":
            if self.noise_std < 0:
                raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")

        # Validate Salt & Pepper parameters
        if self.mode == "salt_pepper":
            if not 0.0 <= self.salt_prob <= 1.0:
                raise ValueError(f"salt_prob must be in [0.0, 1.0], got {self.salt_prob}")
            if not 0.0 <= self.pepper_prob <= 1.0:
                raise ValueError(f"pepper_prob must be in [0.0, 1.0], got {self.pepper_prob}")
            if self.salt_prob + self.pepper_prob > 1.0:
                raise ValueError("Sum of salt_prob and pepper_prob cannot exceed 1.0")
            if self.salt_value is not None and not isinstance(self.salt_value, int | float):
                raise TypeError(f"salt_value must be a number or None, got {type(self.salt_value)}")
            if self.pepper_value is not None and not isinstance(self.pepper_value, int | float):
                raise TypeError(
                    f"pepper_value must be a number or None, got {type(self.pepper_value)}"
                )

        # Validate Poisson parameters
        if self.mode == "poisson":
            if self.lam_scale <= 0:
                raise ValueError(f"lam_scale must be positive, got {self.lam_scale}")


class NoiseOperator(ModalityOperator):
    """Image noise transformation operator.

    Applies noise to images using one of three modes:

    - Gaussian: output = input + N(mean, std²)
    - Salt & Pepper: Random pixels → salt_value or pepper_value
    - Poisson: output = Poisson(input * lam_scale) / lam_scale

    Supports three operation modes:

        1. **Deterministic**: Fixed noise pattern using fixed seed
        2. **Stochastic**: Per-sample random noise from generate_random_params()
        3. **External params**: Accept pre-generated random parameters

    The operator works on single elements (H, W, C images) and is composed into
    batch processing via apply_batch() from the base class.

    Examples:
        Gaussian noise - deterministic:

        ```python
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            noise_mean=0.0,
            stochastic=False
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))
        result, state, metadata = operator.apply(data, state, metadata)
        ```

        Salt & Pepper noise - stochastic:

        ```python
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.02,
            pepper_prob=0.02,
            stochastic=True
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))
        result, state, metadata = operator.apply_batch(batch_data, state, metadata)
        ```

    """

    def __init__(
        self,
        config: NoiseOperatorConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the noise operator.

        Args:
            config: Configuration for noise operation
            rngs: RNG streams for stochastic operations
        """
        super().__init__(config, rngs=rngs)
        # Type narrowing for better IDE support
        self.config: NoiseOperatorConfig = config

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: dict[str, tuple[int, ...]],
    ) -> dict[str, jax.Array]:
        """Generate random noise for stochastic mode.

        In stochastic mode, this pre-generates random noise for the entire batch.
        This approach avoids RNG state mutations inside vmapped apply().

        Args:
            rng: JAX random key
            data_shapes: Dictionary mapping field keys to their shapes.
                        Used to determine batch size and element shapes.

        Returns:
            Dictionary with mode-specific noise data:

                - Gaussian: {"noise": Array of shape (batch, H, W, C)}
                - Salt & Pepper: {"noise_mask": Array of shape (batch, H, W, C)}
                - Poisson: {"poisson_samples": Array of shape (batch, H, W, C)}

        Raises:
            KeyError: If field_key not in data_shapes
        """
        # Get full shape including batch dimension
        full_shape = validate_field_key_shape(data_shapes, self.config.field_key)

        if self.config.mode == "gaussian":
            # Generate Gaussian noise for entire batch
            noise = (
                jax.random.normal(rng, shape=full_shape) * self.config.noise_std
                + self.config.noise_mean
            )
            return {"noise": noise}

        elif self.config.mode == "salt_pepper":
            # Generate uniform random values for salt & pepper selection
            noise_mask = jax.random.uniform(rng, shape=full_shape)
            return {"noise_mask": noise_mask}

        elif self.config.mode == "poisson":
            # For Poisson, we need the image values, so we'll generate keys instead
            # Split RNG for each sample in batch
            batch_size = full_shape[0]
            rngs = jax.random.split(rng, batch_size)
            return {"poisson_rngs": rngs}

        else:
            raise ValueError(f"Unknown noise mode: {self.config.mode}")

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: dict[str, Any],
        random_params: dict[str, jax.Array] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], dict[str, Any]]:
        """Apply noise transformation to a single element.

        This operates on single elements (e.g., one image of shape [H, W, C]).
        For batch processing, use apply_batch() which handles random param generation.

        Args:
            data: Input data dictionary. Must contain field specified by config.field_key
            state: Operator state (unused for noise, passed through)
            metadata: Metadata dictionary (passed through unchanged)
            random_params: Optional random parameters from generate_random_params().
                          If config.stochastic=True and this is provided, uses
                          pre-generated noise/masks.
            stats: Optional statistics dictionary (unused)

        Returns:
            Tuple of (transformed_data, state, metadata)
                - transformed_data: Data dict with noise applied to target field
                - state: Unchanged state dict
                - metadata: Unchanged metadata dict

        Note:
            CRITICAL: Always check config.stochastic flag, not whether random_params is None.
            apply_batch() always passes random_params even in deterministic mode.
        """
        # Extract the field to transform using base class helper
        value = self._extract_field(data, self.config.field_key)

        # Apply mode-specific noise
        if self.config.mode == "gaussian":
            transformed = self._apply_gaussian_noise(value, random_params)
        elif self.config.mode == "salt_pepper":
            transformed = self._apply_salt_pepper_noise(value, random_params)
        elif self.config.mode == "poisson":
            transformed = self._apply_poisson_noise(value, random_params)
        else:
            raise ValueError(f"Unknown noise mode: {self.config.mode}")

        # Apply clipping if configured
        if self.config.clip_range is not None:
            transformed = self._apply_clip_range(transformed)

        # Remap the transformed value back into the data dictionary
        result = self._remap_field(data, transformed)

        return result, state, metadata

    def _apply_gaussian_noise(
        self,
        value: jax.Array,
        random_params: dict[str, jax.Array] | None,
    ) -> jax.Array:
        """Apply Gaussian noise to image."""
        # Short-circuit if std is zero
        if self.config.noise_std == 0.0:
            return value

        # Get or generate noise
        if self.config.stochastic and random_params is not None:
            # Use pre-generated noise
            noise = random_params.get("noise")
            if noise is None:
                raise ValueError(
                    "Stochastic mode requires 'noise' in random_params for Gaussian mode"
                )
        else:
            # Deterministic mode: generate noise with fixed seed
            rng_key = jax.random.key(0)
            noise = (
                jax.random.normal(rng_key, shape=value.shape) * self.config.noise_std
                + self.config.noise_mean
            )

        return value + noise

    def _apply_salt_pepper_noise(
        self,
        value: jax.Array,
        random_params: dict[str, jax.Array] | None,
    ) -> jax.Array:
        """Apply salt and pepper noise to image."""
        # Short-circuit if no salt or pepper
        if self.config.salt_prob == 0.0 and self.config.pepper_prob == 0.0:
            return value

        # Determine salt and pepper values (auto-detect if None)
        if self.config.salt_value is None:
            # Auto-detect based on image max value
            salt_val = jax.lax.cond(
                jnp.max(value) > 1.5,
                lambda: 255.0,  # [0, 255] range
                lambda: 1.0,  # [0, 1] range
            )
        else:
            salt_val = self.config.salt_value

        if self.config.pepper_value is None:
            pepper_val = 0.0
        else:
            pepper_val = self.config.pepper_value

        # Get or generate random mask
        if self.config.stochastic and random_params is not None:
            # Use pre-generated mask
            random_vals = random_params.get("noise_mask")
            if random_vals is None:
                raise ValueError(
                    "Stochastic mode requires 'noise_mask' in random_params for salt_pepper mode"
                )
        else:
            # Deterministic mode: generate mask with fixed seed
            rng_key = jax.random.key(0)
            random_vals = jax.random.uniform(rng_key, shape=value.shape)

        # Apply salt and pepper
        noisy_image = jnp.where(
            random_vals < self.config.salt_prob,
            salt_val,
            jnp.where(
                random_vals < self.config.salt_prob + self.config.pepper_prob, pepper_val, value
            ),
        )

        return noisy_image

    def _apply_poisson_noise(
        self,
        value: jax.Array,
        random_params: dict[str, jax.Array] | None,
    ) -> jax.Array:
        """Apply Poisson noise to image."""
        # Ensure image is non-negative for Poisson
        value = jnp.maximum(value, 0)

        # Get RNG key
        if self.config.stochastic and random_params is not None:
            poisson_rng = random_params.get("poisson_rngs")
            if poisson_rng is None:
                raise ValueError(
                    "Stochastic mode requires 'poisson_rngs' in random_params for poisson mode"
                )
        else:
            # Deterministic mode
            poisson_rng = jax.random.key(0)

        # Apply Poisson noise based on image range
        noisy_image = jax.lax.cond(
            jnp.max(value) > 1.5,
            lambda: self._poisson_255_range(value, poisson_rng),
            lambda: self._poisson_01_range(value, poisson_rng),
        )

        return noisy_image

    def _poisson_255_range(self, image: jax.Array, rng: jax.Array) -> jax.Array:
        """Apply Poisson noise to [0, 255] range image."""
        lam = image * self.config.lam_scale / 255.0
        noisy_image = (
            jax.random.poisson(rng, lam=lam, shape=image.shape) * 255.0 / self.config.lam_scale
        )
        return noisy_image

    def _poisson_01_range(self, image: jax.Array, rng: jax.Array) -> jax.Array:
        """Apply Poisson noise to [0, 1] range image."""
        lam = image * self.config.lam_scale
        noisy_image = jax.random.poisson(rng, lam=lam, shape=image.shape) / self.config.lam_scale
        return noisy_image
