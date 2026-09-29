"""Integration tests for CompositeOperatorModule.

This module tests complex real-world scenarios and integration with other
Datarax components (MapOperator, Batch, etc.).

Test Coverage:
- Sequential of MapOperators
- Parallel of MapOperators
- Nested composites (sequential inside parallel, parallel inside sequential)
- Multi-level nesting (3+ levels)
- Mixed stochastic/deterministic pipelines
- Real use cases (augmentation pipeline, ensemble, branching)
- End-to-end with Batch processing
"""

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core import batch_ops
from datarax.core.element_batch import Element

# GREEN phase - imports enabled
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


class TestCompositeWithMapOperator:
    """Test integration with MapOperator."""

    def test_sequential_of_map_operators(self):
        """Test sequential composite containing MapOperator children."""
        rngs = nnx.Rngs(0)

        # Create 3 MapOperators
        op1 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs)
        op2 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 5, rngs=rngs)
        op3 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 3, rngs=rngs)

        # Compose sequentially
        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            config,
            operators=[op1, op2, op3],
        )

        # Test: ((x * 2) + 5) * 3
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([1.0])}),
                    Element(data={"value": jnp.array([2.0])}),
                ]
            )
        )
        result_batch = composite(batch)
        result_data = result_batch.data
        assert jnp.allclose(
            result_data["value"], jnp.array([[21.0], [27.0]])
        )  # ((1*2)+5)*3=21, ((2*2)+5)*3=27

    def test_parallel_of_map_operators(self):
        """Test parallel composite containing MapOperator children."""
        rngs = nnx.Rngs(0)

        # Create 3 MapOperators
        op1 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs)
        op2 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 3, rngs=rngs)
        op3 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 4, rngs=rngs)

        # Compose in parallel
        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            merge_strategy="concat",
        )
        composite = CompositeOperatorModule(
            config,
            operators=[op1, op2, op3],
        )

        # Test: concat([x*2, x*3, x*4])
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([2.0])})]))
        result_batch = composite(batch)
        result_data = result_batch.data
        assert jnp.allclose(result_data["value"], jnp.array([[4.0, 6.0, 8.0]]))


class TestNestedComposites:
    """Test nested composite operators."""

    def test_nested_sequential_inside_parallel(self):
        """Test sequential composite inside parallel composite."""
        rngs = nnx.Rngs(0)

        # Create two sequential branches
        seq1_ops = [
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs),
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 1, rngs=rngs),
        ]
        branch1 = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=seq1_ops,
        )

        seq2_ops = [
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 3, rngs=rngs),
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 10, rngs=rngs),
        ]
        branch2 = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=seq2_ops,
        )

        # Compose in parallel
        parallel_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            merge_strategy="concat",
        )
        composite = CompositeOperatorModule(
            parallel_config,
            operators=[branch1, branch2],
        )

        # Test: concat([(x*2)+1, (x*3)+10])
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([5.0])})]))
        result_batch = composite(batch)
        result_data = result_batch.data
        assert jnp.allclose(
            result_data["value"], jnp.array([[11.0, 25.0]])
        )  # (5*2)+1=11, (5*3)+10=25

    def test_nested_parallel_inside_sequential(self):
        """Test parallel composite inside sequential composite."""
        rngs = nnx.Rngs(0)

        # Create parallel stage
        par_ops = [
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs),
            MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 3, rngs=rngs),
        ]
        parallel = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.PARALLEL,
                merge_strategy="concat",
            ),
            operators=par_ops,
        )

        # Wrap in sequential with another operator
        final_op = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 100, rngs=rngs
        )

        seq_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            seq_config,
            operators=[parallel, final_op],
        )

        # Test: concat([x*2, x*3]) then +100 to all
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([1.0])})]))
        result_batch = composite(batch)
        result_data = result_batch.data
        assert jnp.allclose(
            result_data["value"], jnp.array([[102.0, 103.0]])
        )  # (1*2)+100=102, (1*3)+100=103

    def test_three_level_nesting(self):
        """Test 3-level deep nesting of composites."""
        rngs = nnx.Rngs(0)

        # Level 3: Sequential (innermost)
        inner_seq = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=[
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 1, rngs=rngs
                ),
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs
                ),
            ],
        )

        # Level 2: Parallel (middle)
        middle_par = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.PARALLEL,
                merge_strategy="concat",
            ),
            operators=[
                inner_seq,
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 10, rngs=rngs
                ),
            ],
        )

        # Level 1: Sequential (outer)
        outer_seq = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=[
                middle_par,
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 100, rngs=rngs
                ),
            ],
        )

        # Test: concat([(x+1)*2, x*10]) then +100
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([3.0])})]))
        result_batch = outer_seq(batch)
        result_data = result_batch.data
        # (3+1)*2=8, 3*10=30, then +100 to both: 108, 130
        assert jnp.allclose(result_data["value"], jnp.array([[108.0, 130.0]]))


class TestMixedModes:
    """Test mixed stochastic/deterministic pipelines."""

    def test_mixed_stochastic_deterministic_pipeline(self):
        """Test pipeline with both stochastic and deterministic operators."""
        rngs = nnx.Rngs(0, augment=1)

        # Stochastic operator (adds noise)
        stoch_config = MapOperatorConfig(stochastic=True, stream_name="augment")
        stoch_op = MapOperator(
            stoch_config, fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1, rngs=rngs
        )

        # Deterministic operator (scales)
        det_config = MapOperatorConfig(stochastic=False)
        det_op = MapOperator(det_config, fn=lambda x, _key: x * 2, rngs=rngs)

        # Build sequential pipeline
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            stochastic=True,
            stream_name="augment",
        )
        composite = CompositeOperatorModule(
            composite_config, operators=[stoch_op, det_op], rngs=rngs
        )

        # Test with batch
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([1.0])}),
                    Element(data={"value": jnp.array([2.0])}),
                ]
            )
        )
        result = composite(batch)
        result_data = result.data

        # Output should differ from deterministic (x * 2) due to added noise
        # But should be roughly 2x the input
        expected_approx = jnp.array([[2.0], [4.0]])
        assert result_data["value"].shape == (2, 1)
        # Close but not exact due to stochastic noise
        assert jnp.allclose(result_data["value"], expected_approx, atol=1.0)


class TestRealWorldUseCases:
    """Test real-world use cases."""

    def test_full_augmentation_pipeline(self):
        """Test realistic image augmentation pipeline."""
        rngs = nnx.Rngs(0)

        # Simulate augmentation pipeline: normalize -> crop -> brightness -> contrast
        normalize = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x / 255.0, rngs=rngs
        )
        crop = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 0.9, rngs=rngs
        )  # Simulate crop
        brightness = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 0.1, rngs=rngs
        )
        contrast = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 1.2, rngs=rngs
        )

        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        pipeline = CompositeOperatorModule(
            config,
            operators=[normalize, crop, brightness, contrast],
        )

        # Test augmentation pipeline
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([128.0])}),
                    Element(data={"value": jnp.array([255.0])}),
                ]
            )
        )
        result_batch = pipeline(batch)
        result_data = result_batch.data

        # (x/255)*0.9+0.1)*1.2: (128/255)*0.9+0.1)*1.2 ≈ 0.659, (255/255)*0.9+0.1)*1.2 = 1.2
        assert result_data["value"].shape == (2, 1)
        assert jnp.all(result_data["value"] >= 0.0)

    def test_ensemble_of_models(self):
        """Test ensemble of multiple models (realistic ML use case)."""
        rngs = nnx.Rngs(0)

        # Simulate 3 different model predictions
        model1 = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 1.1, rngs=rngs
        )
        model2 = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 0.9, rngs=rngs
        )
        model3 = MapOperator(
            MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 1.0, rngs=rngs
        )

        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
        )
        ensemble = CompositeOperatorModule(
            config,
            operators=[model1, model2, model3],
        )

        # Test ensemble prediction (mean of 3 models)
        batch = batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([10.0])})])
        )
        result_batch = ensemble(batch)
        result_data = result_batch.data

        # Mean of [10*1.1, 10*0.9, 10*1.0] = mean of [11, 9, 10] = 10.0
        assert jnp.allclose(result_data["value"], jnp.array([[10.0]]))

    def test_complex_branching_logic(self):
        """Test complex branching with different processing paths."""
        rngs = nnx.Rngs(0)

        # Small images: light augmentation
        small_path = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=[
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 1.1, rngs=rngs
                ),
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 5, rngs=rngs
                ),
            ],
        )

        # Large images: heavy augmentation
        large_path = CompositeOperatorModule(
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
            ),
            operators=[
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 0.8, rngs=rngs
                ),
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 20, rngs=rngs
                ),
            ],
        )

        # Router based on value magnitude (returns integer index, vmap compatible)
        def router(data):
            # 0 = small, 1 = large
            return jax.lax.select(data["value"][0] < 100.0, 0, 1)

        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            router=router,
        )
        branching = CompositeOperatorModule(
            config,
            operators=[small_path, large_path],  # List indexed 0=small, 1=large
        )

        # Test small path
        small_batch = batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([50.0])})])
        )
        small_result_batch = branching(small_batch)
        small_result_data = small_result_batch.data
        assert jnp.allclose(small_result_data["value"], jnp.array([[60.0]]))  # (50*1.1)+5=60

        # Test large path
        large_batch = batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([200.0])})])
        )
        large_result_batch = branching(large_batch)
        large_result_data = large_result_batch.data
        assert jnp.allclose(large_result_data["value"], jnp.array([[180.0]]))  # (200*0.8)+20=180


class TestBatchIntegration:
    """Test integration with Batch processing."""

    def test_end_to_end_with_batch_processing(self):
        """Test composite operator with batched data using inherited apply_batch."""
        rngs = nnx.Rngs(0)

        # Create simple sequential composite
        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            config,
            operators=[
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs
                ),
                MapOperator(
                    MapOperatorConfig(stochastic=False), fn=lambda x, _key: x + 10, rngs=rngs
                ),
            ],
        )

        # Create batch from list of Elements
        elements = [
            Element(data={"value": jnp.array([1.0])}),
            Element(data={"value": jnp.array([2.0])}),
            Element(data={"value": jnp.array([3.0])}),
        ]
        batch = batch_ops.from_stacked(batch_ops.stack(elements))

        # Apply composite using __call__ (which calls apply_batch with vmap)
        result_batch = composite(batch)

        # Verify batch structure and results: (x * 2) + 10
        assert result_batch.batch_size == 3
        # Extract results from batch
        elem0 = batch_ops.element(result_batch, 0)
        elem1 = batch_ops.element(result_batch, 1)
        elem2 = batch_ops.element(result_batch, 2)
        assert jnp.allclose(elem0.data["value"], jnp.array([12.0]))  # (1*2)+10
        assert jnp.allclose(elem1.data["value"], jnp.array([14.0]))  # (2*2)+10
        assert jnp.allclose(elem2.data["value"], jnp.array([16.0]))  # (3*2)+10
