"""Test script for HuggingFace datasets with Datarax.

This script tests various HuggingFace datasets with Datarax to ensure compatibility
with different dataset types, structures, and sizes.
"""

import logging
import time
from typing import Any

import jax
from flax import nnx
from tqdm import tqdm

from datarax.core import Pipeline
from datarax.sources import HFEagerConfig, HFEagerSource


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hf_datasets_test")


# Dataset configurations to test
DATASET_CONFIGS = [
    # Text datasets
    {"path": "imdb", "split": "train", "streaming": True},
    {"path": "squad", "split": "train", "streaming": True},
    # Image datasets
    {"path": "cifar10", "split": "train", "streaming": True},
    {"path": "mnist", "split": "train", "streaming": True},
    # Other datasets
    {"path": "ag_news", "split": "train", "streaming": True},
    {"path": "rotten_tomatoes", "split": "train", "streaming": True},
]


def dataset_compatibility_test(
    path: str,
    split: str,
    streaming: bool = True,
    name: str | None = None,
    data_dir: str | None = None,
    max_examples: int = 10,
    analyze_structure: bool = True,
) -> dict[str, Any]:
    """Test a specific HuggingFace dataset's compatibility with Datarax.

    Args:
        path: Path/name of the dataset
        split: Split to test
        streaming: Whether to use streaming mode
        name: Optional dataset configuration name (for datasets with configs)
        data_dir: Optional directory for cached datasets
        max_examples: Maximum number of examples to process
        analyze_structure: Whether to analyze dataset structure

    Returns:
        Dictionary with test results
    """
    start_time = time.time()
    logger.info(f"Testing dataset: {path} (name: {name}, split: {split})")

    try:
        # Create data source config
        config = HFEagerConfig(
            name=path,
            split=split,
            streaming=streaming,
            data_dir=data_dir,
        )
        source = HFEagerSource(config, rngs=nnx.Rngs(0))

        # Create data stream
        stream = Pipeline(source)

        # Check dataset size if possible
        try:
            dataset_size = len(source)
            size_info = f"Dataset size: {dataset_size}"
        except NotImplementedError:
            size_info = "Dataset size not available (streaming mode)"

        logger.info(size_info)

        # Process examples
        examples_processed = 0
        data_types = {}
        shapes = {}

        example_iter = iter(stream.batch(batch_size=1).iterator())
        for _ in tqdm(range(max_examples), desc="Processing examples"):
            element = next(example_iter)
            examples_processed += 1

            # Analyze structure if needed
            if analyze_structure and examples_processed == 1:
                # Log keys
                logger.info(f"Keys: {list(element.keys())}")

                # Analyze data types and shapes
                for key, value in element.items():
                    if hasattr(value, "dtype"):
                        data_types[key] = str(value.dtype)
                        shapes[key] = str(value.shape) if hasattr(value, "shape") else "scalar"
                    else:
                        data_types[key] = type(value).__name__
                        shapes[key] = "N/A"

        # Test complete
        elapsed_time = time.time() - start_time
        logger.info(
            f"Test completed successfully. Processed {examples_processed} examples in "
            f"{elapsed_time:.2f}s"
        )

        return {
            "status": "success",
            "dataset": path,
            "name": name,
            "split": split,
            "examples_processed": examples_processed,
            "data_types": data_types,
            "shapes": shapes,
            "elapsed_time": elapsed_time,
        }

    except Exception as e:
        # Log failure
        logger.error(f"Test failed: {str(e)}", exc_info=True)

        return {
            "status": "failure",
            "dataset": path,
            "name": name,
            "split": split,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
        }


def run_dataset_tests():
    """Run tests on all configured datasets."""
    logger.info(f"JAX devices: {jax.devices()}")
    logger.info(f"Testing {len(DATASET_CONFIGS)} dataset configurations")

    results = []
    for dataset_config in DATASET_CONFIGS:
        result = dataset_compatibility_test(
            path=dataset_config["path"],
            split=dataset_config["split"],
            streaming=dataset_config.get("streaming", True),
            name=dataset_config.get("name"),
            data_dir="./data/hf_cache",
        )
        results.append(result)

    # Summarize results
    logger.info("=== Test Results Summary ===")
    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Successful tests: {success_count}/{len(results)}")

    logger.info("Dataset compatibility details:")
    for result in results:
        status = "✅" if result["status"] == "success" else "❌"
        dataset_name = f"{result['dataset']}"
        if result.get("name"):
            dataset_name += f"/{result['name']}"

        logger.info(f"{status} {dataset_name} ({result['split']})")
        if result["status"] == "failure":
            logger.info(f"  Error: {result['error']}")
        else:
            logger.info(
                f"  Processed {result['examples_processed']} examples in "
                f"{result['elapsed_time']:.2f}s"
            )
            if result.get("data_types"):
                for key, dtype in result["data_types"].items():
                    shape = result["shapes"].get(key, "N/A")
                    logger.info(f"  - {key}: {dtype} (shape: {shape})")

    return results


if __name__ == "__main__":
    run_dataset_tests()
