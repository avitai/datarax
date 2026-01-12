"""Standardized utilities for handling optional dependencies in tests.

This module provides standardized functions and decorators for dealing with
optional dependencies in the Datarax test suite.
"""

import functools
import logging
from typing import Callable

import pytest


# Configure logging
logger = logging.getLogger(__name__)


def require_package(package_name: str, reason: str | None = None) -> Callable:
    """Skip a test if a required package is not available.

    Args:
        package_name: Name of the package to check for
        reason: Optional reason for skipping (will be added to default message)

    Returns:
        Decorator that skips the test if the package is not available
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Test requires {package_name}"
            if reason:
                message += f": {reason}"

            # Use importorskip to check for the package
            pytest.importorskip(package_name, reason=message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_packages(package_names: list[str], reason: str | None = None) -> Callable:
    """Skip a test if any of the required packages are not available.

    Args:
        package_names: List of package names to check for
        reason: Optional reason for skipping (will be added to default message)

    Returns:
        Decorator that skips the test if any package is not available
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for package in package_names:
                message = f"Test requires {package}"
                if reason:
                    message += f": {reason}"

                # Use importorskip to check for each package
                pytest.importorskip(package, reason=message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def lazy_import(
    package_name: str, as_name: str | None = None, error_handler: Callable | None = None
):
    """Lazily import a package, returning None if the package is not available.

    Args:
        package_name: Name of the package to import
        as_name: Optional alias for the imported package
        error_handler: Optional function to call with the ImportError if import fails

    Returns:
        The imported module or None if not available
    """
    try:
        module = __import__(package_name)
        if as_name:
            return module
        # Handle dot notation imports (e.g., "tensorflow.data")
        for comp in package_name.split(".")[1:]:
            module = getattr(module, comp)
        return module
    except ImportError as e:
        if error_handler is not None:
            error_handler(e)
        return None


# Common dependency groups
TENSORFLOW_DEPENDENCIES = ["tensorflow", "tensorflow_datasets"]
HUGGINGFACE_DEPENDENCIES = ["datasets", "transformers"]
IMAGE_DEPENDENCIES = ["PIL", "matplotlib"]
NLP_DEPENDENCIES = ["nltk", "transformers"]


# Pre-configured decorators for common dependencies
require_tensorflow = require_packages(
    TENSORFLOW_DEPENDENCIES, "TensorFlow and TensorFlow Datasets required"
)

require_huggingface = require_packages(
    HUGGINGFACE_DEPENDENCIES, "HuggingFace Datasets and Transformers required"
)

require_image_processing = require_packages(
    IMAGE_DEPENDENCIES, "PIL and matplotlib required for image processing"
)

require_nlp = require_packages(NLP_DEPENDENCIES, "NLTK and Transformers required for NLP tasks")


def log_missing_dependencies():
    """Log information about missing optional dependencies.

    This is useful to run at the start of the test session to provide
    information about which optional dependencies are missing.
    """
    dependency_groups = {
        "TensorFlow": TENSORFLOW_DEPENDENCIES,
        "HuggingFace": HUGGINGFACE_DEPENDENCIES,
        "Image Processing": IMAGE_DEPENDENCIES,
        "NLP": NLP_DEPENDENCIES,
    }

    missing_deps: dict[str, list[str]] = {}
    for group_name, deps in dependency_groups.items():
        missing = []
        for dep in deps:
            if lazy_import(dep) is None:
                missing.append(dep)
        if missing:
            missing_deps[group_name] = missing

    if missing_deps:
        logger.warning("Missing optional dependencies:")
        for group, deps in missing_deps.items():
            logger.warning(f"  {group}: {', '.join(deps)}")
    else:
        logger.info("All optional dependencies are available.")


def pytest_configure():
    """Configure the module for pytest."""
    # Log information about missing dependencies
    log_missing_dependencies()
