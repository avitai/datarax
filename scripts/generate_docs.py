#!/usr/bin/env python3
"""
Modern Documentation Generator for Datarax.

============================================

PURPOSE:
    Automatically generates documentation from source code,
    with dynamic discovery of project structure and intelligent organization.

FEATURES:
    - Dynamic project structure discovery
    - Automatic module categorization
    - Docstring extraction and formatting
    - Markdown file aggregation from source
    - MkDocs navigation generation
    - Progress tracking and error handling
    - Incremental generation support

USAGE:
    python scripts/generate_docs.py [OPTIONS]

OPTIONS:
    --src-path PATH      Source directory (default: src/datarax)
    --docs-path PATH     Documentation output directory (default: docs)
    --clean              Clean existing docs before generation
    --no-mkdocs          Skip MkDocs configuration update
    --verbose            Enable verbose output
    --incremental        Only regenerate changed files

Examples:
    # Standard generation
    python scripts/generate_docs.py

    # Clean rebuild with verbose output
    python scripts/generate_docs.py --clean --verbose

    # Custom paths
    python scripts/generate_docs.py --src-path src/custom --docs-path docs/custom

OUTPUT:
    Creates/updates documentation in the specified docs directory with:
    - Organized module documentation
    - API reference
    - Copied markdown files from source
    - Updated MkDocs navigation

Author: Datarax Team
License: MIT
"""

import argparse
import ast
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    path: Path
    relative_path: Path
    module_name: str
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    has_docstring: bool = False
    docstring: str | None = None
    imports_count: int = 0
    last_modified: float = 0
    content_hash: str = ""


@dataclass
class DocSection:
    """Documentation section with modules."""

    name: str
    title: str
    path: Path
    modules: list[ModuleInfo] = field(default_factory=list)
    subsections: dict[str, "DocSection"] = field(default_factory=dict)


class ModernDocGenerator:
    """Modern documentation generator with dynamic discovery and smart organization."""

    def __init__(
        self,
        src_path: str = "src/datarax",
        docs_path: str = "docs",
        clean: bool = False,
        verbose: bool = False,
        incremental: bool = False,
    ):
        """Initialize the documentation generator.

        Args:
            src_path: Source code directory
            docs_path: Documentation output directory
            clean: Whether to clean existing docs
            verbose: Enable verbose output
            incremental: Only regenerate changed files
        """
        self.src_path = Path(src_path)
        self.docs_path = Path(docs_path)
        self.clean_mode = clean
        self.verbose = verbose
        self.incremental = incremental

        # Cache for incremental builds
        self.cache_file = self.docs_path / ".doc_cache.json"
        self.cache = self._load_cache() if incremental else {}

        # Discovered structure
        self.sections: dict[str, DocSection] = {}
        self.all_modules: list[ModuleInfo] = []

        # Section name mappings for better titles
        self.section_titles = {
            "core": "Core Components",
            "models": "Model Implementations",
            "training": "Training Systems",
            "inference": "Inference Pipeline",
            "data": "Data Processing",
            "utils": "Utilities",
            "configs": "Configuration",
            "benchmarks": "Benchmarks",
            "cli": "Command Line Interface",
            "extensions": "Extensions",
            "modalities": "Data Modalities",
            "visualization": "Visualization",
            "scaling": "Scaling & Distribution",
            "factories": "Model Factories",
            "losses": "Loss Functions",
            "sampling": "Sampling Methods",
            "metrics": "Evaluation Metrics",
            "optimization": "Optimization",
            "io": "Input/Output",
            "parallelism": "Parallelization",
            "protocols": "Interfaces & Protocols",
        }

    def _load_cache(self) -> dict:
        """Load documentation cache for incremental builds."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}

    def _save_cache(self):
        """Save documentation cache."""
        if self.incremental:
            try:
                self.docs_path.mkdir(exist_ok=True)
                with open(self.cache_file, "w") as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file contents for change detection."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _needs_regeneration(self, module_info: ModuleInfo) -> bool:
        """Check if a module needs documentation regeneration."""
        if not self.incremental:
            return True

        cache_key = str(module_info.relative_path)
        if cache_key not in self.cache:
            return True

        cached_hash = self.cache.get(cache_key, {}).get("hash", "")
        return cached_hash != module_info.content_hash

    def clean_existing_docs(self):
        """Clean existing documentation, preserving important files."""
        logger.info("🧹 Cleaning existing documentation...")

        # Files and directories to preserve
        preserve = {
            "index.md",
            "api.md",
            "_static",
            "_overrides",
            ".doc_cache.json",
            "migration",  # Preserve migration guides
        }

        if not self.docs_path.exists():
            self.docs_path.mkdir(parents=True)
            return

        for item in self.docs_path.iterdir():
            if item.name not in preserve:
                if item.is_file():
                    if self.verbose:
                        logger.info(f"  Removing: {item}")
                    item.unlink()
                elif item.is_dir():
                    if self.verbose:
                        logger.info(f"  Removing directory: {item}")
                    shutil.rmtree(item)

    def discover_project_structure(self):
        """Dynamically discover the project structure."""
        logger.info("🔍 Discovering project structure...")

        if not self.src_path.exists():
            logger.error(f"Source path does not exist: {self.src_path}")
            sys.exit(1)

        # Find all Python files
        python_files = list(self.src_path.rglob("*.py"))

        # Filter out test files and __pycache__
        python_files = [
            f
            for f in python_files
            if "__pycache__" not in str(f)
            and not f.name.startswith("test_")
            and "/tests/" not in str(f)
        ]

        logger.info(f"  Found {len(python_files)} Python files")

        # Organize files by directory structure
        for file_path in python_files:
            module_info = self._extract_module_info(file_path)
            if module_info:
                self.all_modules.append(module_info)
                self._categorize_module(module_info)

        # Log discovered sections
        logger.info(f"  Discovered {len(self.sections)} top-level sections")
        if self.verbose:
            for section_name in sorted(self.sections.keys()):
                section = self.sections[section_name]
                logger.info(f"    - {section_name}: {len(section.modules)} modules")

    def _extract_module_info(self, file_path: Path) -> ModuleInfo | None:
        """Extract information from a Python module."""
        try:
            relative_path = file_path.relative_to(self.src_path)

            # Skip __init__.py files unless they have substantial content
            if file_path.name == "__init__.py":
                if file_path.stat().st_size < 100:  # Skip small init files
                    return None

            # Create module name from path
            module_parts = [*relative_path.parts[:-1], relative_path.stem]
            module_name = ".".join(module_parts)

            # Read and parse file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Get file hash for incremental builds
            content_hash = self._get_file_hash(file_path)

            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                if self.verbose:
                    logger.warning(f"  Syntax error in {file_path}: {e}")
                return None

            # Extract module components
            classes = []
            functions = []
            imports_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("_") or node.name in ["__init__", "__call__"]:
                        functions.append(node.name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports_count += 1

            # Get module docstring
            docstring = ast.get_docstring(tree)

            return ModuleInfo(
                path=file_path,
                relative_path=relative_path,
                module_name=module_name,
                classes=classes,
                functions=functions,
                has_docstring=docstring is not None,
                docstring=docstring,
                imports_count=imports_count,
                last_modified=file_path.stat().st_mtime,
                content_hash=content_hash,
            )

        except Exception as e:
            if self.verbose:
                logger.warning(f"  Could not process {file_path}: {e}")
            return None

    def _categorize_module(self, module_info: ModuleInfo):
        """Categorize a module into the appropriate documentation section."""
        # Get the top-level directory from the module path
        parts = module_info.relative_path.parts

        if len(parts) == 1:
            # Root level module
            section_name = "root"
        else:
            # Get the first directory as the section
            section_name = parts[0]

            # Special handling for nested structures like generative_models
            if section_name == "generative_models" and len(parts) > 2:
                # Use the subdirectory as the section
                section_name = parts[1]

        # Create section if it doesn't exist
        if section_name not in self.sections:
            section_title = self.section_titles.get(
                section_name, section_name.replace("_", " ").title()
            )

            section_path = self.docs_path / section_name
            self.sections[section_name] = DocSection(
                name=section_name, title=section_title, path=section_path
            )

        # Add module to section
        self.sections[section_name].modules.append(module_info)

    def generate_module_documentation(self, module_info: ModuleInfo) -> str:
        """Generate documentation for a single module."""
        # Check if regeneration is needed
        if not self._needs_regeneration(module_info):
            if self.verbose:
                logger.info(f"    Skipping unchanged: {module_info.relative_path}")
            return ""

        lines = []

        # Module title
        module_title = module_info.path.stem.replace("_", " ").title()
        if module_title == "Init":
            module_title = "Package Initialization"

        # Construct the full dotted path for mkdocstrings
        # e.g. src/datarax/core/batcher.py -> datarax.core.batcher
        # module_info.module_name is currently relative to src_path (e.g. core.batcher)
        # We need to prepend the root package name based on the src path structure

        # Assumption: src_path is something like "src/datarax"
        # and module_info.module_name was calculated relative to that.
        # So "core.batcher" is correct relative to "datarax" package root.
        # Wait, the tool is called with src_path="src/datarax".
        # _extract_module_info calculates relative path from "src/datarax".
        # So module_name is "core.batcher".
        # The import path should be "datarax.core.batcher".

        root_package = self.src_path.name  # "datarax"
        full_module_path = f"{root_package}.{module_info.module_name}"

        lines.extend(
            [
                f"# {module_title}",
                "",
                f"::: {full_module_path}",
                "",
            ]
        )

        # Update cache
        if self.incremental:
            cache_key = str(module_info.relative_path)
            self.cache[cache_key] = {
                "hash": module_info.content_hash,
                "timestamp": module_info.last_modified,
            }

        return "\n".join(lines)

    def generate_section_documentation(self):
        """Generate documentation for all sections."""
        logger.info("📚 Generating section documentation...")

        for section_name in sorted(self.sections.keys()):
            section = self.sections[section_name]

            if not section.modules:
                continue

            logger.info(f"  Generating {section.title} ({len(section.modules)} modules)")

            # Create section directory
            section.path.mkdir(parents=True, exist_ok=True)

            # Generate index for the section
            self._generate_section_index(section)

            # Generate documentation for each module
            for module_info in sorted(section.modules, key=lambda m: m.module_name):
                if module_info.path.name == "__init__.py":
                    continue  # Skip init files for individual docs

                module_doc = self.generate_module_documentation(module_info)
                if module_doc:  # Only write if content was generated
                    doc_filename = f"{module_info.path.stem}.md"
                    doc_path = section.path / doc_filename

                    with open(doc_path, "w", encoding="utf-8") as f:
                        f.write(module_doc)

    def _generate_section_index(self, section: DocSection):
        """Generate index file for a documentation section."""
        lines = [
            f"# {section.title}",
            "",
            f"This section contains documentation for {section.title.lower()}.",
            "",
            "## Modules",
            "",
        ]

        # Group modules by subdirectory if applicable
        module_groups = defaultdict(list)

        for module_info in sorted(section.modules, key=lambda m: m.module_name):
            if module_info.path.name == "__init__.py":
                continue

            # Check if module is in a subdirectory
            rel_parts = module_info.relative_path.parts
            if len(rel_parts) > 2:
                group_name = rel_parts[-2]
            else:
                group_name = "main"

            module_groups[group_name].append(module_info)

        # Generate module listing
        if len(module_groups) == 1 and "main" in module_groups:
            # Simple flat listing
            for module_info in module_groups["main"]:
                doc_name = module_info.path.stem
                lines.append(f"- [{doc_name}]({doc_name}.md)")
        else:
            # Grouped listing
            for group_name in sorted(module_groups.keys()):
                if group_name != "main":
                    lines.extend(
                        [
                            "",
                            f"### {group_name.replace('_', ' ').title()}",
                            "",
                        ]
                    )

                for module_info in module_groups[group_name]:
                    doc_name = module_info.path.stem
                    lines.append(f"- [{doc_name}]({doc_name}.md)")

        # Add statistics
        lines.extend(
            [
                "",
                "## Statistics",
                "",
                f"- Total modules: {len(section.modules)}",
                f"- Total classes: {sum(len(m.classes) for m in section.modules)}",
                f"- Total functions: {sum(len(m.functions) for m in section.modules)}",
            ]
        )

        # Write section index
        index_path = section.path / "index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def copy_markdown_files(self):
        """Copy existing markdown files from source directories."""
        logger.info("📋 Copying markdown files from source...")

        copied_count = 0

        # Directories to scan for markdown files
        scan_dirs = ["src", "tests", "examples", "notebooks"]

        for dir_name in scan_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                continue

            # Find all markdown files
            markdown_files = list(dir_path.rglob("*.md"))

            for md_file in markdown_files:
                # Skip README files in subdirectories (keep only root README)
                if md_file.name == "README.md" and md_file.parent != dir_path:
                    continue

                # Calculate target path
                relative_path = md_file.relative_to(dir_path)
                target_path = self.docs_path / dir_name / relative_path

                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(md_file, target_path)
                copied_count += 1

                if self.verbose:
                    logger.info(f"  Copied: {md_file} -> {target_path}")

        logger.info(f"  Copied {copied_count} markdown files")

    def generate_main_index(self):
        """Generate the main documentation index."""
        logger.info("📝 Creating main documentation index...")

        lines = [
            "# Datarax: High-Performance JAX Data Pipelines",
            "",
            "[![CI](https://github.com/avitai/datarax/actions/workflows/ci.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/ci.yml)",
            "[![Test Coverage](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml/badge.svg)](https://github.com/avitai/datarax/actions/workflows/test-coverage.yml)",
            "[![codecov](https://codecov.io/gh/avitai/datarax/branch/main/graph/badge.svg)](https://codecov.io/gh/avitai/datarax)",
            "",
            "Datarax is a high-performance, extensible data pipeline framework specifically ",
            "engineered for JAX-based machine learning workflows. It simplifies and accelerates ",
            "the development of efficient and scalable data loading, preprocessing, and ",
            "augmentation pipelines for JAX, leveraging the full potential of JAX's ",
            "Just-In-Time (JIT) compilation, automatic differentiation, and hardware ",
            "acceleration capabilities.",
            "",
            "## Key Features",
            "",
            "- **High Performance:** Leverages JAX's JIT compilation and XLA backend to ",
            "achieve near-optimal data processing speeds on CPUs, GPUs, and TPUs.",
            "- **JAX-Native Design:** All core components and operations are designed with JAX's ",
            "functional programming paradigm and immutable data structures (PyTrees) in mind.",
            "- **Scalability:** Supports efficient data loading and processing for large datasets ",
            "and distributed training scenarios, including multi-host and multi-device setups.",
            "- **Extensibility:** Easily define and integrate custom data sources, ",
            "transformations, and augmentation operations.",
            "- **Usability:** Provides a clear, intuitive Python API and a flexible configuration ",
            "system for defining and managing pipelines.",
            "- **Determinism:** Pipeline runs are deterministic by default, crucial for ",
            "reproducibility.",
            "- **Complete Feature Set:** Supports common operators, advanced transformations, ",
            "batching, sharding, checkpointing, and caching.",
            "- **Ecosystem Integration:** Facilitates smooth integration with other JAX libraries ",
            "like Flax, Optax, and Orbax.",
            "",
            "## Quick Navigation",
            "",
        ]

        # Add links to main sections
        for section_name in sorted(self.sections.keys()):
            section = self.sections[section_name]
            if section.modules:  # Only include non-empty sections
                lines.append(
                    f"- [{section.title}]({section_name}/index.md) - {len(section.modules)} modules"
                )

        lines.extend(
            [
                "",
                "## Installation",
                "",
                "```bash",
                "# Install via uv (recommended)",
                "uv pip install datarax",
                "",
                "# Install with optional dependencies",
                "uv pip install datarax[data]     # For data loading (HF, TFDS)",
                "uv pip install datarax[gpu]      # For GPU support",
                "",
                "# Or locally for development",
                "pip install -e .",
                "```",
                "",
                "## Quick Start",
                "",
                "Here's a simple example of using Datarax's DAG-based architecture:",
                "",
                "```python",
                "import jax",
                "import jax.numpy as jnp",
                "from flax import nnx",
                "from datarax import from_source",
                "from datarax.core.nodes import OperatorNode",
                "from datarax.operators import ElementOperator, ElementOperatorConfig",
                "from datarax.sources import MemorySource, MemorySourceConfig",
                "",
                "# 1. Define operations",
                "def normalize(element, key=None):",
                '    return element.update_data({"image": element.data["image"] / 255.0})',
                "",
                "# 2. Create data source",
                "source_config = MemorySourceConfig()",
                "source = MemorySource(source_config, data=my_data_dict, rngs=nnx.Rngs(0))",
                "",
                "# 3. Create operators",
                "normalizer = ElementOperator(",
                "    ElementOperatorConfig(), ",
                "    fn=normalize, ",
                "    rngs=nnx.Rngs(0)",
                ")",
                "",
                "# 4. Build pipeline",
                "pipeline = (",
                "    from_source(source, batch_size=32)",
                "    >> OperatorNode(normalizer)",
                ")",
                "",
                "# 5. Run pipeline",
                "for batch in pipeline:",
                "    process(batch)",
                "```",
                "",
                "## Documentation Structure",
                "",
                "- **API Reference** - Complete API documentation",
                "- **Module Documentation** - Detailed documentation for each module",
                "- **Examples** - Usage examples and tutorials",
                "- **Migration Guides** - Guides for migrating between versions",
                "",
                "## Project Statistics",
                "",
                f"- Total Modules: {len(self.all_modules)}",
                f"- Total Classes: {sum(len(m.classes) for m in self.all_modules)}",
                f"- Total Functions: {sum(len(m.functions) for m in self.all_modules)}",
                "- Documentation Coverage: "
                f"{sum(1 for m in self.all_modules if m.has_docstring)}/"
                f"{len(self.all_modules)} modules with docstrings",
                "",
                "## Contributing",
                "",
                "To contribute to the documentation:",
                "",
                "1. Add docstrings to your Python code",
                "2. Run the documentation generator: `python scripts/generate_docs.py`",
                "3. Build the documentation: `mkdocs build`",
                "4. Preview locally: `mkdocs serve`",
                "",
                "---",
                "",
                "*Documentation generated automatically by generate_docs.py*",
            ]
        )

        # Write main index
        index_path = self.docs_path / "index.md"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def update_mkdocs_config(self):
        """Update MkDocs configuration with the generated documentation structure."""
        logger.info("🔧 Updating MkDocs configuration...")

        mkdocs_path = Path("mkdocs.yml")

        if not mkdocs_path.exists():
            logger.warning("  mkdocs.yml not found, skipping navigation update")
            return

        try:
            # Read current configuration
            with open(mkdocs_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Generate navigation structure
            nav_items = ["  - Home: index.md"]

            # Add manual guides if they exist
            if (self.docs_path / "getting_started").exists():
                nav_items.append("  - Getting Started:")
                nav_items.append("      - Installation: getting_started/installation.md")
                nav_items.append("      - Quick Start: getting_started/quick_start.md")
                nav_items.append("      - Core Concepts: getting_started/core_concepts.md")

            if (self.docs_path / "user_guide").exists():
                nav_items.append("  - User Guide:")
                nav_items.append("      - Data Sources: user_guide/data_sources.md")
                nav_items.append("      - DAG Construction: user_guide/dag_construction.md")
                nav_items.append("      - Benchmarking: user_guide/benchmarking.md")
                nav_items.append("      - Troubleshooting: user_guide/troubleshooting_guide.md")
                nav_items.append("      - NNX Best Practices: user_guide/nnx_best_practices.md")
                nav_items.append("      - Checkpointing: user_guide/checkpointing_guide.md")

            if (self.docs_path / "examples").exists():
                nav_items.append("  - Examples: examples/README.md")

            # Add API reference section
            nav_items.append("  - API Reference:")

            # Add sections in a logical order
            section_order = [
                "core",
                "models",
                "training",
                "inference",
                "data",
                "modalities",
                "benchmarks",
                "configs",
                "factories",
                "scaling",
                "optimization",
                "utils",
                "cli",
                "extensions",
                "visualization",
            ]

            # Add ordered sections first
            for section_name in section_order:
                if section_name in self.sections:
                    section = self.sections[section_name]
                    if section.modules:
                        nav_items.append(f"      - {section.title}:")
                        nav_items.append(f"          - Overview: {section_name}/index.md")

                        # Group modules by subdirectory
                        module_groups = defaultdict(list)
                        for module_info in sorted(section.modules, key=lambda m: m.module_name):
                            if module_info.path.name == "__init__.py":
                                continue

                            rel_parts = module_info.relative_path.parts
                            if len(rel_parts) > 2:
                                group_name = rel_parts[-2]
                            else:
                                group_name = "main"
                            module_groups[group_name].append(module_info)

                        # Add main modules
                        if "main" in module_groups:
                            for module_info in sorted(
                                module_groups["main"], key=lambda m: m.module_name
                            ):
                                nav_items.append(
                                    f"          - {module_info.path.stem}: "
                                    f"{section_name}/{module_info.path.stem}.md"
                                )

                        # Add grouped modules
                        for group_name in sorted(module_groups.keys()):
                            if group_name != "main":
                                nav_items.append(
                                    f"          - {group_name.replace('_', ' ').title()}:"
                                )
                                for module_info in sorted(
                                    module_groups[group_name], key=lambda m: m.module_name
                                ):
                                    nav_items.append(
                                        f"              - {module_info.path.stem}: "
                                        f"{section_name}/{module_info.path.stem}.md"
                                    )

            # Add any remaining sections
            for section_name in sorted(self.sections.keys()):
                if section_name not in section_order and section_name != "examples":
                    section = self.sections[section_name]
                    if section.modules:
                        nav_items.append(f"      - {section.title}:")
                        nav_items.append(f"          - Overview: {section_name}/index.md")

                        # Group modules by subdirectory
                        module_groups = defaultdict(list)
                        for module_info in sorted(section.modules, key=lambda m: m.module_name):
                            if module_info.path.name == "__init__.py":
                                continue

                            rel_parts = module_info.relative_path.parts
                            if len(rel_parts) > 2:
                                group_name = rel_parts[-2]
                            else:
                                group_name = "main"
                            module_groups[group_name].append(module_info)

                        # Add main modules
                        if "main" in module_groups:
                            for module_info in sorted(
                                module_groups["main"], key=lambda m: m.module_name
                            ):
                                nav_items.append(
                                    f"          - {module_info.path.stem}: "
                                    f"{section_name}/{module_info.path.stem}.md"
                                )

                        # Add grouped modules
                        for group_name in sorted(module_groups.keys()):
                            if group_name != "main":
                                nav_items.append(
                                    f"          - {group_name.replace('_', ' ').title()}:"
                                )
                                for module_info in sorted(
                                    module_groups[group_name], key=lambda m: m.module_name
                                ):
                                    nav_items.append(
                                        f"              - {module_info.path.stem}: "
                                        f"{section_name}/{module_info.path.stem}.md"
                                    )

            if (self.docs_path / "distributed_training.md").exists():
                nav_items.append("  - Distributed Training: distributed_training.md")

            if (self.docs_path / "contributing").exists():
                nav_items.append("  - Contributing:")
                nav_items.append("      - Guide: contributing/contributing_guide.md")
                nav_items.append(
                    "      - Performance: contributing/performance_optimization_guide.md"
                )

            # Create navigation section
            nav_content = "nav:\n" + "\n".join(nav_items) + "\n"

            # Replace navigation in mkdocs.yml
            nav_pattern = r"nav:\s*\n(?:(?:  |\t).*\n)*"
            new_content = re.sub(nav_pattern, nav_content, content)

            # Write updated configuration
            with open(mkdocs_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            logger.info("  MkDocs configuration updated successfully")

        except Exception as e:
            logger.error(f"  Failed to update mkdocs.yml: {e}")

    def generate_summary(self):
        """Generate a summary of the documentation generation."""
        logger.info("📊 Documentation Generation Summary")
        logger.info("=" * 50)
        logger.info(f"  Total modules processed: {len(self.all_modules)}")
        logger.info(f"  Total sections created: {len(self.sections)}")
        logger.info(
            f"  Modules with docstrings: {sum(1 for m in self.all_modules if m.has_docstring)}"
        )

        if self.incremental:
            logger.info(f"  Cache entries: {len(self.cache)}")

        # Show section breakdown
        logger.info("\n  Section breakdown:")
        for section_name in sorted(self.sections.keys()):
            section = self.sections[section_name]
            if section.modules:
                logger.info(f"    - {section.title}: {len(section.modules)} modules")

    def run(self):
        """Run the complete documentation generation process."""
        logger.info("🚀 Starting modern documentation generation...")

        try:
            # Step 1: Clean if requested
            if self.clean_mode:
                self.clean_existing_docs()

            # Step 2: Discover project structure
            self.discover_project_structure()

            # Step 3: Generate section documentation
            self.generate_section_documentation()

            # Step 4: Copy markdown files
            self.copy_markdown_files()

            # Step 5: Generate main index
            self.generate_main_index()

            # Step 6: Update MkDocs config
            self.update_mkdocs_config()

            # Step 7: Save cache for incremental builds
            self._save_cache()

            # Step 8: Generate summary
            self.generate_summary()

            logger.info("\n✅ Documentation generation complete!")
            logger.info("📖 Run 'mkdocs build' to build the documentation")
            logger.info("🌐 Run 'mkdocs serve' to preview the documentation")

        except Exception as e:
            logger.error(f"❌ Documentation generation failed: {e}")
            if self.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point for the documentation generator."""
    parser = argparse.ArgumentParser(
        description="Modern documentation generator for Datarax",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--src-path", default="src/datarax", help="Source code directory (default: src/datarax)"
    )

    parser.add_argument(
        "--docs-path", default="docs", help="Documentation output directory (default: docs)"
    )

    parser.add_argument(
        "--clean", action="store_true", help="Clean existing documentation before generation"
    )

    parser.add_argument("--no-mkdocs", action="store_true", help="Skip MkDocs configuration update")

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Only regenerate changed files (faster for large projects)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run generator
    generator = ModernDocGenerator(
        src_path=args.src_path,
        docs_path=args.docs_path,
        clean=args.clean,
        verbose=args.verbose,
        incremental=args.incremental,
    )

    generator.run()


if __name__ == "__main__":
    main()
