#!/usr/bin/env python3
"""
Component Extractor for Datarax Tests.

This script extracts components from test files based on imports and
provides a way to reorganize them along component lines.
"""

import os
import re
import shutil
from pathlib import Path


# Mapping of source modules to their components
# Updated to reflect the unified operator architecture
MODULE_COMPONENTS = {
    "core": ["pipeline", "stream", "batching", "worker", "operator", "structural", "module"],
    "operators": ["element_operator", "map_operator", "composite", "batch_mix", "modality"],
    "sources": ["memory_source", "tfds_source", "hf_source", "prefetchable"],
    "checkpoint": ["handlers", "iterators"],
    "utils": ["rng", "vectorize", "gpu", "prng", "nnx"],
    "monitoring": ["callbacks", "metrics", "reporters"],
    "dag": ["dag_executor", "dag_config", "nodes"],
}


class ComponentExtractor:
    """Extracts components from test files and organizes them by module."""

    def __init__(self, tests_dir: str):
        """Initialize with the tests directory path."""
        self.tests_dir = Path(tests_dir)
        self.backup_dir = self.tests_dir / "component_backup"
        os.makedirs(self.backup_dir, exist_ok=True)

    def _backup_file(self, file_path: Path) -> None:
        """Create a backup of the file."""
        backup_path = self.backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        print(f"Backed up {file_path.name} to {backup_path}")

    def _analyze_file(self, file_path: Path) -> dict[str, set[str]]:
        """Analyze a file to determine which components it tests."""
        content = file_path.read_text()

        # Track modules and components found
        modules_found = {}

        # Check for datarax imports
        for module, components in MODULE_COMPONENTS.items():
            if f"datarax.{module}" in content:
                modules_found[module] = set()

                # Check for specific component imports
                for component in components:
                    if component in content.lower():
                        modules_found[module].add(component)

        return modules_found

    def extract_integration_components(self, file_path: Path) -> dict[str, list[str]]:
        """Extract components from an integration test for reorganization."""
        # Analyze the file to identify components
        modules = self._analyze_file(file_path)
        if not modules:
            return {}

        # Create new files based on detected components
        new_files = {}
        file_content = file_path.read_text()

        for module, components in modules.items():
            if not components:
                continue

            for component in components:
                # Determine new file path
                new_file_path = self.tests_dir / module / f"test_{component}_integration.py"

                # Extract test code relevant to this component
                # This is a simplified approach - we're looking for functions
                # containing the component name
                # In a real implementation, more sophisticated parsing would
                # be needed

                # Check for functions testing this component
                component_pattern = re.compile(
                    rf"def test_[a-zA-Z0-9_]*{component}[a-zA-Z0-9_]*\s*\(", re.IGNORECASE
                )

                # Find all matches
                matches = list(component_pattern.finditer(file_content))
                if not matches:
                    continue

                # Create header with imports and helpers
                header_end = matches[0].start()
                header = file_content[:header_end].strip()

                # Create a file with the header and matching component tests
                component_tests = []
                for i, match in enumerate(matches):
                    start = match.start()

                    # Find end of function (next function def or EOF)
                    if i < len(matches) - 1:
                        next_match = matches[i + 1]
                        end = next_match.start()
                    else:
                        end = len(file_content)

                    component_tests.append(file_content[start:end])

                # Combine into new file content
                new_content = f"{header}\n\n\n" + "\n\n".join(component_tests)
                new_files[str(new_file_path)] = new_content

        return new_files

    def reorganize_integration_tests(self) -> None:
        """Reorganize integration tests into component-specific files."""
        integration_dir = self.tests_dir / "integration"
        if not integration_dir.exists():
            print("Integration directory not found")
            return

        for file_path in integration_dir.glob("test_*.py"):
            print(f"Analyzing {file_path}")
            self._backup_file(file_path)

            # Extract component tests
            new_files = self.extract_integration_components(file_path)

            # Create new component-specific files
            for new_path, content in new_files.items():
                # Ensure the directory exists
                new_file_path = Path(new_path)
                os.makedirs(new_file_path.parent, exist_ok=True)

                # Ensure __init__.py exists in directory
                init_path = new_file_path.parent / "__init__.py"
                if not init_path.exists():
                    init_path.touch()

                # Create the new file
                print(f"Creating {new_file_path}")
                new_file_path.write_text(content)

            print(f"Processed {file_path}")

    def run(self) -> None:
        """Run the complete component extraction process."""
        print("Starting component extraction...")
        self.reorganize_integration_tests()
        print("\nComponent extraction completed!")
        print(f"Original files have been backed up to {self.backup_dir}")


if __name__ == "__main__":
    extractor = ComponentExtractor("tests")
    extractor.run()
