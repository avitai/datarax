#!/usr/bin/env python3
"""
Test Plan Implementation Tracker

This script scans the test files and generates a report on the implementation
progress of the complete test plan.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Set


# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
ENDC = "\033[0m"
BOLD = "\033[1m"

# Regex patterns for test detection
TEST_FUNCTION_PATTERN = re.compile(r"def\s+(test_\w+)\s*\(")
TEST_CLASS_PATTERN = re.compile(r"class\s+(Test\w+)\s*\(")
TEST_SKIPPED_PATTERN = re.compile(r"@pytest.mark.skip|pytest.skip\(")


@dataclass
class ModuleStatus:
    """Status of test implementation for a module."""

    implemented: Set[str] = None
    skipped: Set[str] = None
    total_planned: int = 0

    def __post_init__(self):
        if self.implemented is None:
            self.implemented = set()
        if self.skipped is None:
            self.skipped = set()

    @property
    def implementation_percentage(self) -> float:
        """Calculate the percentage of implemented tests."""
        if self.total_planned == 0:
            return 0.0
        return (len(self.implemented) / self.total_planned) * 100


def find_test_files(tests_dir: Path) -> list[Path]:
    """Find all Python test files in the given directory."""
    return list(tests_dir.glob("**/test_*.py"))


def extract_test_names(file_path: Path) -> tuple[Set[str], Set[str]]:
    """Extract test names and skipped tests from a file."""
    implemented = set()
    skipped = set()

    with open(file_path, "r") as f:
        content = f.read()

    # Extract test functions
    for match in TEST_FUNCTION_PATTERN.finditer(content):
        test_name = match.group(1)
        start = match.start()

        # Check if the test is skipped
        line_start = content.rfind("\n", 0, start) + 1
        if line_start > 0:
            preceding_lines = content[max(0, line_start - 300) : start]
            if TEST_SKIPPED_PATTERN.search(preceding_lines):
                skipped.add(test_name)
            else:
                implemented.add(test_name)

    # Extract test classes and their methods
    for class_match in TEST_CLASS_PATTERN.finditer(content):
        class_name = class_match.group(1)
        class_start = class_match.end()

        # Find the end of the class definition
        class_level = 1
        class_end = class_start
        for i, char in enumerate(content[class_start:], class_start):
            if char == "{":
                class_level += 1
            elif char == "}":
                class_level -= 1
                if class_level == 0:
                    class_end = i + 1
                    break

        # Extract test methods within the class
        class_content = content[class_start:class_end]
        for method_match in re.finditer(r"def\s+(test_\w+)\s*\(", class_content):
            test_name = f"{class_name}.{method_match.group(1)}"
            method_start = method_match.start()

            # Check if the method is skipped
            method_line_start = class_content.rfind("\n", 0, method_start) + 1
            if method_line_start > 0:
                preceding_lines = class_content[max(0, method_line_start - 300) : method_start]
                if TEST_SKIPPED_PATTERN.search(preceding_lines):
                    skipped.add(test_name)
                else:
                    implemented.add(test_name)

    return implemented, skipped


def load_test_plan() -> dict[str, list[str]]:
    """Load the test plan from the complete test plan document."""
    # This is a simplified version; in practice, you would parse the actual document
    # or maintain a separate machine-readable representation of the test plan

    # Example structure (would be populated from an actual file)
    return {
        "core": [
            "test_pipeline_creation",
            "test_pipeline_execution",
            "test_pipeline_error_handling",
            "test_pipeline_with_transforms",
            "test_pipeline_state_management",
            "test_random_key_generation",
            "test_random_seed_reproduction",
            "test_split_keys",
            "test_random_in_pipeline",
            "test_source_interface_implementations",
            "test_transform_interface_implementations",
            "test_batch_interface_implementations",
        ],
        "sources": [
            "test_memory_source_creation",
            "test_memory_source_iteration",
            "test_memory_source_batch_shape",
            "test_memory_source_with_numpy",
            "test_memory_source_with_jax",
            "test_tfds_source_creation",
            "test_tfds_source_iteration",
            "test_tfds_source_feature_extraction",
            "test_tfds_source_with_transforms",
            "test_hf_source_creation",
            "test_hf_source_iteration",
            "test_hf_source_feature_extraction",
            "test_hf_source_with_transforms",
            "test_prefetchable_source_basic",
            "test_prefetchable_source_performance",
            "test_prefetchable_source_with_random",
        ],
        # ... other modules would be included
    }


def analyze_test_coverage(
    tests_dir: Path, test_plan: dict[str, list[str]]
) -> dict[str, ModuleStatus]:
    """Analyze test coverage based on the test plan."""
    test_files = find_test_files(tests_dir)

    # Initialize status for each module
    module_status = {}
    for module, planned_tests in test_plan.items():
        module_status[module] = ModuleStatus(total_planned=len(planned_tests))

    # Analyze each test file
    for file_path in test_files:
        # Determine which module this test file belongs to
        relative_path = file_path.relative_to(tests_dir)
        module = str(relative_path).split(os.sep)[0]

        # Skip if module is not in our test plan
        if module not in module_status:
            continue

        implemented, skipped = extract_test_names(file_path)
        module_status[module].implemented.update(implemented)
        module_status[module].skipped.update(skipped)

    return module_status


def generate_progress_report(module_status: dict[str, ModuleStatus]) -> str:
    """Generate a progress report in Markdown format."""
    report = "# Datarax Test Plan Implementation Progress\n\n"

    # Overall statistics
    total_implemented = sum(len(status.implemented) for status in module_status.values())
    total_skipped = sum(len(status.skipped) for status in module_status.values())
    total_planned = sum(status.total_planned for status in module_status.values())

    overall_percentage = 0
    if total_planned > 0:
        overall_percentage = (total_implemented / total_planned) * 100

    report += "## Overall Progress\n\n"
    report += (
        f"- **Implemented Tests:** {total_implemented} / {total_planned} "
        f"({overall_percentage:.1f}%)\n"
    )
    report += f"- **Skipped Tests:** {total_skipped}\n\n"

    # Module-specific statistics
    report += "## Module-specific Progress\n\n"
    report += "| Module | Implemented | Skipped | Total Planned | Progress |\n"
    report += "|--------|-------------|---------|---------------|----------|\n"

    for module, status in sorted(module_status.items()):
        progress = status.implementation_percentage
        report += (
            f"| {module} | {len(status.implemented)} | "
            f"{len(status.skipped)} | {status.total_planned} | "
            f"{progress:.1f}% |\n"
        )

    return report


def print_terminal_report(module_status: dict[str, ModuleStatus]) -> None:
    """Print a colorized report to the terminal."""
    # Overall statistics
    total_implemented = sum(len(status.implemented) for status in module_status.values())
    total_skipped = sum(len(status.skipped) for status in module_status.values())
    total_planned = sum(status.total_planned for status in module_status.values())

    overall_percentage = 0
    if total_planned > 0:
        overall_percentage = (total_implemented / total_planned) * 100

    print(f"{BOLD}Datarax Test Plan Implementation Progress{ENDC}\n")

    print(f"{BOLD}Overall Progress:{ENDC}")
    print(
        f"- Implemented Tests: {GREEN}{total_implemented}{ENDC} / {total_planned} "
        f"({CYAN}{overall_percentage:.1f}%{ENDC})"
    )
    print(f"- Skipped Tests: {YELLOW}{total_skipped}{ENDC}\n")

    print(f"{BOLD}Module-specific Progress:{ENDC}")
    print(f"{'Module':<15} {'Implemented':<15} {'Skipped':<10} {'Total':<10} {'Progress':<10}")
    print("-" * 60)

    for module, status in sorted(module_status.items()):
        progress = status.implementation_percentage
        progress_color = GREEN if progress >= 80 else (YELLOW if progress >= 40 else RED)

        print(
            f"{BLUE}{module:<15}{ENDC} "
            f"{GREEN}{len(status.implemented):<15}{ENDC} "
            f"{YELLOW}{len(status.skipped):<10}{ENDC} "
            f"{status.total_planned:<10} "
            f"{progress_color}{progress:.1f}%{ENDC}"
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Track Datarax test plan implementation")
    parser.add_argument("--tests-dir", type=str, default="tests", help="Directory containing tests")
    parser.add_argument("--output", type=str, help="Output file for Markdown report")
    args = parser.parse_args()

    tests_dir = Path(args.tests_dir)
    if not tests_dir.exists():
        print(f"Error: Tests directory '{tests_dir}' does not exist", file=sys.stderr)
        return 1

    # Load test plan and analyze coverage
    test_plan = load_test_plan()
    module_status = analyze_test_coverage(tests_dir, test_plan)

    # Print terminal report
    print_terminal_report(module_status)

    # Generate and save Markdown report if requested
    if args.output:
        report = generate_progress_report(module_status)
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nMarkdown report saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
