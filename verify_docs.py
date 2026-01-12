import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import List, Tuple


def extract_code_blocks(md_content: str) -> List[Tuple[int, str]]:
    """Extract python code blocks from markdown content.

    Returns list of (line_number, code_content).
    """
    lines = md_content.split("\n")
    blocks = []
    in_block = False
    start_line = 0
    current_block = []

    for i, line in enumerate(lines):
        if line.strip().startswith("```python"):
            if in_block:
                # Nested or broken block, markdown doesn't support nested without indent
                pass
            in_block = True
            start_line = i + 1
            current_block = []
            continue

        if line.strip() == "```" and in_block:
            in_block = False
            code = "\n".join(current_block)
            blocks.append((start_line, code))
            current_block = []
            continue

        if in_block:
            current_block.append(line)

    return blocks


def run_file(file_path: str):
    print(f"Checking {file_path}...")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"Skipping binary/unreadable file: {file_path}")
        return True

    blocks = extract_code_blocks(content)

    if not blocks:
        print("  No python blocks found.")
        return True

    # Global/Local scope for the file
    # maintaining state across blocks in the single file
    file_globals = {}

    for start_line, code in blocks:
        if "TODO" in code or "..." in code and len(code.strip()) < 20:
            # Heuristic to skip pseudo-code or small placeholders
            # But let's try to run everything unless it's obviously just text
            pass

        print(f"  Line {start_line}: Running block...")
        try:
            # We wrap in exec to handle statements
            exec(code, file_globals)  # nosec B102
        except Exception:
            print(f"FAILED at line {start_line} in {file_path}")
            print("-" * 40)
            print(code)
            print("-" * 40)
            traceback.print_exc()
            return False

    print(f"PASSED {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify python code blocks in markdown.")
    parser.add_argument("path", help="File or directory to verify")
    args = parser.parse_args()

    target = Path(args.path)

    if target.is_file():
        success = run_file(str(target))
        sys.exit(0 if success else 1)

    if target.is_dir():
        failed_files = []
        for root, dirs, files in os.walk(target):
            for file in files:
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    if not run_file(full_path):
                        failed_files.append(full_path)

        if failed_files:
            print("\nfailures detected in:")
            for f in failed_files:
                print(f"- {f}")
            sys.exit(1)
        else:
            print("\nAll files passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()
