"""Tests for scripts/check_docs_pairs.py: code a docs page shows with output is its script's."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script


_SCRIPT = """# %%
size = 4
values = list(range(size))
print(values)

# %%
total = sum(values)
print(total)
"""

_PAIR = """```python
{code}
```

**Terminal Output:**

```text
{output}
```
"""

_LINK = "[`examples/core/01_small.py`](https://github.com/avitai/datarax/blob/main/examples/core/01_small.py)"


@pytest.fixture(scope="module")
def check_docs_pairs() -> ModuleType:
    return load_script("check_docs_pairs")


def _page(root: Path, body: str, *, links: str = _LINK) -> Path:
    """Write the script and a docs page whose body follows the link line."""
    script = root / "examples" / "core" / "01_small.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_SCRIPT, encoding="utf-8")
    page = root / "docs" / "examples" / "core" / "small.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(f"# Small\n\n- Script: {links}\n\n{body}", encoding="utf-8")
    return page


def test_code_copied_from_the_script_holds(check_docs_pairs: ModuleType, tmp_path: Path) -> None:
    body = _PAIR.format(code="total = sum(values)\nprint(total)", output="6")
    page = _page(tmp_path, body)

    assert check_docs_pairs.check_page(page, tmp_path) == []


def test_a_changed_line_is_reported_with_the_line(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    body = _PAIR.format(code="total = sum(values) + 1\nprint(total)", output="7")
    page = _page(tmp_path, body)

    findings = check_docs_pairs.check_page(page, tmp_path)

    assert len(findings) == 1
    assert "total = sum(values) + 1" in findings[0]


def test_script_lines_out_of_order_are_reported(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    """Each line exists in the script, but not as one contiguous run."""
    body = _PAIR.format(code="print(total)\ntotal = sum(values)", output="6")
    page = _page(tmp_path, body)

    assert len(check_docs_pairs.check_page(page, tmp_path)) == 1


def test_trailing_whitespace_does_not_count(check_docs_pairs: ModuleType, tmp_path: Path) -> None:
    body = _PAIR.format(code="total = sum(values)   \nprint(total)", output="6")
    page = _page(tmp_path, body)

    assert check_docs_pairs.check_page(page, tmp_path) == []


def test_a_block_showing_no_output_is_not_checked(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    """An illustrative snippet without a Terminal Output block is prose, not the script's run."""
    body = "```python\nfrom datarax import Pipeline  # an illustration\n```\n"
    page = _page(tmp_path, body)

    assert check_docs_pairs.check_page(page, tmp_path) == []


@pytest.mark.parametrize(
    "links",
    [
        "no script named here",
        _LINK + " and [`examples/core/02_other.py`](https://example.org/examples/core/02_other.py)",
    ],
    ids=["no script", "two scripts"],
)
def test_a_page_with_output_must_name_exactly_one_script(
    check_docs_pairs: ModuleType, tmp_path: Path, links: str
) -> None:
    """Which script a pair comes from must be unambiguous, so such a page fails, never skips."""
    other = tmp_path / "examples" / "core" / "02_other.py"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("value = 1\n", encoding="utf-8")
    body = _PAIR.format(code="total = sum(values)\nprint(total)", output="6")
    page = _page(tmp_path, body, links=links)

    findings = check_docs_pairs.check_page(page, tmp_path)

    assert len(findings) == 1
    assert "exactly one" in findings[0]


def test_the_pages_checked_are_the_docs_examples_pages(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    page = _page(tmp_path, _PAIR.format(code="print(total)", output="6"))
    outside = tmp_path / "docs" / "user_guide" / "guide.md"
    outside.parent.mkdir(parents=True)
    outside.write_text("# Guide\n", encoding="utf-8")

    assert check_docs_pairs.docs_pages(tmp_path) == [page]


def test_the_command_line_exits_non_zero_on_a_finding(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    _page(tmp_path, _PAIR.format(code="print(total + 1)", output="7"))

    assert check_docs_pairs.main(["--root", str(tmp_path)]) == 1


def test_the_command_line_exits_zero_when_every_pair_holds(
    check_docs_pairs: ModuleType, tmp_path: Path
) -> None:
    _page(tmp_path, _PAIR.format(code="print(total)", output="6"))

    assert check_docs_pairs.main(["--root", str(tmp_path)]) == 0
