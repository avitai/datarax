"""No source holds a seed of its own: every order is drawn from the pipeline's key (brief T2).

A literal seed in a source's code (a ``seed=0`` default, ``np.random.default_rng(3)``,
``jax.random.key(0)``) would order records by a number nobody chose. The search reads code only,
strings and comments blanked, so a docstring's example of a user's ``nnx.Rngs(0)`` is not a
source's seed. A planted literal in a copy of the sources is found, so a clean result is a
measurement and not a broken search.
"""

from __future__ import annotations

import io
import re
import shutil
import tokenize
from pathlib import Path

import datarax.sources


_SOURCES = Path(datarax.sources.__file__).parent

_LITERAL_SEED = re.compile(
    r"\bseed\s*(?::[^=\n]*)?=\s*\d"  # seed = 0, seed: int = 42, f(seed=7)
    r"|\b(?:PRNGKey|key|Rngs|default_rng|RandomState|Philox|seed)\(\s*\d"  # key(0), Rngs(0)
)

# ArrayRecord orders its records by a seed of its own until C5a-4 reads it as an INDEXED source
# (OWN-1001-ARRAYRECORD); this exclusion goes with that step.
_ORDERED_BY_ITS_OWN_SEED = frozenset({"array_record_source.py"})


def _code(path: Path) -> str:
    """The file's code with every string and comment blanked, line structure kept."""
    lines = path.read_text().splitlines(keepends=True)
    blanked = [list(line) for line in lines]
    for token in tokenize.generate_tokens(io.StringIO("".join(lines)).readline):
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        (first, start), (last, end) = token.start, token.end
        for row in range(first - 1, last):
            begin = start if row == first - 1 else 0
            stop = end if row == last - 1 else len(blanked[row])
            for column in range(begin, stop):
                if blanked[row][column] != "\n":
                    blanked[row][column] = " "
    return "".join("".join(line) for line in blanked)


def _literal_seeds(directory: Path) -> list[str]:
    """``file:line`` of every literal seed in the code of the Python files under ``directory``."""
    return [
        f"{path.relative_to(directory)}:{number}"
        for path in sorted(directory.rglob("*.py"))
        if path.name not in _ORDERED_BY_ITS_OWN_SEED
        for number, line in enumerate(_code(path).splitlines(), start=1)
        if _LITERAL_SEED.search(line)
    ]


def test_no_source_holds_a_literal_seed() -> None:
    assert _literal_seeds(_SOURCES) == []


def test_the_excluded_source_still_holds_its_own_seed() -> None:
    """The exclusion is live: once ArrayRecord holds no seed, this fails and the exclusion goes."""
    for name in _ORDERED_BY_ITS_OWN_SEED:
        assert _LITERAL_SEED.search(_code(_SOURCES / name)), name


def test_a_literal_seed_planted_in_a_copy_is_found(tmp_path: Path) -> None:
    copy = tmp_path / "sources"
    shutil.copytree(_SOURCES, copy, ignore=shutil.ignore_patterns("__pycache__"))
    planted = copy / "hf_source.py"
    planted.write_text(planted.read_text() + "\n_ORDER = np.random.default_rng(7)\n")
    lines = len(planted.read_text().splitlines())

    assert _literal_seeds(copy) == [f"hf_source.py:{lines}"]


def test_a_literal_in_a_docstring_or_comment_is_not_a_seed(tmp_path: Path) -> None:
    module = tmp_path / "module.py"
    module.write_text('"""Example: Pipeline(..., rngs=nnx.Rngs(0))."""\n# seed = 3\nx = 1\n')

    assert _literal_seeds(tmp_path) == []
