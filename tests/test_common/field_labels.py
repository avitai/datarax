"""How a test finds a field's label in a refusal under the one display rule.

A refusal names a field by its spelling in quotes (``'x'``), never by jax ``keystr``
(``['x']``). A substring check for ``'x'`` would also match ``['x']``, so it could not tell the
two displays apart.
"""

from __future__ import annotations

import re


def unbracketed(spelling: str) -> str:
    """A regular expression for ``'spelling'`` as a label, not inside ``[...]``.

    Args:
        spelling: The field's spelling.

    Returns:
        The pattern, for ``re.search`` or ``pytest.raises(match=...)``.
    """
    return rf"(?<!\[)'{re.escape(spelling)}'(?!\])"
