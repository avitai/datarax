"""Five Hub datasets across modalities load through ``HFStreamingSource`` with the fields we expect.

An integration test (``--integration``): it streams the first records of each dataset as one
batch, so it downloads no split. It is skipped only when the Hub cannot be reached or refuses for
load (a connection error, 429 or a 5xx); any other failure, a 404 for a path the source asked
for included, fails it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError

from datarax.sources.hf_source import HFStreamingConfig, HFStreamingSource


pytestmark = pytest.mark.integration

RECORDS = 4


def image(height: int, width: int) -> Callable[[Any], bool]:
    return lambda value: np.asarray(value).shape[:2] == (height, width)


def integer(value: Any) -> bool:
    return np.issubdtype(np.asarray(value).dtype, np.integer)


# name, the fields each record must carry and a check for each, fields excluded (strings)
CASES: list[tuple[str, dict[str, Callable[[Any], bool]], set[str]]] = [
    ("ylecun/mnist", {"image": image(28, 28), "label": integer}, set()),
    ("uoft-cs/cifar10", {"img": image(32, 32), "label": integer}, set()),
    ("stanfordnlp/imdb", {"label": integer}, {"text"}),
    ("fancyzhx/ag_news", {"label": integer}, {"text"}),
    ("cornell-movie-review-data/rotten_tomatoes", {"label": integer}, {"text"}),
]


def unreachable(error: BaseException) -> bool:
    """Whether ``error`` means the Hub could not be reached or refused for load."""
    if isinstance(error, requests.ConnectionError | requests.Timeout):
        return True
    if isinstance(error, HfHubHTTPError) and error.response is not None:
        return error.response.status_code == 429 or error.response.status_code >= 500
    return False


@pytest.mark.parametrize(("name", "fields", "excluded"), CASES, ids=[case[0] for case in CASES])
def test_a_hub_dataset_streams_its_expected_fields(
    name: str, fields: dict[str, Callable[[Any], bool]], excluded: set[str]
) -> None:
    config = HFStreamingConfig(name=name, split="train", exclude_keys=excluded)
    try:
        batch = HFStreamingSource(config).get_batch(RECORDS)
    except Exception as error:  # noqa: BLE001 - classified: skip only an unreachable Hub
        if unreachable(error) or unreachable(error.__cause__ or error):
            pytest.skip(f"Hugging Face Hub unavailable: {error}")
        raise

    assert batch.batch_size == RECORDS
    assert not excluded & set(batch.data)
    for field, check in fields.items():
        for row in np.asarray(batch[field]):
            assert check(row), (name, field, row.shape)
