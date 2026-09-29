"""Every shipped operator's per-record form equals its batch form (design D6).

``apply`` sees one record without a batch axis, so stacking ``apply_record`` over a batch's records
must give what the operator returns for the batch. Keys come from each record's identity in both
forms, so this holds in training mode as well as in eval mode. The cases are the per-record
fixture's table, which names every operator kind datarax ships.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core import batch_ops
from datarax.core.operator import OperatorModule
from datarax.pipeline.dag import name_records
from tests.scripts.script_loader import load_script


generator = load_script("generate_per_record_outputs")

WHOLE_BATCH_CASES = {"batch mix mixup", "batch mix cutmix"}

CASES = [pytest.param(build, data, id=name) for name, build, data in generator.CASES]


def _stacked(operator: OperatorModule, data: dict[str, Any]) -> tuple[Any, Any]:
    batch = name_records(batch_ops.from_arrays(data), generator.INDICES, generator.EPOCH)
    stats = operator.compute_statistics(batch)
    records = [
        operator.apply_record(batch_ops.element(batch, i), stats) for i in range(batch.batch_size)
    ]
    stacked = jax.tree.map(lambda *rows: jnp.stack(rows), *[record.data for record in records])
    return stacked, operator(batch).data


def test_the_whole_batch_cases_are_exactly_the_batch_mixers() -> None:
    whole = {name for name, build, _ in generator.CASES if not build().has_record_form}
    assert whole == WHOLE_BATCH_CASES


@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize(("build", "data"), CASES)
def test_stacking_apply_record_equals_the_batch_form(
    build: Callable[[], OperatorModule], data: dict[str, Any], mode: str
) -> None:
    operator = build()
    if not operator.has_record_form:
        pytest.skip("a whole-batch operator has no per-record form (the batch mixers)")
    getattr(operator, mode)()
    stacked, batched = _stacked(operator, data)
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6), stacked, batched
    )
