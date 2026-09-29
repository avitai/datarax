"""Unit tests for sequential strategy."""

from typing import Any
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule
from datarax.operators.strategies.sequential import (
    ConditionalSequentialStrategy,
    SequentialStrategy,
)
from tests.test_common.mock_operators import MultiplierMockOperator as MockOperator


class AddOperator(OperatorModule):
    """Adds a constant to the data."""

    def __init__(self, value: float) -> None:
        super().__init__(OperatorConfig(stochastic=False))
        self.value = value

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        del key, stats
        return element.replace(data=element.data + self.value)


class TestSequentialStrategy:
    def test_apply_chaining(self) -> None:
        element = Element(jnp.array([1.0, 2.0]), state={"count": 0})

        result = SequentialStrategy().apply([MockOperator(2.0), MockOperator(3.0)], element, None)

        assert jnp.array_equal(result.data, jnp.array([6.0, 12.0]))
        assert result.state["count"] == 2

    def test_operators_apply_in_order(self) -> None:
        operators = [AddOperator(1.0), MockOperator(10.0)]

        result = SequentialStrategy().apply(operators, Element(jnp.array([1.0])), None)

        assert result.data[0] == 20.0  # (1 + 1) * 10, not 1 * 10 + 1

    def test_apply_empty_list(self) -> None:
        element = Element(jnp.array([1.0]))
        assert SequentialStrategy().apply([], element, None) is element

    def test_each_child_gets_the_record_and_its_own_statistics(self) -> None:
        child = MagicMock(spec=OperatorModule)
        element = Element(jnp.array([1.0]))
        child.apply_record.return_value = element

        SequentialStrategy().apply([child], element, {"children": ({"mean": 1.0},)})

        child.apply_record.assert_called_once_with(element, {"mean": 1.0})


class TestConditionalSequentialStrategy:
    def test_each_condition_reads_the_record_as_it_reaches_its_operator(self) -> None:
        # Input 1.0 -> +10 -> 11.0 -> (11 > 5) -> +100 -> 111.0
        # Input -20.0 -> +10 -> -10.0 -> (-10 > 5 is False) -> -10.0
        strategy = ConditionalSequentialStrategy(
            [lambda _x: jnp.array(True), lambda x: jnp.sum(x) > 5.0]
        )
        operators = [AddOperator(10.0), AddOperator(100.0)]

        assert strategy.apply(operators, Element(jnp.array([1.0])), None).data[0] == 111.0
        assert strategy.apply(operators, Element(jnp.array([-20.0])), None).data[0] == -10.0
