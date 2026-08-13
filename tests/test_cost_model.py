import pytest

from src.execution.cost_model import ExecutionCostModel


def test_constructor_stores_parameters():
    model = ExecutionCostModel(
        sigma=0.2,
        eta=0.001,
        gamma=0.0005,
    )

    assert model.sigma == 0.2
    assert model.eta == 0.001
    assert model.gamma == 0.0005


def test_negative_parameters_raise_error():
    with pytest.raises(ValueError):
        ExecutionCostModel(-0.2, 0.001, 0.0005)

    with pytest.raises(ValueError):
        ExecutionCostModel(0.2, -0.001, 0.0005)

    with pytest.raises(ValueError):
        ExecutionCostModel(0.2, 0.001, -0.0005)


def test_temporary_impact():
    model = ExecutionCostModel(
        sigma=0.2,
        eta=0.001,
        gamma=0.0005,
    )

    assert model.temporary_impact(100) == 10.0


def test_negative_quantity_raises_error():
    model = ExecutionCostModel(
        sigma=0.2,
        eta=0.001,
        gamma=0.0005,
    )

    with pytest.raises(ValueError):
        model.temporary_impact(-10)