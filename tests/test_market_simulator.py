import pytest

from src.execution.execution_result import ExecutionResult
from src.execution.market_simulator import MarketSimulator


DEFAULT_PARAMS = {
    "initial_price": 100.0,
    "sigma": 1.0,
    "eta": 2.5e-6,
    "gamma": 2.5e-7,
    "execution_horizon": 60.0,
}


@pytest.fixture
def simulator():
    return MarketSimulator(random_seed=42)


@pytest.fixture
def schedule():
    return [200, 200, 200, 200, 200]


def test_returns_execution_result(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert isinstance(result, ExecutionResult)


def test_market_prices_length(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert len(result.market_prices) == len(schedule) + 1


def test_execution_prices_length(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert len(result.execution_prices) == len(schedule)


def test_inventory_length(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert len(result.inventory) == len(schedule) + 1


def test_inventory_starts_with_total_shares(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert result.inventory[0] == sum(schedule)


def test_inventory_ends_at_zero(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert result.inventory[-1] == 0


def test_cash_spent_positive(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert result.cash_spent > 0


def test_implementation_shortfall_is_float(simulator, schedule):
    result = simulator.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert isinstance(result.implementation_shortfall, float)


def test_same_seed_produces_same_results(schedule):
    sim1 = MarketSimulator(random_seed=42)
    sim2 = MarketSimulator(random_seed=42)

    result1 = sim1.execute(schedule=schedule, **DEFAULT_PARAMS)
    result2 = sim2.execute(schedule=schedule, **DEFAULT_PARAMS)

    assert result1.market_prices == result2.market_prices
    assert result1.execution_prices == result2.execution_prices
    assert result1.cash_spent == pytest.approx(result2.cash_spent)
    assert result1.implementation_shortfall == pytest.approx(
        result2.implementation_shortfall
    )


def test_empty_schedule_raises(simulator):
    with pytest.raises(ValueError):
        simulator.execute(
            schedule=[],
            **DEFAULT_PARAMS,
        )