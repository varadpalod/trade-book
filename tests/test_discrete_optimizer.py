# tests/test_discrete_optimizer.py

import pytest

from src.execution.discrete_optimizer import DiscreteOptimizer


@pytest.fixture
def optimizer():
    return DiscreteOptimizer()


DEFAULT_PARAMS = {
    "execution_horizon": 60,
    "intervals": 5,
    "sigma": 0.95,
    "eta": 2.5e-6,
    "gamma": 2.5e-7,
}


# -------------------------------------------------
# Input validation
# -------------------------------------------------

def test_negative_total_shares(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=-100,
            lambda_risk=1e-6,
            **DEFAULT_PARAMS
        )


def test_zero_execution_horizon(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=0,
            intervals=5,
            sigma=0.95,
            eta=2.5e-6,
            gamma=2.5e-7,
            lambda_risk=1e-6,
        )


def test_zero_intervals(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=60,
            intervals=0,
            sigma=0.95,
            eta=2.5e-6,
            gamma=2.5e-7,
            lambda_risk=1e-6,
        )


def test_negative_sigma(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=60,
            intervals=5,
            sigma=-1,
            eta=2.5e-6,
            gamma=2.5e-7,
            lambda_risk=1e-6,
        )


def test_negative_eta(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=60,
            intervals=5,
            sigma=0.95,
            eta=-1,
            gamma=2.5e-7,
            lambda_risk=1e-6,
        )


def test_negative_gamma(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=60,
            intervals=5,
            sigma=0.95,
            eta=2.5e-6,
            gamma=-1,
            lambda_risk=1e-6,
        )


def test_negative_lambda(optimizer):
    with pytest.raises(ValueError):
        optimizer.optimize(
            total_shares=1000,
            execution_horizon=60,
            intervals=5,
            sigma=0.95,
            eta=2.5e-6,
            gamma=2.5e-7,
            lambda_risk=-1,
        )


# -------------------------------------------------
# Basic optimizer tests
# -------------------------------------------------

def test_total_shares_preserved(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert sum(schedule) == 1000


def test_schedule_length(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert len(schedule) == 5


def test_schedule_is_integer(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert all(isinstance(x, int) for x in schedule)


def test_non_negative_trades(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert all(x >= 0 for x in schedule)


def test_zero_shares(optimizer):
    schedule = optimizer.optimize(
        total_shares=0,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert schedule == [0, 0, 0, 0, 0]


# -------------------------------------------------
# Financial behaviour
# -------------------------------------------------

def test_zero_lambda_is_uniform(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=0,
        **DEFAULT_PARAMS
    )

    assert schedule == [200, 200, 200, 200, 200]


def test_front_loading_with_higher_lambda(optimizer):
    low = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-8,
        **DEFAULT_PARAMS
    )

    high = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-5,
        **DEFAULT_PARAMS
    )

    assert high[0] > low[0]


def test_first_trade_monotonic(optimizer):
    lambdas = [0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5]

    first = [
        optimizer.optimize(
            total_shares=1000,
            lambda_risk=l,
            **DEFAULT_PARAMS
        )[0]
        for l in lambdas
    ]

    assert first == sorted(first)


def test_last_trade_monotonic(optimizer):
    lambdas = [0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5]

    last = [
        optimizer.optimize(
            total_shares=1000,
            lambda_risk=l,
            **DEFAULT_PARAMS
        )[-1]
        for l in lambdas
    ]

    assert last == sorted(last, reverse=True)


def test_inventory_decreases(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    inventory = 1000
    inventories = []

    for trade in schedule:
        inventory -= trade
        inventories.append(inventory)

    assert inventories == sorted(inventories, reverse=True)
    assert inventory == 0


# -------------------------------------------------
# Numerical sanity
# -------------------------------------------------

def test_more_intervals(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        execution_horizon=60,
        intervals=20,
        sigma=0.95,
        eta=2.5e-6,
        gamma=2.5e-7,
        lambda_risk=1e-6,
    )

    assert len(schedule) == 20
    assert sum(schedule) == 1000


def test_different_share_count(optimizer):
    schedule = optimizer.optimize(
        total_shares=537,
        lambda_risk=1e-6,
        **DEFAULT_PARAMS
    )

    assert sum(schedule) == 537


def test_large_lambda_frontloads(optimizer):
    schedule = optimizer.optimize(
        total_shares=1000,
        lambda_risk=1e-5,
        **DEFAULT_PARAMS
    )

    assert schedule[0] > schedule[-1]