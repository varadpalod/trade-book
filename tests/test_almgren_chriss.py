import pytest

from src.execution.almgren_chriss import AlmgrenChrissStrategy


@pytest.fixture
def strategy():
    return AlmgrenChrissStrategy()


def test_schedule_length(strategy):
    schedule = strategy.generate_schedule(1000, 5)

    assert len(schedule) == 5


def test_total_shares_preserved(strategy):
    schedule = strategy.generate_schedule(1000, 5)

    assert sum(schedule) == 1000


def test_zero_shares(strategy):
    assert strategy.generate_schedule(0, 5) == []


def test_negative_shares(strategy):
    with pytest.raises(ValueError):
        strategy.generate_schedule(-100, 5)


def test_zero_intervals(strategy):
    with pytest.raises(ValueError):
        strategy.generate_schedule(1000, 0)


def test_high_lambda_front_loads():
    strategy = AlmgrenChrissStrategy(lambda_risk=1.0)

    schedule = strategy.generate_schedule(1000, 5)

    assert schedule[0] > schedule[-1]


def test_low_lambda_near_twap():
    strategy = AlmgrenChrissStrategy(lambda_risk=0.0)

    schedule = strategy.generate_schedule(1000, 5)

    assert schedule == [200, 200, 200, 200, 200]