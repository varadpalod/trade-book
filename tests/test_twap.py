import pytest

from src.execution.twap import TWAPStrategy


@pytest.fixture
def strategy():
    """Create a TWAPStrategy instance for tests."""
    return TWAPStrategy()


def test_equal_distribution(strategy):
    """1000 shares over 10 intervals."""
    schedule = strategy.generate_schedule(1000, 10)

    assert schedule == [100] * 10


def test_remainder_distribution(strategy):
    """1003 shares over 10 intervals."""
    schedule = strategy.generate_schedule(1003, 10)

    assert schedule == [
        101, 101, 101,
        100, 100, 100, 100, 100, 100, 100
    ]


def test_more_intervals_than_shares(strategy):
    """5 shares over 10 intervals."""
    schedule = strategy.generate_schedule(5, 10)

    assert schedule == [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    ]


def test_zero_shares(strategy):
    """Zero shares should return an empty schedule."""
    assert strategy.generate_schedule(0, 10) == []


def test_negative_shares(strategy):
    """Negative shares should raise ValueError."""
    with pytest.raises(ValueError):
        strategy.generate_schedule(-1, 10)


def test_zero_intervals(strategy):
    """Zero intervals should raise ValueError."""
    with pytest.raises(ValueError):
        strategy.generate_schedule(100, 0)

def test_total_shares_preserved(strategy):
    """The schedule must allocate every share exactly once."""
    total_shares = 1234
    intervals = 17

    schedule = strategy.generate_schedule(total_shares, intervals)

    assert sum(schedule) == total_shares
    assert len(schedule) == intervals