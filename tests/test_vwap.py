import pytest

from src.execution.vwap import VWAPStrategy


@pytest.fixture
def strategy():
    return VWAPStrategy()


def test_basic_vwap(strategy):
    volume_profile = [0.1, 0.2, 0.4, 0.3]

    schedule = strategy.generate_schedule(
        total_shares=1000,
        intervals=4,
        market_data=volume_profile,
    )

    assert schedule == [100, 200, 400, 300]


def test_total_shares_preserved(strategy):
    volume_profile = [0.1, 0.2, 0.4, 0.3]

    schedule = strategy.generate_schedule(
        1234,
        4,
        volume_profile,
    )

    assert sum(schedule) == 1234


def test_invalid_profile_sum(strategy):
    profile = [0.2, 0.2, 0.2, 0.2]

    with pytest.raises(ValueError):
        strategy.generate_schedule(
            1000,
            4,
            profile,
        )


def test_negative_weight(strategy):
    profile = [0.2, -0.1, 0.5, 0.4]

    with pytest.raises(ValueError):
        strategy.generate_schedule(
            1000,
            4,
            profile,
        )


def test_wrong_length(strategy):
    profile = [0.5, 0.5]

    with pytest.raises(ValueError):
        strategy.generate_schedule(
            1000,
            4,
            profile,
        )


def test_zero_shares(strategy):
    profile = [0.25, 0.25, 0.25, 0.25]

    assert strategy.generate_schedule(
        0,
        4,
        profile,
    ) == []