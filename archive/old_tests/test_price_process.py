import pytest

from src.execution.price_process import PriceProcess


def test_returns_intervals_plus_one_prices():
    process = PriceProcess(random_seed=42)

    prices = process.generate(
        initial_price=100.0,
        sigma=1.0,
        intervals=5,
        tau=1.0,
    )

    assert len(prices) == 6


def test_first_price_is_initial_price():
    process = PriceProcess(random_seed=42)

    prices = process.generate(
        initial_price=100.0,
        sigma=1.0,
        intervals=5,
        tau=1.0,
    )

    assert prices[0] == 100.0


def test_same_seed_produces_same_path():
    p1 = PriceProcess(random_seed=42)
    p2 = PriceProcess(random_seed=42)

    path1 = p1.generate(100.0, 1.0, 10, 1.0)
    path2 = p2.generate(100.0, 1.0, 10, 1.0)

    assert path1 == path2


def test_different_seed_produces_different_path():
    p1 = PriceProcess(random_seed=42)
    p2 = PriceProcess(random_seed=7)

    path1 = p1.generate(100.0, 1.0, 10, 1.0)
    path2 = p2.generate(100.0, 1.0, 10, 1.0)

    assert path1 != path2


def test_negative_sigma():
    process = PriceProcess()

    with pytest.raises(ValueError):
        process.generate(100.0, -1.0, 5, 1.0)


def test_zero_intervals():
    process = PriceProcess()

    with pytest.raises(ValueError):
        process.generate(100.0, 1.0, 0, 1.0)


def test_negative_tau():
    process = PriceProcess()

    with pytest.raises(ValueError):
        process.generate(100.0, 1.0, 5, -1.0)


def test_invalid_initial_price():
    process = PriceProcess()

    with pytest.raises(ValueError):
        process.generate(0.0, 1.0, 5, 1.0)