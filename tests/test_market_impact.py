import pytest

from src.execution.market_impact import MarketImpact


@pytest.fixture
def impact():
    return MarketImpact()


def test_permanent_impact(impact):
    assert impact.permanent_impact(
        trade_size=100,
        gamma=0.1,
    ) == pytest.approx(10.0)


def test_temporary_impact(impact):
    assert impact.temporary_impact(
        trade_size=100,
        eta=0.2,
    ) == pytest.approx(20.0)


def test_zero_trade_size_permanent(impact):
    assert impact.permanent_impact(
        trade_size=0,
        gamma=0.5,
    ) == 0.0


def test_zero_trade_size_temporary(impact):
    assert impact.temporary_impact(
        trade_size=0,
        eta=0.5,
    ) == 0.0


def test_negative_gamma_raises(impact):
    with pytest.raises(ValueError):
        impact.permanent_impact(
            trade_size=100,
            gamma=-0.1,
        )


def test_negative_eta_raises(impact):
    with pytest.raises(ValueError):
        impact.temporary_impact(
            trade_size=100,
            eta=-0.2,
        )


def test_permanent_impact_linear(impact):
    assert impact.permanent_impact(
        trade_size=200,
        gamma=0.05,
    ) == pytest.approx(10.0)


def test_temporary_impact_linear(impact):
    assert impact.temporary_impact(
        trade_size=200,
        eta=0.05,
    ) == pytest.approx(10.0)