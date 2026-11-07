import pytest

from forecaster.mathx import (blend_linear, gain_vs_market, liquidity_weights, logit, mean_price, policy_a,
                              pool_logodds_median, round_half_away, sigmoid)


def test_logit_sigmoid_roundtrip():
    for p in (0.01, 0.3, 0.5, 0.77, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p)
    assert sigmoid(-800) >= 0 and sigmoid(800) <= 1
    with pytest.raises(ValueError):
        logit(float("nan"))


def test_pool_median_ignores_outlier():
    assert pool_logodds_median([0.6, 0.62, 0.01]) == pytest.approx(0.6)
    assert pool_logodds_median([0.5]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pool_logodds_median([])


def test_blend_and_policy():
    assert blend_linear(0.7, 0.5, 0.8) == pytest.approx(0.66)
    pol = policy_a(0.70, 0.50, 0.8, lo=1, hi=99)
    assert pol.proposal == 66 and pol.blend_pts == pytest.approx(66.0)
    assert policy_a(0.999, 0.999, 1.0, lo=1, hi=99).proposal == 99
    with pytest.raises(ValueError):
        blend_linear(0.5, 0.5, 1.2)


def test_liquidity_weights():
    w = liquidity_weights({"a": 10, "b": 1000, "c": 100, "d": None}, 0.67, 0.80)
    assert w["a"] == pytest.approx(0.67) and w["b"] == pytest.approx(0.80)
    assert w["c"] == pytest.approx(0.735) and w["d"] == 0.67
    assert liquidity_weights({"a": 5, "b": 5}, 0.6, 0.9) == {"a": 0.9, "b": 0.9}
    assert liquidity_weights({"a": None}, 0.6, 0.9) == {"a": 0.6}


def test_rounding_mean_gain():
    assert round_half_away(2.5) == 3 and round_half_away(-2.5) == -3
    assert mean_price([0.6, None, 0.7]) == pytest.approx(0.65)
    with pytest.raises(ValueError):
        mean_price([None])
    assert gain_vs_market(0.8, 0.7, 1) == pytest.approx(5.0)
    assert gain_vs_market(0.7, 0.7, 0) == 0
