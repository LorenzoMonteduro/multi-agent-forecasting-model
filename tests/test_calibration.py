import numpy as np
import pandas as pd
import pytest

from forecaster.calibration import drift, event_units, loo_scores, score, student_t_dof
from forecaster.prices import daily_moves, event_day_moves, typical_move


def series(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_daily_moves_units():
    s = series([100, 101, 99.99])
    assert daily_moves(s, "pct").round(4).tolist() == [1.0, -1.0]
    assert daily_moves(s, "points").round(4).tolist() == [1.0, -1.01]
    y = series([4.00, 4.05])
    assert daily_moves(y, "bp").round(6).tolist() == [5.0]
    with pytest.raises(ValueError):
        daily_moves(s, "x")


def test_typical_move_and_event_moves():
    s = series(100 + np.arange(80) * 0.5)
    assert typical_move(s, "points", 60) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        typical_move(series([1, 2]), "points", 60)
    d = str(s.index[10].date())
    assert event_day_moves(s, "points", [d, "2099-01-01"]) == {d: 0.5}


def test_score_and_drift():
    assert score([0.5, -7]) == pytest.approx(2.75)
    units = {"a": 2.0, "b": 2.2, "c": 1.8, "d": 2.1}
    assert drift(units)[1] == "median" and drift(units)[0] == pytest.approx(2.05)
    noisy = {"a": 2.0, "b": -2.0, "c": 1.5, "d": -1.5}
    assert drift(noisy) == (0.0, "zero", loo_scores(noisy))
    assert loo_scores({"a": 1.0})["median"] == float("inf")


def test_event_units_uses_own_typical_move():
    vals = np.full(200, 100.0) + np.tile([0.0, 1.0], 100)   # moves alternate +1 / -1: typical move 1
    s = series(vals)
    s.iloc[150:] += 3                                          # a +3 jump on day 150
    ev = str(s.index[149].date())
    u = event_units(s, "points", [ev], 60, 2)
    assert u[ev] == pytest.approx(3 + (1 if vals[150] > vals[149] else -1))


def test_student_t_dof_on_t_data():
    rng = np.random.default_rng(0)
    s = series(np.cumsum(rng.standard_t(4, 3000)) + 1000)
    assert 3 < student_t_dof({"x": s}, {"x": "points"}, 60) < 6
    assert student_t_dof({}, {}, 60) == 4.0
