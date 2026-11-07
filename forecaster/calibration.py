"""Statistics for move questions, estimated from the price history (no language model).

    python -m forecaster.calibration --config config.toml     prints every number the move policy uses
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from .prices import daily_moves

SCORE_CAP = 5.0


def score(errors_in_typical_moves) -> float:
    """Mean of min(|error| / typical move, 5): lower is better."""
    return float(np.mean(np.minimum(np.abs(np.asarray(errors_in_typical_moves, dtype=float)), SCORE_CAP)))


def event_units(levels: pd.Series, unit: str, events: list[str], window: int, lag_bdays: int) -> dict[str, float]:
    """Day-after-event move of each past event, in typical moves; each event uses its own typical move
    (median absolute move over `window` days ending `lag_bdays` business days before the event)."""
    mv, out = daily_moves(levels, unit), {}
    for d in events:
        e = pd.Timestamp(d)
        hist, nxt = mv[mv.index <= e - pd.offsets.BDay(lag_bdays)].tail(window), mv[mv.index > e]
        if len(hist) >= window // 2 and len(nxt) and (t := hist.abs().median()) > 0:
            out[d] = float(nxt.iloc[0] / t)
    return out


def loo_scores(units: dict[str, float]) -> dict[str, float]:
    """Leave-one-event-out score of two candidate drifts: zero, and the median of the other events."""
    u = np.array(list(units.values()))
    if len(u) < 2:
        return {"zero": score(u) if len(u) else 0.0, "median": float("inf")}
    return {"zero": score(u), "median": score([u[i] - np.median(np.delete(u, i)) for i in range(len(u))])}


def drift(units: dict[str, float]) -> tuple[float, str, dict[str, float]]:
    """The candidate with the lower out-of-sample score; ties go to zero (the simpler rule)."""
    s = loo_scores(units)
    pick = "median" if s["median"] < s["zero"] else "zero"
    return (float(np.median(list(units.values()))) if pick == "median" else 0.0), pick, s


def student_t_dof(series: dict[str, pd.Series], units: dict[str, str], window: int, end: str | None = None) -> float:
    """Maximum-likelihood degrees of freedom of daily moves divided by their trailing typical move, pooled over
    the instruments (fat tails: about 3-5 for financial data)."""
    zs = []
    for key, unit in units.items():
        mv = daily_moves(series[key], unit)
        z = (mv / mv.abs().rolling(window).median().shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
        zs.append((z[z.index <= pd.Timestamp(end)] if end else z).to_numpy())
    data = np.concatenate(zs) if zs else np.array([])
    if len(data) < 100:
        return 4.0   # too little data to fit: a common default for daily returns
    return float(stats.t.fit(data, floc=0)[0])


def move_table(series: dict[str, pd.Series], moves, events: list[str], window: int, lag_bdays: int) -> dict[str, dict]:
    """Per move question: past event moves (typical units), chosen drift, its out-of-sample scores and the largest
    past event move (used as the bound for scenario moves; 3 typical moves if there is no history)."""
    out = {}
    for q in moves:
        units = event_units(series[q.instrument], q.unit, events, window, lag_bdays)
        d, pick, s = drift(units) if units else (0.0, "zero", {})
        out[q.id] = {"units": {k: round(v, 3) for k, v in units.items()}, "drift_u": d, "drift_rule": pick,
                     "loo": {k: round(v, 3) for k, v in s.items()},
                     "max_abs_u": float(max(map(abs, units.values()))) if units else 3.0}
    return out


def main() -> None:
    from . import prices, questions, settings
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    s = settings.load(ap.parse_args().config)
    qs, series = questions.load(s.path("questions")), prices.load_series(s.path("price_history"))
    m = s.moves
    for qid, row in move_table(series, qs.moves, m.past_events, m.typical_window, m.event_window_lag_bdays).items():
        print(qid, row)
    print("Student-t degrees of freedom:",
          round(student_t_dof(series, {q.instrument: q.unit for q in qs.moves}, m.typical_window, m.typical_end or None), 3))


if __name__ == "__main__":
    main()
