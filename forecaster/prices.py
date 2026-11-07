"""Daily closing levels for move questions.

CSV columns: date (YYYY-MM-DD), instrument, close. Units of the move (questions.py):
  pct    -> percentage change of the close
  points -> difference of the close (for example a volatility index)
  bp     -> difference x 100, for yields quoted in percent (basis points)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_series(path: str | Path) -> dict[str, pd.Series]:
    df = pd.read_csv(path, parse_dates=["date"])
    if miss := {"date", "instrument", "close"} - set(df.columns):
        raise ValueError(f"{path}: missing column(s) {sorted(miss)}")
    return {str(k): g.set_index("date")["close"].astype(float).dropna().sort_index()
            for k, g in df.groupby("instrument")}


def daily_moves(levels: pd.Series, unit: str) -> pd.Series:
    lv = levels.dropna()
    if unit == "bp":
        return (lv.diff() * 100).dropna()
    if unit == "points":
        return lv.diff().dropna()
    if unit == "pct":
        return (lv.pct_change() * 100).dropna()
    raise ValueError(f"unknown unit {unit}")


def typical_move(levels: pd.Series, unit: str, window: int, end: str | None = None) -> float:
    """Median absolute daily move over the `window` trading days up to `end` (default: the last date)."""
    mv = daily_moves(levels, unit)
    if end:
        mv = mv[mv.index <= pd.Timestamp(end)]
    mv = mv.tail(window)
    if len(mv) < max(2, window // 2):
        raise ValueError("not enough data for the typical move")
    t = float(mv.abs().median())
    if t <= 0:
        raise ValueError("typical move is zero")
    return t


def recent_context(levels: pd.Series, unit: str, n: int = 10) -> dict:
    lv, mv = levels.dropna(), daily_moves(levels, unit)
    return {"last_date": str(lv.index[-1].date()), "last_level": round(float(lv.iloc[-1]), 4),
            "recent_levels": {str(d.date()): round(float(x), 4) for d, x in lv.tail(n).items()},
            "recent_moves": {str(d.date()): round(float(x), 3) for d, x in mv.tail(n).items()},
            "median_abs_move_20d": round(float(mv.tail(20).abs().median()), 3)}


def event_day_moves(levels: pd.Series, unit: str, events: list[str]) -> dict[str, float]:
    """Move on the first trading day after each event date (the event is assumed to resolve overnight)."""
    mv = daily_moves(levels, unit)
    return {d: round(float(mv[mv.index > pd.Timestamp(d)].iloc[0]), 3) for d in events if (mv.index > pd.Timestamp(d)).any()}
