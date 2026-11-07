"""Prediction-market prices for binary questions.

The default source reads a CSV (columns below). To use live data, write a class with the same `snapshot(qid)`
method and pass it to `run.py` (see README, "Plugging in live market data").

CSV columns (one row per quote; several rows per question and venue are fine, the latest is used):
  question_id, venue, time_utc (ISO 8601), price  [, bid, ask, volume]
  - price, bid, ask: probability of YES in [0, 1]; if bid and ask are both given, their midpoint is used
  - volume: traded volume over a recent window, same unit for every question (only ratios matter)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import pandas as pd

from .mathx import mean_price


@dataclass
class Venue:
    venue: str
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    time_utc: str | None = None


@dataclass
class MarketSnapshot:
    qid: str
    venues: list[Venue]
    market: float                        # mean of the venues' prices: the benchmark
    volume: float | None                 # summed over venues (None if no venue reports it)
    time_utc: str | None                 # oldest quote used
    change_7d_pts: float | None = None
    stale: bool = False
    stale_reason: str = ""
    w_market: float | None = None        # set by run.py from the volumes of all binary questions
    source: str = "csv"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class MarketSource(Protocol):
    def snapshot(self, qid: str) -> MarketSnapshot: ...


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def venue_price(price, bid, ask) -> float | None:
    b, a = _num(bid), _num(ask)
    return (b + a) / 2 if b is not None and a is not None else _num(price)


class CsvMarketSource:
    def __init__(self, path: str | Path, max_age_h: float | None = None, now: pd.Timestamp | None = None):
        df = pd.read_csv(path)
        need = {"question_id", "venue", "time_utc"}
        if miss := need - set(df.columns):
            raise ValueError(f"{path}: missing column(s) {sorted(miss)}")
        if "price" not in df.columns and not {"bid", "ask"} <= set(df.columns):
            raise ValueError(f"{path}: need a price column or both bid and ask")
        for c in ("price", "bid", "ask", "volume"):
            if c not in df.columns:
                df[c] = None
        df["time"] = pd.to_datetime(df["time_utc"], utc=True)
        self.df, self.max_age_h = df.sort_values("time"), max_age_h
        self.now = now if now is not None else pd.Timestamp.now(tz="UTC")

    def snapshot(self, qid: str) -> MarketSnapshot:
        d = self.df[self.df.question_id.astype(str) == qid]
        if d.empty:
            raise RuntimeError(f"no market price for {qid}")
        venues, old = [], []
        for name, g in d.groupby("venue", sort=True):
            last = g.iloc[-1]
            p = venue_price(last.price, last.bid, last.ask)
            if p is None:
                continue
            venues.append(Venue(str(name), p, _num(last.bid), _num(last.ask), _num(last.volume),
                                last.time.strftime("%Y-%m-%d %H:%M")))
            week = g[g.time >= last.time - pd.Timedelta(days=7)].iloc[0]
            old.append(venue_price(week.price, week.bid, week.ask))
        if not venues:
            raise RuntimeError(f"no usable market price for {qid}")
        m = mean_price(v.price for v in venues)
        vols = [v.volume for v in venues if v.volume is not None]
        old_vals = [x for x in old if x is not None]
        snap = MarketSnapshot(qid, venues, m, sum(vols) if vols else None, min(v.time_utc for v in venues),
                              100 * (m - mean_price(old_vals)) if old_vals else None)
        age_h = (self.now - pd.Timestamp(snap.time_utc, tz="UTC")).total_seconds() / 3600
        if self.max_age_h is not None and age_h > self.max_age_h:
            snap.stale, snap.stale_reason = True, f"oldest quote {age_h:.1f} h old"
        return snap
