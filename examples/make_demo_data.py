"""Create the FICTIONAL demo data in examples/demo/ (synthetic prices and invented market quotes).
Nothing here describes a real election, market or instrument; it only lets `python run.py --dry` run end to end.

    python examples/make_demo_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "demo"
EVENTS = ["2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05"]


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", "2026-11-03")
    n = len(days)
    eq = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_t(4, n) / np.sqrt(2)))     # equity index level
    vol = np.empty(n)
    vol[0] = 18.0
    for i in range(1, n):                                                                # mean-reverting volatility index
        vol[i] = max(9.0, vol[i - 1] + 0.08 * (18 - vol[i - 1]) + 1.1 * rng.standard_t(4) / np.sqrt(2))
    yld = 3 + np.cumsum(0.05 * rng.standard_t(4, n) / np.sqrt(2))                       # 10-year yield, percent
    yld = np.clip(yld, 0.2, 8)
    for e in EVENTS:   # a small post-event pattern so the drift rule has something to find
        i = days.searchsorted(pd.Timestamp(e), side="right")
        if i < n:
            eq[i:] *= 1.004
            vol[i:] -= np.linspace(1.5, 0, n - i).clip(0, 1.5)
    rows = [pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "instrument": k, "close": np.round(v, 4)})
            for k, v in (("DEMO_EQUITY", eq), ("DEMO_VOL", vol), ("DEMO_10Y", yld))]
    pd.concat(rows).to_csv(OUT / "prices_daily.csv", index=False)

    quotes = {  # question: (venue_a bid, venue_a ask, venue_a volume, venue_b price, venue_b volume)
        "LOWER": (0.62, 0.64, 900000, 0.635, 1500000), "UPPER": (0.44, 0.46, 400000, 0.45, 700000),
        "D1": (0.70, 0.74, 20000, 0.71, 15000), "D2": (0.35, 0.39, 8000, 0.37, 5000)}
    t1, t0 = pd.Timestamp("2026-11-03 20:00", tz="UTC"), pd.Timestamp("2026-10-27 20:00", tz="UTC")
    mrows = []
    for q, (bid, ask, va, pb, vb) in quotes.items():
        for t, shift in ((t0, -0.03), (t1, 0.0)):
            mrows += [{"question_id": q, "venue": "venue_a", "time_utc": t.isoformat(), "price": "", "bid": round(bid + shift, 3),
                       "ask": round(ask + shift, 3), "volume": va}, {"question_id": q, "venue": "venue_b", "time_utc": t.isoformat(),
                       "price": round(pb + shift, 3), "bid": "", "ask": "", "volume": vb}]
    pd.DataFrame(mrows).to_csv(OUT / "market_prices.csv", index=False)
    print(f"wrote {OUT / 'prices_daily.csv'} and {OUT / 'market_prices.csv'}")


if __name__ == "__main__":
    main()
