"""Probability maths for binary questions. p in (0, 1); 'pts' = percentage points."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median

EPS = 1e-4


def logit(p: float) -> float:
    if not math.isfinite(p):
        raise ValueError(f"probability must be finite, got {p}")
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def pool_logodds_median(ps) -> float:
    """Median of the forecasts in log-odds: robust to one outlying model."""
    xs = [logit(p) for p in ps]
    if not xs:
        raise ValueError("no forecasts to pool")
    return sigmoid(median(xs))


def blend_linear(p_market: float, p_model: float, w_market: float) -> float:
    """Weighted average of probabilities, as in Halawi et al. (2024) and the AIA Forecaster report (2025)."""
    if not 0 <= w_market <= 1:
        raise ValueError("w_market must be in [0, 1]")
    return w_market * p_market + (1 - w_market) * p_model


def liquidity_weights(volumes: dict[str, float | None], lo: float, hi: float) -> dict[str, float]:
    """Market weight per question, linear in log volume between the least (lo) and the most (hi) traded
    question of the run; no volume -> lo."""
    logs = {q: math.log(v) for q, v in volumes.items() if v and v > 0}
    if not logs:
        return dict.fromkeys(volumes, lo)
    a, b = min(logs.values()), max(logs.values())
    return {q: lo if q not in logs else hi if b == a else lo + (hi - lo) * (logs[q] - a) / (b - a) for q in volumes}


def round_half_away(x: float) -> int:
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def mean_price(values) -> float:
    """Market reference: mean of the venues' prices (any number of venues)."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        raise ValueError("no market price available")
    return sum(vals) / len(vals)


@dataclass
class PolicyA:
    market_pts: float
    model_pts: float
    w_market: float
    blend_pts: float
    proposal: int

    def to_dict(self) -> dict:
        return asdict(self)


def policy_a(p_market: float, p_model: float, w_market: float, *, lo: int, hi: int) -> PolicyA:
    """Linear blend -> whole percent in [lo, hi]."""
    b = 100 * blend_linear(p_market, p_model, w_market)
    return PolicyA(100 * p_market, 100 * p_model, w_market, b, min(max(round_half_away(b), lo), hi))


def gain_vs_market(p: float, m: float, outcome: int) -> float:
    """Squared-error gain over the market, x100: positive = the answer beat the market price."""
    return ((m - outcome) ** 2 - (p - outcome) ** 2) * 100
