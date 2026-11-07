"""Scenarios for move questions: every combination of the outcomes of 1-3 binary questions, their probabilities,
and the median of the scenario mixture."""
from __future__ import annotations

from itertools import product

from scipy import optimize, stats

MAX_SCENARIO_QUESTIONS = 3


def scenario_keys(qids: list[str]) -> list[str]:
    """'Q1=YES|Q2=NO' ..., YES first."""
    if not 1 <= len(qids) <= MAX_SCENARIO_QUESTIONS:
        raise ValueError(f"use 1 to {MAX_SCENARIO_QUESTIONS} scenario questions")
    return ["|".join(f"{q}={'YES' if y else 'NO'}" for q, y in zip(qids, combo)) for combo in product((True, False), repeat=len(qids))]


def scenario_probs(p: dict[str, float], coupling: str = "comonotone") -> dict[str, float]:
    """Joint probabilities with the given marginals.
    comonotone: one common latent swing decides every question (the likelier YES happens whenever the less likely
                one does), which suits outcomes driven by the same national mood;
    independent: product of the marginals."""
    qids = list(p)
    if any(not 0 <= v <= 1 for v in p.values()):
        raise ValueError("probabilities must be in [0, 1]")
    out = {}
    for key, combo in zip(scenario_keys(qids), product((True, False), repeat=len(qids))):
        if coupling == "independent":
            pr = 1.0
            for q, y in zip(qids, combo):
                pr *= p[q] if y else 1 - p[q]
        elif coupling == "comonotone":   # YES for q iff U < p[q], U uniform: the scenario is an interval of U
            lo = max([p[q] for q, y in zip(qids, combo) if not y], default=0.0)
            hi = min([p[q] for q, y in zip(qids, combo) if y], default=1.0)
            pr = max(0.0, hi - lo)
        else:
            raise ValueError(f"unknown coupling {coupling}")
        out[key] = pr
    return out


def describe(key: str, labels: dict[str, str]) -> str:
    """'Q1=YES|Q2=NO' -> 'Senate: YES; House: NO' using each question's short label."""
    parts = [x.split("=") for x in key.split("|")]
    return "; ".join(f"{labels.get(q, q)}: {o}" for q, o in parts)


def demean(shifts: dict, probs: dict) -> dict[str, float]:
    """Prices already reflect the expected outcome before it is known: only the surprise moves them."""
    mean = sum(probs[s] * shifts[s] for s in probs)
    return {s: shifts[s] - mean for s in probs}


def noise_scale_units(dof: float) -> float:
    """Student-t scale whose median absolute value is one typical move."""
    return 1 / float(stats.t.ppf(0.75, dof))


def mixture_median(shifts: dict, probs: dict, dof: float) -> float:
    """Median of sum_s p_s * t_dof(shift_s, scale), in typical moves (the best single number for absolute error)."""
    scale, total = noise_scale_units(dof), sum(probs.values())
    if total <= 0:
        raise ValueError("scenario probabilities sum to zero")
    f = lambda x: sum(probs[s] / total * stats.t.cdf((x - shifts[s]) / scale, dof) for s in probs) - 0.5
    return float(optimize.brentq(f, min(shifts.values()) - 50 * scale, max(shifts.values()) + 50 * scale, xtol=1e-10))
