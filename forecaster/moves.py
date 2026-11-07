"""Move questions: research -> scenario moves from several models -> demeaned Student-t mixture median ->
drift (estimated from past events) + scenario shift -> cross-instrument check (logged) -> written reason.

forecast = typical move x (drift + scenario shift), all in typical-move units, rounded to the question's decimals."""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from . import prompts
from .binary import clip, explain
from .jsonparse import last_json_object, to_number
from .llm import BudgetSkip, Caller
from .questions import Move, QuestionSet
from .scenarios import demean, describe, mixture_median


def clamp(x: float, cap: float) -> float:
    return max(-cap, min(cap, x))


def round_unit(x: float, decimals: int) -> float:
    return round(x, decimals) or 0.0   # no "-0.0"


@dataclass
class PolicyMove:
    qid: str
    unit: str
    typical: float
    drift_u: float
    scenario_u: float
    total_u: float
    proposal: float

    def to_dict(self) -> dict:
        return asdict(self)


def policy_move(q: Move, typical: float, drift_u: float, scenario_u: float) -> PolicyMove:
    if not typical > 0 or not math.isfinite(typical):
        raise ValueError("typical move must be positive")
    total = drift_u + scenario_u
    return PolicyMove(q.id, q.unit_label, typical, drift_u, scenario_u, total,
                      round_unit(clamp(total * typical, q.limit), q.decimals))


def parse_moves(text: str, keys: list[str]) -> dict | None:
    sm = (last_json_object(text) or {}).get("scenario_moves")
    if not isinstance(sm, dict):
        return None
    out = {k: to_number(sm.get(k)) for k in keys}
    return out if None not in out.values() else None


def fake_moves(qid: str, name: str, keys: list[str]) -> str:
    g = 1 if sum(map(ord, qid + name)) % 2 else -1
    return "[dry]\n" + json.dumps({"scenario_moves": {k: round(g * (0.6 - 0.4 * i), 2) for i, k in enumerate(keys)},
                                   "reason": "dry"})


def base_state(q: Move, info: dict, probs: dict) -> dict:
    return {"qid": q.id, "kind": "move", "label": q.label, "unit": q.unit_label, "decimals": q.decimals, **info,
            "probs": probs, "steps": [], "forecasts": [],
            "aggregate": {"mean_moves": None, "demeaned": None, "scenario_u": 0.0, "n": 0}}


def prepare(caller: Caller, q: Move, qs: QuestionSet, info: dict, probs: dict, dof: float, briefing: str, now: str) -> dict:
    s, M = caller.s, caller.s.models
    base = {"context": s.method.context, "now": now}
    labels = {b.id: b.short or b.id for b in qs.binary}
    keys = list(probs)
    st = base_state(q, info, probs)
    try:
        r = caller.ask(q.id, "research", prompts.MOVE_RESEARCH.format(
            **base, label=q.label, window=qs.move_window, scenario_names="; ".join(describe(k, labels) for k in keys)),
            [M.research, M.research_fallback], step="research", online=True,
            fake=lambda p: f"[dry] research https://example.org/{q.id}")
        dossier = clip(r.text, s.method.dossier_chars) if r.ok else "not available"
        st["steps"].append({"step": "research", "ok": r.ok, "model": r.model, "urls": r.urls})
    except BudgetSkip as e:
        dossier = "not available (skipped)"
        st["steps"].append({"step": "research", "ok": False, "skipped": str(e)})
    template = "{" + ", ".join(f'"{k}": <num>' for k in keys) + "}"
    fprompt = prompts.MOVE_FORECAST.format(
        **base, label=q.label, qid=q.id, window=qs.move_window, unit=q.unit_label, typical=round(info["typical"], 4),
        typical_window=s.moves.typical_window, recent=prompts.jdump(info["context"]), history=prompts.jdump(info["history"]),
        dossier=dossier, briefing=clip(briefing, s.method.dossier_chars), bound=round(info["max_abs_u"], 1), template=template,
        scenarios="\n".join(f"- {k}: {describe(k, labels)} (probability {100 * probs[k]:.0f}%)" for k in keys))
    try:
        with ThreadPoolExecutor(max(1, len(M.move_forecasters))) as ex:
            replies = list(ex.map(lambda n: caller.ask(q.id, "forecast", fprompt, [n, M.forecaster_fallback],
                                                       step=f"forecast:{n}", validate=lambda t: parse_moves(t, keys) is not None,
                                                       fake=lambda p: fake_moves(q.id, n, keys)), M.move_forecasters))
    except BudgetSkip as e:
        replies = []
        st["steps"].append({"step": "forecast", "ok": False, "skipped": str(e)})
    for r in (r for r in replies if r.ok):
        st["forecasts"].append({"model": r.model, "reason": str((last_json_object(r.text) or {}).get("reason", ""))[:300],
                                "scenario_moves": {k: clamp(v, info["max_abs_u"]) for k, v in parse_moves(r.text, keys).items()}})
    fc = st["forecasts"]
    if fc:
        mean = {k: sum(f["scenario_moves"][k] for f in fc) / len(fc) for k in keys}
        dm = demean(mean, probs)
        st["aggregate"] = {"mean_moves": mean, "demeaned": dm, "dof": dof, "scenario_u": mixture_median(dm, probs, dof),
                           "n": len(fc)}
    caller.event("aggregate", q.id, st["aggregate"])
    return st


def cross_check(caller: Caller, states: dict, now: str) -> dict[str, list[str]]:
    """Consistency of the proposed moves across instruments: logged and shown to the explainer, never changes a number."""
    s = caller.s
    if not s.models.supervisor:
        return {}
    rows = [f"{q} {st['label']}: drift {st['drift_u']:+.2f}, scenario {st['aggregate']['scenario_u']:+.2f} => "
            f"{(st['drift_u'] + st['aggregate']['scenario_u']) * st['typical']:+.3f} {st['unit']} (typical {st['typical']:.3f})"
            for q, st in states.items()]
    notes = f"Expected relations: {s.method.cross_check_notes}" if s.method.cross_check_notes else ""
    try:
        r = caller.ask("general", "critique", prompts.CROSS_CHECK.format(context=s.method.context, now=now,
                                                                         table="\n".join(rows), notes=notes),
                       [s.models.supervisor], step="cross_check", optional=True, fake=lambda p: '{"flags": [], "reason": "dry"}')
    except BudgetSkip:
        return {}
    o = (last_json_object(r.text) if r.ok else None) or {}
    flags = [str(f) for f in o.get("flags") or []] if isinstance(o.get("flags"), list) else []
    caller.event("cross_check", "general", {"ok": r.ok, "flags": flags, "reason": o.get("reason")})
    return {q: [f for f in flags if f.startswith((q + ":", q + " "))] for q in states}


def finalise(caller: Caller, q: Move, qs: QuestionSet, st: dict, flags: list[str] | None, now: str) -> dict:
    s, ag = caller.s, st["aggregate"]
    pol = policy_move(q, st["typical"], st["drift_u"], ag["scenario_u"])
    st["policy"], st["cross_flags"] = pol.to_dict(), flags or []
    caller.event("policy", q.id, st["policy"])
    if not ag["n"]:   # no scenario forecast: drift only
        why = st.get("error") or "no valid scenario forecast"
        st["fallback"] = {"why": why}
        st["final"] = {"answer": pol.proposal, "source": "fallback: drift only", "reason": why}
        caller.event("final", q.id, st["final"])
        return st
    st["final"] = {"answer": pol.proposal, "source": "policy", "reason": None}
    if s.method.explain_final and s.models.explainer:
        prompt = prompts.EXPLAIN_MOVE.format(
            context=s.method.context, now=now, label=q.label, qid=q.id, window=qs.move_window, unit=q.unit_label,
            typical=round(st["typical"], 4), drift_u=pol.drift_u, drift_note=st["drift_note"], scen_u=pol.scenario_u,
            total_u=pol.total_u, proposal=pol.proposal, cross_note="; ".join(flags) if flags else "no inconsistency flagged",
            scen_detail="; ".join(f"{f['model']}: " + ", ".join(f"{k} {v:+.1f}" for k, v in f["scenario_moves"].items())
                                  for f in st["forecasts"]))
        explain(caller, st, prompt, pol.proposal)
    caller.event("final", q.id, st["final"])
    return st
