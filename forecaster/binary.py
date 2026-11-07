"""Binary questions: research angles (+ gap-driven follow-ups) -> evidence review -> independent forecasts with the
market hidden -> log-odds median -> supervisor -> liquidity-weighted blend with the market -> written reason.
The number always comes from code; models supply evidence, estimates and the explanation."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .jsonparse import ends_with_json_line, last_json_object, to_number
from .llm import BudgetSkip, Caller
from .markets import MarketSnapshot
from .mathx import policy_a, pool_logodds_median, round_half_away
from .questions import Binary, QuestionSet


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...truncated]"


def market_filter(terms: list[str]) -> re.Pattern:
    return re.compile("|".join(terms), re.I)


def drop_market_lines(text: str, pattern: re.Pattern) -> tuple[str, int]:
    """Remove every line that mentions a prediction market or betting: the forecasters must not see the market."""
    lines = text.splitlines()
    kept = [x for x in lines if not pattern.search(x)]
    return "\n".join(kept), len(lines) - len(kept)


def strip_json_line(text: str) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join(lines[:-1]) if lines and lines[-1].strip().startswith("{") else text


def review_ok(text: str) -> bool:
    o = last_json_object(text)
    return isinstance(o, dict) and isinstance(o.get("unresolved"), list) and "CONFLICTS" in text.upper()


def p_yes(text: str) -> float | None:
    p = to_number((last_json_object(text) or {}).get("p_yes"))
    return p if p is not None and 1 <= p <= 99 else None


def explanation_ok(text: str, proposal: float) -> bool:
    o = last_json_object(text) or {}
    a = to_number(o.get("answer"))
    return ends_with_json_line(text) and a is not None and abs(a - proposal) < 1e-9 and bool(o.get("reason"))


def fake_forecast(qid: str, name: str) -> str:
    p = 30 + sum(map(ord, qid + name)) % 41
    return f'[dry]\n{{"p_yes": {p}, "key_evidence": "dry run"}}'


def angles(qs: QuestionSet) -> tuple[dict[str, str], dict[str, str]]:
    a = qs.research_angles or prompts.RESEARCH_ANGLES
    scopes = qs.gap_scopes if qs.research_angles else prompts.GAP_SCOPES
    return a, {k: scopes.get(k, f"Judge only the '{k}' angle.") for k in a}


def run_question(caller: Caller, q: Binary, qs: QuestionSet, snap: MarketSnapshot, briefing: str, now: str) -> dict:
    s, M = caller.s, caller.s.models
    meth, pat = s.method, market_filter(s.method.market_terms)
    base = {"context": meth.context, "now": now}
    st = {"qid": q.id, "kind": "binary", "question": q.question, "market": snap.to_dict(), "steps": []}
    research_models = [M.research, M.research_fallback]
    angle_text, scopes = angles(qs)

    def research(item):
        angle, text = item
        try:
            r = caller.ask(q.id, "research", prompts.RESEARCH.format(**base, question=q.question, yes_means=q.yes_means,
                                                                     angle_text=f"{angle.upper()}. {text}"),
                           research_models, step=f"research_{angle}", online=True,
                           fake=lambda p: f"[dry] {angle} dossier https://example.org/{q.id}/{angle}")
        except BudgetSkip as e:
            return angle, ["not available (skipped)"], {"step": f"research_{angle}", "ok": False, "skipped": str(e)}
        step = {"step": f"research_{angle}", "ok": r.ok, "model": r.model, "urls": r.urls}
        if not r.ok:
            return angle, ["not available"], step
        parts, rounds = [r.text], []
        for k in range(1, meth.research_rounds + 1):   # iterative search: find the gaps, search only those
            try:
                g = caller.ask(q.id, "research", prompts.GAPS.format(**base, question=q.question, angle=angle,
                                                                     scope=scopes[angle], dossier=clip("\n\n".join(parts), meth.dossier_chars)),
                               research_models, step=f"gaps_{angle}_{k}", optional=True,
                               validate=lambda t: isinstance(last_json_object(t), dict) and "enough" in last_json_object(t),
                               fake=lambda p: json.dumps({"enough": k > 1, "missing": "dry", "queries": ["dry query"] if k == 1 else []}))
                o = (last_json_object(g.text) if g.ok else None) or {}
                queries = [str(x) for x in (o.get("queries") or [])][:3]
                rounds.append({"round": k, "enough": o.get("enough"), "missing": str(o.get("missing", ""))[:300], "queries": queries})
                if not g.ok or o.get("enough") is True or not queries:
                    break
                f = caller.ask(q.id, "research", prompts.FOLLOWUP.format(**base, question=q.question, angle=angle,
                                                                         missing=o.get("missing", ""), queries="; ".join(queries)),
                               research_models, step=f"followup_{angle}_{k}", online=True, optional=True,
                               fake=lambda p: f"[dry] follow-up https://example.org/{q.id}/{angle}/{k}")
                rounds[-1] |= {"ok": f.ok, "urls": f.urls}
                if not f.ok:
                    break
                parts.append(f.text)
            except BudgetSkip as e:
                rounds.append({"round": k, "skipped": str(e)})
                break
        if rounds:
            step["rounds"] = rounds
        return angle, parts, step

    with ThreadPoolExecutor(max(1, len(angle_text))) as ex:
        res = list(ex.map(research, angle_text.items()))
    st["steps"] += [step for *_, step in res]
    raw, removed = {}, 0
    for a, parts, _ in res:
        txt, n = drop_market_lines("\n\n".join(parts[:1] + [f"[follow-up round {i}]\n{t}" for i, t in enumerate(parts[1:], 1)]), pat)
        raw[a], removed = clip(txt, meth.review_input_chars), removed + n
    briefing, n_b = drop_market_lines(briefing, pat)
    st["market_lines_removed"] = removed + n_b

    reviewed = review(caller, q, raw, now)
    if reviewed is None:   # reviewer failed: the raw dossiers
        evidence = "\n\n".join(f"RESEARCH DOSSIER ({a}):\n{clip(t, meth.dossier_chars)}" for a, t in raw.items())
        st["review"] = {"ok": False}
    else:
        evidence, st["review"] = reviewed

    fprompt = prompts.FORECAST.format(**base, question=q.question, yes_means=q.yes_means, notes=q.notes,
                                      evidence=evidence, briefing=clip(briefing, meth.dossier_chars))
    try:
        with ThreadPoolExecutor(max(1, len(M.forecasters))) as ex:
            replies = list(ex.map(lambda n: caller.ask(q.id, "forecast", fprompt, [n, M.forecaster_fallback],
                                                       step=f"forecast:{n}", validate=lambda t: p_yes(t) is not None,
                                                       fake=lambda p: fake_forecast(q.id, n)), M.forecasters))
    except BudgetSkip as e:
        replies = []
        st["steps"].append({"step": "forecast", "ok": False, "skipped": str(e)})
    fc = [{"model": r.model, "p": p_yes(r.text) / 100,
           "evidence": str((last_json_object(r.text) or {}).get("key_evidence", ""))[:300]} for r in replies if r.ok]
    st["forecasts"] = fc
    if not fc:
        return fallback(caller, st, snap, "no valid independent forecast")

    ps = [f["p"] for f in fc]
    pooled, spread = pool_logodds_median(ps), 100 * (max(ps) - min(ps))
    agg = {"n": len(ps), "pooled": pooled, "spread_pts": spread, "supervisor": None}
    if M.supervisor:
        try:   # always reviews; overrides only with a cited, dated new fact
            r = caller.ask(q.id, "critique", prompts.SUPERVISE.format(
                **base, question=q.question, yes_means=q.yes_means, spread=spread, evidence=evidence,
                forecasts="\n".join(f"- {f['model']}: {100 * f['p']:.0f}% ({f['evidence']})" for f in fc)),
                [M.supervisor], step="supervisor", online=True, optional=True,
                fake=lambda p: '{"adjusted_p_yes": null, "new_fact": "", "source": "", "reason": "dry"}')
            o = (last_json_object(r.text) if r.ok else None) or {}
            adj, src, fact = to_number(o.get("adjusted_p_yes")), str(o.get("source") or ""), str(o.get("new_fact") or "")
            accepted = adj is not None and 1 <= adj <= 99 and src.startswith("http") and len(fact) > 10
            agg["supervisor"] = {"ok": r.ok, "model": r.model, "adjusted": adj, "fact": fact, "source": src,
                                 "accepted": accepted, "reason": str(o.get("reason", ""))}
            pooled = adj / 100 if accepted else pooled
        except BudgetSkip as e:
            agg["supervisor"] = {"skipped": str(e)}
    own = pooled   # no extremizing: the forecasters share the same evidence, so their errors are correlated
    st["aggregate"] = agg | {"pooled_after_supervisor": pooled, "model_estimate": own}
    caller.event("aggregate", q.id, st["aggregate"])

    w = meth.w_market_lo if snap.w_market is None else snap.w_market
    pol = policy_a(snap.market, own, w, lo=meth.answer_lo, hi=meth.answer_hi)
    st["policy"] = pol.to_dict()
    caller.event("policy", q.id, st["policy"])
    st["final"] = {"answer": pol.proposal, "source": "policy", "reason": None}
    if meth.explain_final and M.explainer:
        sup = agg["supervisor"] or {}
        notes = (f"traded volume {snap.volume:,.0f}" if snap.volume is not None else "volume not reported") + \
                (f"; change over 7 days {snap.change_7d_pts:+.1f} points" if snap.change_7d_pts is not None else "") + \
                (f"; WARNING stale ({snap.stale_reason})" if snap.stale else "")
        prompt = prompts.EXPLAIN_BINARY.format(
            **base, question=q.question, own_int=round_half_away(100 * own), pooled=100 * pooled,
            forecasts=", ".join(f"{f['model']} {100 * f['p']:.0f}%" for f in fc),
            supervisor=(f"\n- supervisor override accepted: {sup['adjusted']:.0f}% because {sup['fact']} ({sup['source']})"
                        if sup.get("accepted") else "\n- supervisor reviewed; no override" if sup.get("ok") else ""),
            venues=", ".join(f"{v.venue} {100 * v.price:.1f}%" for v in snap.venues), market=100 * snap.market,
            market_notes=notes, proposal=pol.proposal, w_market=pol.w_market, w_model=1 - pol.w_market,
            blend=pol.blend_pts, w_lo=meth.w_market_lo, w_hi=meth.w_market_hi)
        explain(caller, st, prompt, pol.proposal)
    caller.event("final", q.id, st["final"])
    return st


def review(caller: Caller, q: Binary, raw: dict[str, str], now: str) -> tuple[str, dict] | None:
    """One clean evidence sheet (latest dated values, data quality, news ranked by importance and date), no probability,
    market hidden. Unresolved conflicts get one targeted web search. None if the reviewer fails."""
    s, M = caller.s, caller.s.models
    if not M.reviewer:
        return None
    pat, base = market_filter(s.method.market_terms), {"context": s.method.context, "now": now}
    dossiers = "\n\n".join(f"DOSSIER {i} ({a}):\n{t}" for i, (a, t) in enumerate(raw.items(), 1))
    try:
        r = caller.ask(q.id, "review", prompts.REVIEW.format(**base, question=q.question, yes_means=q.yes_means,
                                                             notes=q.notes, dossiers=dossiers),
                       [M.reviewer, M.reviewer_fallback], step="review", validate=review_ok,
                       fake=lambda p: "[dry] FUNDAMENTALS ... CONFLICTS: none\n" + json.dumps(
                           {"unresolved": ["dry conflict"], "queries": ["dry query"]}))
    except BudgetSkip:
        return None
    if not r.ok:
        return None
    o = last_json_object(r.text) or {}
    sheet, n = drop_market_lines(strip_json_line(r.text), pat)
    unresolved, queries = [str(x) for x in o.get("unresolved") or []][:3], [str(x) for x in o.get("queries") or []][:3]
    info = {"ok": True, "model": r.model, "unresolved": unresolved, "queries": queries, "market_lines_removed": n}
    if unresolved and queries:
        try:
            c = caller.ask(q.id, "research", prompts.CONFLICT.format(**base, question=q.question,
                                                                     unresolved="\n".join(f"- {u}" for u in unresolved),
                                                                     queries="; ".join(queries)),
                           [M.research, M.research_fallback], step="conflict_search", online=True, optional=True,
                           fake=lambda p: "[dry] conflict settled https://example.org/conflict")
            info |= {"conflict_search_ok": c.ok, "urls": c.urls}
            if c.ok:
                extra, m = drop_market_lines(c.text, pat)
                sheet += f"\n\nCONFLICT CHECK (targeted search, {now}):\n{clip(extra, s.method.conflict_chars)}"
                info["market_lines_removed"] += m
        except BudgetSkip as e:
            info["conflict_search_skipped"] = str(e)
    caller.event("review", q.id, info)
    return clip(sheet, s.method.evidence_chars), info


def explain(caller: Caller, st: dict, prompt: str, proposal: float) -> None:
    """The model writes the reason; the number stays the code's (a reply that changes it is rejected)."""
    try:
        r = caller.ask(st["qid"], "explain", prompt, [caller.s.models.explainer], step="explain", optional=True,
                       validate=lambda t: explanation_ok(t, proposal),
                       fake=lambda p: "[dry]\n" + json.dumps({"answer": proposal, "reason": "dry"}))
    except BudgetSkip:
        return
    if r.ok:
        st["final"] |= {"reason": str((last_json_object(r.text) or {}).get("reason", "")), "explainer": r.model,
                        "explanation": strip_json_line(r.text)}


def fallback(caller: Caller, st: dict, snap: MarketSnapshot, why: str) -> dict:
    """No usable forecast: the market price rounded (zero expected gain over the market)."""
    m = caller.s.method
    p = min(max(round_half_away(100 * snap.market), m.answer_lo), m.answer_hi)
    st["fallback"] = {"why": why}
    st["final"] = {"answer": p, "source": "fallback: market price", "reason": why}
    caller.event("final", st["qid"], st["final"])
    return st
