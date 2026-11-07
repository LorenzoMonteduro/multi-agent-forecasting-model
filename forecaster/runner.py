"""Orchestration of one run: data first, then binary questions (scenario questions first), then move questions."""
from __future__ import annotations

import importlib
import json
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import binary, calibration, moves, prices, prompts, questions, report
from .llm import BudgetSkip, Caller
from .markets import CsvMarketSource, MarketSource
from .mathx import liquidity_weights
from .scenarios import scenario_probs
from .settings import Settings, missing_models

DRY_MODELS = {"research": "dry-research", "forecasters": ["dry-model-a", "dry-model-b", "dry-model-c"],
              "move_forecasters": ["dry-model-a", "dry-model-b"], "reviewer": "dry-reviewer",
              "supervisor": "dry-supervisor", "explainer": "dry-explainer"}


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


def git_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def load_source(spec: str | None, s: Settings) -> MarketSource:
    """Default: the CSV in settings.paths.market_prices. Otherwise 'module:Class', constructed with the settings."""
    if not spec:
        return CsvMarketSource(s.path("market_prices"), max_age_h=s.method.odds_max_age_h)
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)(s)


def move_infos(s: Settings, qs: questions.QuestionSet, series: dict) -> tuple[dict[str, dict], float]:
    m = s.moves
    if missing := [q.instrument for q in qs.moves if q.instrument not in series]:
        raise ValueError(f"no price history for {missing}")
    table = calibration.move_table(series, qs.moves, m.past_events, m.typical_window, m.event_window_lag_bdays)
    dof = calibration.student_t_dof(series, {q.instrument: q.unit for q in qs.moves}, m.typical_window, m.typical_end or None)
    out = {}
    for q in qs.moves:
        t, lv = table[q.id], series[q.instrument]
        if not t["units"]:
            note = "zero: no comparable past events in the data"
        elif t["drift_rule"] == "median":
            note = (f"median of the past day-after-event moves ({t['drift_u']:+.2f} typical moves); leave-one-event-out score "
                    f"{t['loo']['median']:.2f} vs {t['loo']['zero']:.2f} for zero")
        else:
            note = f"zero: beat the median of past events out of sample ({t['loo']['zero']:.2f} vs {t['loo']['median']:.2f})"
        out[q.id] = {"instrument": q.instrument, "typical": prices.typical_move(lv, q.unit, m.typical_window, m.typical_end or None),
                     "drift_u": t["drift_u"], "drift_note": note, "max_abs_u": t["max_abs_u"], "event_units": t["units"],
                     "context": prices.recent_context(lv, q.unit), "history": prices.event_day_moves(lv, q.unit, m.past_events)}
    return out, dof


def run(s: Settings, run_name: str, *, dry: bool = False, only: list[str] | None = None, source: MarketSource | None = None,
        client=None) -> dict[str, dict]:
    qs = questions.load(s.path("questions"))
    wanted = only or qs.ids()
    if unknown := [w for w in wanted if w not in qs.ids()]:
        raise ValueError(f"unknown question id(s): {unknown}")
    qb_ = [q for q in qs.binary if q.id in wanted]
    qm = [q for q in qs.moves if q.id in wanted]
    if dry:   # fill empty model slots so an offline run works with the example config
        for k, v in DRY_MODELS.items():
            if not getattr(s.models, k):
                setattr(s.models, k, v)
    elif miss := missing_models(s, bool(qb_), bool(qm)):
        raise ValueError(f"set these in the config: {', '.join(miss)}")
    # scenario questions run first: the move questions need their answers
    need = list(dict.fromkeys([q.id for q in qb_] + (qs.scenario_questions if qm else [])))
    qb = sorted((q for q in qs.binary if q.id in need), key=lambda q: q.id not in qs.scenario_questions)
    caller = Caller(s, run_name, dry=dry, client=client)
    t0 = time.monotonic()

    # all data before the first model call: a data problem never leaves a half-finished run
    source = source or load_source(None, s)
    every = {q.id: source.snapshot(q.id) for q in qs.binary}   # all binary questions: the market weight is relative
    weights = liquidity_weights({q: x.volume for q, x in every.items()}, s.method.w_market_lo, s.method.w_market_hi)
    for q, x in every.items():
        x.w_market = weights[q]
    snaps = {q: every[q] for q in need}
    infos, dof = move_infos(s, qs, prices.load_series(s.path("price_history"))) if qm else ({}, None)
    if s.web_search.mode == "none":
        log("WARNING web_search.mode = none: research calls rely on the models' own knowledge")
    caller.event("run_start", "general", {"questions": wanted, "commit": git_commit(s.root), "dry": dry,
                                          "market_weights": weights, "student_t_dof": dof, "settings": asdict(s)})
    for q, x in snaps.items():
        caller.event("market_snapshot", q, x.to_dict())
    for q, i in infos.items():
        caller.event("move_data", q, i)
    log("markets: " + ", ".join(f"{q} {100 * x.market:.1f}% w{x.w_market:.2f}{'!' * x.stale}" for q, x in snaps.items())
        + "  (w = market weight, ! = stale)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    def briefing(kind: str) -> str:
        topic = qs.briefing_topic if kind == "binary" else qs.move_briefing_topic
        if not topic:
            return "not available"
        tmpl = prompts.BRIEFING if kind == "binary" else prompts.MOVE_BRIEFING
        try:
            r = caller.ask("general", "research", tmpl.format(context=s.method.context, now=now, topic=topic, window=qs.move_window),
                           [s.models.research, s.models.research_fallback], step=f"briefing_{kind}", online=True,
                           fake=lambda p: f"[dry] {kind} briefing https://example.org/{kind}")
        except BudgetSkip:
            return "not available"
        caller.event("briefing", "general", {"kind": kind, "ok": r.ok, "model": r.model, "urls": r.urls})
        return r.text if r.ok else "not available"

    kinds = ["binary"] * bool(qb) + ["moves"] * bool(qm)
    with ThreadPoolExecutor(2) as ex:
        briefs = dict(zip(kinds, ex.map(briefing, kinds)))
    state_dir = caller.dir / "state"
    state_dir.mkdir(exist_ok=True)

    def save(st: dict) -> dict:
        (state_dir / f"{st['qid']}.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        log(f"{st['qid']} answer={st['final']['answer']}" + (f" (market {100 * snaps[st['qid']].market:.1f}%)"
                                                             if st["qid"] in snaps else f" {st.get('unit', '')}"))
        return st

    def do_binary(q):
        snap = snaps[q.id]
        try:
            return save(binary.run_question(caller, q, qs, snap, briefs.get("binary", ""), now))
        except Exception as e:
            caller.event("pipeline_error", q.id, {"error": repr(e), "trace": traceback.format_exc()[-2000:]})
            log(f"{q.id} error {e!r} -> market price")
            return save(binary.fallback(caller, {"qid": q.id, "kind": "binary", "question": q.question,
                                                 "market": snap.to_dict(), "steps": []}, snap, f"pipeline error {e.__class__.__name__}"))

    states = {}
    with ThreadPoolExecutor(s.limits.workers) as pool:
        fut = {q.id: pool.submit(do_binary, q) for q in qb}
        if qm:
            p = {q: fut[q].result()["final"]["answer"] / 100 for q in qs.scenario_questions}
            probs = scenario_probs(p, s.moves.scenario_coupling)
            caller.event("scenario_probs", "general", {"marginals": p, "coupling": s.moves.scenario_coupling, "probs": probs})
            by_id = {q.id: q for q in qm}

            def prep(q):
                try:
                    return moves.prepare(caller, q, qs, infos[q.id], probs, dof, briefs.get("moves", ""), now)
                except Exception as e:
                    caller.event("pipeline_error", q.id, {"error": repr(e), "trace": traceback.format_exc()[-2000:]})
                    log(f"{q.id} error {e!r} -> drift only")
                    return moves.base_state(q, infos[q.id], probs) | {"error": f"pipeline error {e.__class__.__name__}"}

            prepared = {st["qid"]: st for st in pool.map(prep, qm)}
            flags = moves.cross_check(caller, prepared, now) if len(prepared) >= 3 else {}
            for qid, st in prepared.items():
                states[qid] = save(moves.finalise(caller, by_id[qid], qs, st, flags.get(qid), now))
        states |= {q: f.result() for q, f in fut.items()}

    caller.event("run_summary", "general", {"seconds": round(time.monotonic() - t0), **caller.usage})
    idx = report.write_outputs(caller.dir, qs.ids())
    log(f"done in {time.monotonic() - t0:.0f}s, {caller.usage['calls']} calls; answers: {caller.dir / 'answers.csv'}; cards: {idx}")
    return {q: states[q] for q in wanted if q in states}
