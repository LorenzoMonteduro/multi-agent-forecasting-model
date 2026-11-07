"""Outputs built from a run's logs: answers.csv and one Markdown decision card per question."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .llm import read_jsonl


def load_states(run_dir: Path) -> dict[str, dict]:
    d = run_dir / "state"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def _sources(st: dict) -> list[str]:
    urls = []
    for s in st.get("steps", []):
        urls += s.get("urls") or []
        for r in s.get("rounds") or []:
            urls += r.get("urls") or []
    urls += (st.get("review") or {}).get("urls") or []
    return list(dict.fromkeys(u for u in urls if "example.org" not in u))


def _tail(st: dict) -> list[str]:
    fin = st.get("final") or {}
    out = ([f"**Fallback**: {st['fallback']['why']}"] if st.get("fallback") else []) + [
        f"## Final\nAnswer **{fin.get('answer')}** ({fin.get('source')})"]
    if fin.get("explanation"):
        out.append(f"Reason ({fin.get('explainer')}): {fin['explanation']}")
    return out + ["## Sources"] + ([f"- {u}" for u in _sources(st)] or ["- none captured"])


def card(st: dict) -> str:
    if st.get("kind") == "binary":
        m, ag, pol = st["market"], st.get("aggregate"), st.get("policy")
        venues = ", ".join(f"{v['venue']} {100 * v['price']:.1f}% (volume {v['volume']}, {v['time_utc']} UTC)" for v in m["venues"])
        out = [f"# {st['qid']} - {st['question']}",
               f"**Market** {100 * m['market']:.1f}% = mean of {venues}; market weight {m.get('w_market')}"
               + (f"; **STALE** {m['stale_reason']}" if m.get("stale") else ""),
               f"**Evidence review** {json.dumps(st.get('review'), ensure_ascii=False)}; market lines removed: "
               f"{st.get('market_lines_removed', 0)}", "## Forecasts (market hidden)"] + [
            f"- {f['model']}: {100 * f['p']:.0f}% - {f['evidence']}" for f in st.get("forecasts", [])]
        if ag:
            out.append(f"Pooled {100 * ag['pooled']:.1f}% (range {ag['spread_pts']:.0f} points), after the supervisor "
                       f"{100 * ag['pooled_after_supervisor']:.1f}% (our estimate); supervisor: "
                       f"{json.dumps(ag['supervisor'], ensure_ascii=False)}")
        if pol:
            out.append(f"## Policy\n{pol['w_market']:.3f} x market {pol['market_pts']:.1f} + {1 - pol['w_market']:.3f} x own "
                       f"{pol['model_pts']:.1f} = {pol['blend_pts']:.2f}% -> **{pol['proposal']}**")
    else:
        ag, pol = st.get("aggregate", {}), st.get("policy")
        out = [f"# {st['qid']} - {st['label']} ({st['unit']})",
               f"Typical move {st['typical']:.4f} {st['unit']}; drift: {st.get('drift_note')}",
               f"Scenario probabilities {json.dumps({k: round(v, 3) for k, v in st['probs'].items()})}; past event moves "
               f"{json.dumps(st.get('history'))}", "## Scenario moves (typical-move units)"] + [
            f"- {f['model']}: {json.dumps(f['scenario_moves'])} - {f['reason']}" for f in st.get("forecasts", [])] + [
            f"Mixture median {ag.get('scenario_u', 0):+.3f} (Student-t, {ag.get('dof') or 0:.2f} degrees of freedom); "
            f"cross-instrument flags {st.get('cross_flags')}"]
        if pol:
            out.append(f"## Policy\ndrift {pol['drift_u']:+.2f} + scenario {pol['scenario_u']:+.2f} = {pol['total_u']:+.2f} "
                       f"typical moves -> **{pol['proposal']} {st['unit']}**")
    return "\n\n".join(out + _tail(st)) + "\n"


def write_outputs(run_dir: Path, order: list[str]) -> Path:
    """answers.csv and cards/ (index.md + one card per question)."""
    states, cards = load_states(run_dir), run_dir / "cards"
    cards.mkdir(exist_ok=True)
    qids = [q for q in order if q in states]
    with open(run_dir / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "kind", "answer", "unit", "market_or_typical", "own_estimate", "source"])
        for q in qids:
            st, fin = states[q], states[q].get("final") or {}
            if st.get("kind") == "binary":
                own = (st.get("aggregate") or {}).get("model_estimate")
                w.writerow([q, "binary", fin.get("answer"), "% YES", round(100 * st["market"]["market"], 2),
                            "" if own is None else round(100 * own, 2), fin.get("source")])
            else:
                w.writerow([q, "move", fin.get("answer"), st["unit"], round(st["typical"], 4),
                            round((st.get("policy") or {}).get("total_u", 0), 3), fin.get("source")])
    rows = ["| Question | market / typical move | own estimate / total | answer | fallback |", "|---|---|---|---|---|"]
    for q in qids:
        st, fin = states[q], states[q].get("final") or {}
        (cards / f"{q}.md").write_text(card(st), encoding="utf-8")
        if st.get("kind") == "binary":
            own = (st.get("aggregate") or {}).get("model_estimate")
            left, mid = f"{100 * st['market']['market']:.1f}%", "" if own is None else f"{100 * own:.0f}%"
        else:
            left, mid = f"{st['typical']:.3f} {st['unit']}", f"{(st.get('policy') or {}).get('total_u', 0):+.2f} typical"
        rows.append(f"| [{q}]({q}.md) | {left} | {mid} | {fin.get('answer')} | {'yes' if st.get('fallback') else ''} |")
    checks = [f"- {e['kind']}: {json.dumps(e['data'], ensure_ascii=False)[:1500]}" for e in read_jsonl(run_dir / "events.jsonl")
              if e["kind"] in ("briefing", "scenario_probs", "cross_check", "budget_skip", "pipeline_error", "run_summary")]
    (cards / "index.md").write_text("\n".join(["# Decision cards", "", *rows, "", "## Run-level checks", *(checks or ["- none"])]) + "\n",
                                    encoding="utf-8")
    return cards / "index.md"
