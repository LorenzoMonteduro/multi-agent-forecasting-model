"""Model gateway and end-to-end runs: offline (dry) and through a fake OpenAI-compatible client (the live code path)."""
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from forecaster import runner, settings
from forecaster.binary import drop_market_lines, market_filter
from forecaster.llm import BudgetSkip, Caller

ROOT = Path(__file__).resolve().parent.parent


def make_settings(tmp_path, **sections) -> settings.Settings:
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    data = {"models": {"research": "r", "forecasters": ["f1", "f2", "f3"], "move_forecasters": ["f1", "f2"],
                       "reviewer": "rev", "supervisor": "sup", "explainer": "exp"},
            "moves": {"past_events": ["2014-11-04", "2016-11-08", "2018-11-06", "2020-11-03", "2022-11-08", "2024-11-05"]},
            "method": {"context": "test"}} | sections
    return settings.from_dict(data, tmp_path)


def test_request_modes(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    s = make_settings(tmp_path, provider={"no_temperature_models": ["reason/"], "extra_body": {"a": 1}},
                      web_search={"mode": "extra_body", "extra_body": {"plugins": [{"id": "web"}]}})
    c = Caller(s, "t", client=object())
    kw = c.request("research", "m", [], online=True, attempt=0)
    assert kw["extra_body"] == {"a": 1, "plugins": [{"id": "web"}]} and kw["temperature"] == 0.2 and kw["max_tokens"] == 8000
    assert "plugins" not in c.request("research", "m", [], online=False, attempt=1)["extra_body"]
    assert c.request("research", "m", [], online=False, attempt=1)["max_tokens"] == 16000
    assert "temperature" not in c.request("forecast", "reason/x", [], online=False, attempt=0)
    s.web_search.mode, s.web_search.suffix = "suffix", ":search"
    assert c.request("research", "m", [], online=True, attempt=0)["model"] == "m:search"


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        Caller(make_settings(tmp_path), "t")


def test_call_cap_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-key-123")
    s = make_settings(tmp_path, limits={"max_calls": 1})
    c = Caller(s, "t", dry=True)
    r = c.ask("Q", "other", "say secret-key-123", ["m"], step="x", fake=lambda p: p)
    assert r.ok
    with pytest.raises(BudgetSkip):
        c.ask("Q", "other", "hi", ["m"], step="y", optional=True)
    assert c.ask("Q", "other", "hi", ["m"], step="z").ok   # non-optional calls still run
    log = (c.dir / "calls.jsonl").read_text()
    assert "secret-key-123" not in log and "<LLM_API_KEY>" in log


def test_fallback_model_and_validation(tmp_path):
    c = Caller(make_settings(tmp_path), "t", dry=True)
    r = c.ask("Q", "other", "p", ["bad", "good"], step="s", fake=lambda p: "x", validate=lambda t: False)
    assert not r.ok and r.attempts == 4


def test_market_filter():
    text, n = drop_market_lines("poll: +3\nPolymarket has 60%\nKalshi 58c\nrating: lean", market_filter(["kalshi", "polymarket"]))
    assert text == "poll: +3\nrating: lean" and n == 2


def test_dry_run_end_to_end(tmp_path):
    s = make_settings(tmp_path)
    states = runner.run(s, "e2e", dry=True)
    assert set(states) == {"LOWER", "UPPER", "D1", "D2", "M1", "M2", "M3"}
    for q, st in states.items():
        a = st["final"]["answer"]
        if st["kind"] == "binary":
            pol = st["policy"]
            assert a == pol["proposal"] and 1 <= a <= 99
            assert abs(a - 100 * st["market"]["market"]) <= 33.5 + 0.5    # market weight >= 0.67
        else:
            assert a == st["policy"]["proposal"] and not st.get("fallback")
    d = tmp_path / "runs" / "dry-e2e"
    assert (d / "answers.csv").read_text().count("\n") == 8 and (d / "cards" / "index.md").exists()
    kinds = {json.loads(l)["kind"] for l in (d / "events.jsonl").read_text().splitlines()}
    assert {"run_start", "market_snapshot", "move_data", "aggregate", "policy", "final", "scenario_probs", "run_summary"} <= kinds


class FakeClient:
    """Answers each prompt with a valid reply of the kind it asks for, like a well-behaved model."""

    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def reply(self, prompt: str, model: str) -> str:
        if m := re.search(r"The answer is (-?[\d.]+);", prompt):
            return "Because.\n" + json.dumps({"answer": float(m.group(1)), "reason": "fake"})
        if '"enough"' in prompt:
            return json.dumps({"enough": True, "missing": "", "queries": []})
        if '"unresolved"' in prompt:
            return "FUNDAMENTALS x\nCONFLICTS: none\n" + json.dumps({"unresolved": [], "queries": []})
        if '"adjusted_p_yes"' in prompt:
            return json.dumps({"adjusted_p_yes": None, "new_fact": "", "source": "", "reason": "fine"})
        if '"p_yes"' in prompt:
            return "Reasoning.\n" + json.dumps({"p_yes": 55 + len(model), "key_evidence": "fake"})
        if '"scenario_moves"' in prompt:
            keys = re.search(r'"scenario_moves": \{(.*?)\}, "reason"', prompt).group(1)
            ks = re.findall(r'"([^"]+)": <num>', keys)
            return json.dumps({"scenario_moves": {k: 0.5 - 0.3 * i for i, k in enumerate(ks)}, "reason": "fake"})
        if '"flags"' in prompt:
            return json.dumps({"flags": [], "reason": "consistent"})
        return "Research text with a source https://news.example.com/story and Kalshi at 60% (must be filtered)."

    def create(self, **kw):
        self.calls.append(kw)
        text = self.reply(kw["messages"][-1]["content"], kw["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, annotations=[]))],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_live_path_with_fake_client(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    s = make_settings(tmp_path, web_search={"mode": "extra_body", "extra_body": {"plugins": [{"id": "web"}]}})
    client = FakeClient()
    states = runner.run(s, "live", client=client)
    assert all(not st.get("fallback") for st in states.values())
    assert all(st["final"]["reason"] == "fake" for st in states.values())
    st = states["D1"]
    assert [f["p"] for f in st["forecasts"]] == [0.57, 0.57, 0.57]
    research = [c for c in client.calls if c.get("extra_body", {}).get("plugins")]
    assert research and all(c["model"] == "r" or c["model"] == "sup" for c in research)
    calls = (tmp_path / "runs" / "live" / "calls.jsonl").read_text()
    forecasts = [json.loads(l) for l in calls.splitlines() if json.loads(l)["role"] == "forecast"
                 and json.loads(l)["question"] == "D1"]
    assert forecasts and all("Kalshi" not in json.dumps(f["prompt"]) for f in forecasts)   # market hidden
    summary = [json.loads(l) for l in (tmp_path / "runs" / "live" / "events.jsonl").read_text().splitlines()
               if json.loads(l)["kind"] == "run_summary"][0]["data"]
    assert summary["prompt_tokens"] == 10 * summary["calls"]


def test_evaluate(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT))
    from evaluate import evaluate
    s = make_settings(tmp_path)
    states = runner.run(s, "ev", dry=True)
    d = tmp_path / "runs" / "dry-ev"
    res = evaluate(d, {"D1": 1, "M1": 0.0})
    st = states["D1"]
    m, a = st["market"]["market"], st["final"]["answer"] / 100
    assert res["binary_mean_gain"] == pytest.approx(((m - 1) ** 2 - (a - 1) ** 2) * 100)
    assert res["move_zero_error"] == 0 and res["move_n"] == 1
