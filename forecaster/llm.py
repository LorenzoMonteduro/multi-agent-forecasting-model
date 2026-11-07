"""One gateway to the models: any OpenAI-compatible endpoint, optional web search, fallbacks, retries, a call cap,
a full prompt/reply log (append-only JSON lines) and a deterministic fake for offline dry runs."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from .jsonparse import extract_urls
from .settings import Settings


class BudgetSkip(RuntimeError):
    """An optional or non-essential call refused by the call cap."""


@dataclass
class Reply:
    ok: bool
    text: str
    model: str
    urls: list[str] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0


def safe_name(run: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(run))


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def role_key(role: str) -> str:
    return role if role in ("research", "forecast", "review", "critique", "explain") else "other"


class Caller:
    """`dry=True` never touches the network: each call returns `fake(prompt)`."""

    def __init__(self, settings: Settings, run: str, dry: bool = False, client=None):
        self.s, self.run, self.dry = settings, str(run), dry
        self.dir = settings.path("runs") / (("dry-" if dry else "") + safe_name(run))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock, self._calls, self._usage = threading.Lock(), 0, {"prompt_tokens": 0, "completion_tokens": 0}
        self._client = client
        if not dry and client is None:
            key = os.environ.get(settings.provider.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {settings.provider.api_key_env} is not set (see .env.example)")
            from openai import OpenAI
            self._client = OpenAI(base_url=settings.provider.base_url, api_key=key,
                                  max_retries=settings.provider.transport_retries, timeout=settings.provider.timeout_s)

    # ---- logging -------------------------------------------------------------------------------------------
    def _redact(self, text: str) -> str:
        key = os.environ.get(self.s.provider.api_key_env)
        return text.replace(key, f"<{self.s.provider.api_key_env}>") if key and len(key) >= 6 else text

    def _append(self, name: str, rec: dict) -> None:
        line = self._redact(json.dumps({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        "run": self.run, **rec}, ensure_ascii=False, default=str))
        with self._lock, open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def event(self, kind: str, qid: str, data: dict) -> None:
        self._append("events.jsonl", {"kind": kind, "question": qid, "data": data})

    @property
    def usage(self) -> dict:
        return {"calls": self._calls, **self._usage}

    # ---- calls ---------------------------------------------------------------------------------------------
    def ask(self, qid: str, role: str, prompt: str | list[dict], models: Sequence[str], *, step: str,
            online: bool = False, fake: Callable[[str], str] | None = None,
            validate: Callable[[str], bool] | None = None, optional: bool = False) -> Reply:
        """Try each model up to `call_attempts` times within `step_deadline_s`; return the first valid reply.
        Beyond `max_calls`, optional calls are refused (BudgetSkip) and others still run."""
        lim, models = self.s.limits, [m for m in models if m]
        if optional and self._calls >= lim.max_calls:
            self.event("budget_skip", qid, {"step": step, "role": role, "why": f"call cap {lim.max_calls} reached"})
            raise BudgetSkip(f"{step}: skipped (call cap reached)")
        if not models:
            return Reply(False, "", "?", error="no model configured for this step")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        err, n, t0 = None, 0, time.monotonic()
        for name in models:
            for attempt in range(lim.call_attempts):
                if n and time.monotonic() - t0 > lim.step_deadline_s:
                    return Reply(False, "", name, error=f"step deadline reached after {n} attempts", attempts=n)
                n, started = n + 1, time.monotonic()
                with self._lock:
                    self._calls += 1
                try:
                    if self.dry:
                        text = fake(messages[-1]["content"]) if fake else f"[dry] {step}"
                        urls, usage = extract_urls(text), None
                    else:
                        text, urls, usage = self._live(role, name, messages, online, attempt)
                    ok = bool(text.strip()) and (validate(text) if validate else True)
                    err = None if ok else "empty or invalid reply"
                except Exception as e:   # transport, timeout, 4xx/5xx after the client's retries
                    text, urls, usage, ok, err = "", [], None, False, f"{e.__class__.__name__}: {str(e)[:300]}"
                self._append("calls.jsonl", {"question": qid, "role": role, "step": step, "model": name,
                                             "online": online, "attempt": n, "ok": ok, "error": err,
                                             "seconds": round(time.monotonic() - started, 2), "usage": usage,
                                             "prompt": messages, "reply": text, "urls": urls, "dry": self.dry})
                if ok:
                    return Reply(True, text, name, urls, None, n)
        return Reply(False, "", models[-1], error=err, attempts=n)

    def request(self, role: str, model: str, messages: list[dict], online: bool, attempt: int) -> dict:
        """The keyword arguments of one chat-completions request (separate so it can be tested offline)."""
        p, ws, lim = self.s.provider, self.s.web_search, self.s.limits
        body = dict(p.extra_body)
        if online and ws.mode == "extra_body":
            body |= ws.extra_body
        model_id = model + ws.suffix if online and ws.mode == "suffix" else model
        kw = {"model": model_id, "messages": messages,
              "max_tokens": int(lim.max_tokens.get(role_key(role), 4000)) * (2 if attempt else 1)}   # a cut reply: double
        if p.send_temperature and not any(model.startswith(x) for x in p.no_temperature_models):
            kw["temperature"] = float(lim.temperature.get(role_key(role), 0.0))
        if body:
            kw["extra_body"] = body
        return kw

    def _live(self, role: str, model: str, messages: list[dict], online: bool,
              attempt: int) -> tuple[str, list[str], dict | None]:
        resp = self._client.chat.completions.create(**self.request(role, model, messages, online, attempt))
        msg = resp.choices[0].message
        text = msg.content or ""
        cited = [getattr(getattr(a, "url_citation", None), "url", None) for a in getattr(msg, "annotations", None) or []]
        u = getattr(resp, "usage", None)
        usage = None
        if u is not None:
            usage = {"prompt_tokens": getattr(u, "prompt_tokens", None), "completion_tokens": getattr(u, "completion_tokens", None)}
            with self._lock:
                for k in self._usage:
                    self._usage[k] += usage[k] or 0
        return text, list(dict.fromkeys([x for x in cited if x] + extract_urls(text))), usage
