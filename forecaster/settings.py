"""Run settings, read from a TOML file (see config.example.toml). Every number the method uses lives here."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Provider:
    """Any endpoint that speaks the OpenAI chat-completions protocol."""
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "LLM_API_KEY"       # name of the environment variable that holds the key (never the key)
    timeout_s: float = 120.0
    transport_retries: int = 1
    send_temperature: bool = True
    no_temperature_models: list[str] = field(default_factory=list)   # model ids (or prefixes) that reject temperature
    extra_body: dict = field(default_factory=dict)                     # sent with every call (provider-specific)


@dataclass
class WebSearch:
    """How a research call asks the provider to search the web.
    mode = "none"        no search: research relies on the model's own knowledge (a warning is logged)
         = "extra_body"  merge `extra_body` into the request (for example a provider's web-search plugin)
         = "suffix"      append `suffix` to the model id (for providers that expose search models that way)
    """
    mode: str = "none"
    extra_body: dict = field(default_factory=dict)
    suffix: str = ""


@dataclass
class Models:
    research: str = ""
    research_fallback: str = ""
    forecasters: list[str] = field(default_factory=list)          # binary questions: several families
    forecaster_fallback: str = ""
    move_forecasters: list[str] = field(default_factory=list)     # move questions: scenario moves
    reviewer: str = ""
    reviewer_fallback: str = ""
    supervisor: str = ""
    explainer: str = ""                                            # writes the reason for the final number


@dataclass
class Method:
    context: str = ""                    # one paragraph of background inserted in every prompt
    research_rounds: int = 2             # gap-driven follow-up searches per research angle (0 = one pass)
    review_input_chars: int = 16000
    evidence_chars: int = 12000
    conflict_chars: int = 2500
    dossier_chars: int = 7000
    w_market_lo: float = 0.67            # market weight for the least-traded question
    w_market_hi: float = 0.80            # market weight for the most-traded question
    answer_lo: int = 1                   # binary answers are whole percentages in [answer_lo, answer_hi]
    answer_hi: int = 99
    market_terms: list[str] = field(default_factory=lambda: [
        "kalshi", "polymarket", "predictit", "prediction[- ]market", "betting", "bookmaker", "sportsbook"])
    odds_max_age_h: float = 24.0         # older market quotes are flagged stale (logged, not refused)
    explain_final: bool = True           # one extra call per question that writes the reason (never the number)
    cross_check_notes: str = ""          # expected relations between instruments, for the logged cross-check


@dataclass
class Moves:
    """Statistics for move questions, estimated from the price history at run time."""
    typical_window: int = 60             # trading days in the median absolute move
    typical_end: str = ""                # last date of that window (YYYY-MM-DD); empty = last date in the data
    event_window_lag_bdays: int = 2      # for past events, the window ends this many business days before the event
    past_events: list[str] = field(default_factory=list)          # dates (YYYY-MM-DD) of comparable past events
    scenario_coupling: str = "comonotone"   # "comonotone" (one common swing) or "independent"


@dataclass
class Limits:
    workers: int = 8
    call_attempts: int = 2               # per model, then the next model in the list
    step_deadline_s: float = 300.0
    max_calls: int = 2000                # kill switch: optional and non-final calls stop beyond this
    max_tokens: dict = field(default_factory=lambda: {
        "research": 8000, "forecast": 8000, "review": 10000, "critique": 8000, "explain": 4000, "other": 4000})
    temperature: dict = field(default_factory=lambda: {
        "research": 0.2, "forecast": 0.2, "review": 0.0, "critique": 0.0, "explain": 0.0, "other": 0.0})


@dataclass
class Paths:
    questions: str = "examples/demo/questions.json"
    market_prices: str = "examples/demo/market_prices.csv"
    price_history: str = "examples/demo/prices_daily.csv"
    runs: str = "runs"


@dataclass
class Settings:
    provider: Provider = field(default_factory=Provider)
    web_search: WebSearch = field(default_factory=WebSearch)
    models: Models = field(default_factory=Models)
    method: Method = field(default_factory=Method)
    moves: Moves = field(default_factory=Moves)
    limits: Limits = field(default_factory=Limits)
    paths: Paths = field(default_factory=Paths)
    root: Path = field(default_factory=Path.cwd)

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else self.root / p

    def model_ids(self) -> set[str]:
        m = self.models
        return {x for x in (m.research, m.research_fallback, *m.forecasters, m.forecaster_fallback, *m.move_forecasters,
                            m.reviewer, m.reviewer_fallback, m.supervisor, m.explainer) if x}


def _build(cls, data: dict, where: str):
    known = {f.name for f in fields(cls)}
    if unknown := set(data) - known:
        raise ValueError(f"unknown setting(s) in [{where}]: {', '.join(sorted(unknown))}")
    return cls(**data)


SECTIONS = {"provider": Provider, "web_search": WebSearch, "models": Models, "method": Method, "moves": Moves,
            "limits": Limits, "paths": Paths}


def from_dict(data: dict, root: Path | None = None) -> Settings:
    if unknown := set(data) - set(SECTIONS):
        raise ValueError(f"unknown section(s): {', '.join(sorted(unknown))}")
    s = Settings(**{k: _build(cls, data.get(k, {}), k) for k, cls in SECTIONS.items()})
    s.root = root or Path.cwd()
    validate(s)
    return s


def load(path: str | Path) -> Settings:
    p = Path(path)
    with open(p, "rb") as f:
        return from_dict(tomllib.load(f), p.resolve().parent)


def validate(s: Settings) -> None:
    m = s.method
    if not 0 <= m.w_market_lo <= m.w_market_hi <= 1:
        raise ValueError("need 0 <= w_market_lo <= w_market_hi <= 1")
    if not 0 <= m.answer_lo < m.answer_hi <= 100:
        raise ValueError("need 0 <= answer_lo < answer_hi <= 100")
    if s.web_search.mode not in ("none", "extra_body", "suffix"):
        raise ValueError("web_search.mode must be none, extra_body or suffix")
    if s.moves.scenario_coupling not in ("comonotone", "independent"):
        raise ValueError("moves.scenario_coupling must be comonotone or independent")
    if m.research_rounds < 0:
        raise ValueError("research_rounds must be >= 0")


def missing_models(s: Settings, need_binary: bool, need_moves: bool) -> list[str]:
    m, out = s.models, []
    if need_binary or need_moves:
        out += [] if m.research else ["models.research"]
    if need_binary:
        out += [] if m.forecasters else ["models.forecasters"]
    if need_moves:
        out += [] if m.move_forecasters else ["models.move_forecasters"]
    return out
