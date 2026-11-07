"""Question file (JSON). Two kinds:

- binary: "will X happen?" -> probability (whole percent) that the YES outcome happens, anchored on market prices;
- moves:  "how much will instrument I move between two closes?" -> one number (median of the forecast distribution).

See examples/demo/questions.json for the full format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

UNITS = {"pct": "%", "points": "points", "bp": "bp"}   # how a move is measured from closing levels


@dataclass
class Binary:
    id: str
    question: str
    yes_means: str = ""          # what counts as YES, stated once for every prompt
    short: str = ""              # short label used in scenario names (for example "Senate")
    notes: str = ""              # facts the forecasters must not get wrong (rules, candidates, thresholds)


@dataclass
class Move:
    id: str
    label: str
    instrument: str              # name in the price-history CSV
    unit: str = "pct"            # pct (percentage change), points (level difference), bp (yield change x 100)
    decimals: int = 2
    limit: float = 1e9           # answers are clamped to +/- limit

    @property
    def unit_label(self) -> str:
        return UNITS[self.unit]


@dataclass
class QuestionSet:
    binary: list[Binary] = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)
    scenario_questions: list[str] = field(default_factory=list)   # binary ids whose outcomes drive the move scenarios
    move_window: str = ""            # for example "from the close on 3 Nov 2026 to the close on 4 Nov 2026"
    research_angles: dict[str, str] = field(default_factory=dict)  # optional override of the default angles
    gap_scopes: dict[str, str] = field(default_factory=dict)
    briefing_topic: str = ""         # common background for all binary questions (empty = no briefing call)
    move_briefing_topic: str = ""    # common background for all move questions

    def ids(self) -> list[str]:
        return [q.id for q in self.binary] + [q.id for q in self.moves]

    def binary_by_id(self) -> dict[str, Binary]:
        return {q.id: q for q in self.binary}


def from_dict(d: dict) -> QuestionSet:
    qs = QuestionSet(binary=[Binary(**x) for x in d.get("binary", [])], moves=[Move(**x) for x in d.get("moves", [])],
                     scenario_questions=list(d.get("scenario_questions", [])), move_window=d.get("move_window", ""),
                     research_angles=dict(d.get("research_angles", {})), gap_scopes=dict(d.get("gap_scopes", {})),
                     briefing_topic=d.get("briefing_topic", ""), move_briefing_topic=d.get("move_briefing_topic", ""))
    if unknown := set(d) - {"binary", "moves", "scenario_questions", "move_window", "research_angles", "gap_scopes",
                            "briefing_topic", "move_briefing_topic", "_comment"}:
        raise ValueError(f"unknown key(s) in the question file: {sorted(unknown)}")
    ids = qs.ids()
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    if bad := [m.id for m in qs.moves if m.unit not in UNITS]:
        raise ValueError(f"unknown unit for {bad}: use one of {', '.join(UNITS)}")
    if bad := [q for q in qs.scenario_questions if q not in qs.binary_by_id()]:
        raise ValueError(f"scenario_questions must be binary question ids: {bad}")
    if qs.moves and not qs.scenario_questions:
        raise ValueError("move questions need at least one entry in scenario_questions")
    if set(qs.gap_scopes) - set(qs.research_angles):
        raise ValueError("every gap_scopes key must also be a research_angles key")
    return qs


def load(path: str | Path) -> QuestionSet:
    return from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
