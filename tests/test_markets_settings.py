import pandas as pd
import pytest

from forecaster import questions, settings
from forecaster.markets import CsvMarketSource, venue_price

CSV = """question_id,venue,time_utc,price,bid,ask,volume
Q1,a,2026-01-01T10:00:00Z,,0.40,0.44,100
Q1,a,2026-01-08T10:00:00Z,,0.50,0.54,100
Q1,b,2026-01-08T09:00:00Z,0.56,,,300
Q2,a,2026-01-08T10:00:00Z,0.2,,,
"""


def test_csv_source(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(CSV)
    src = CsvMarketSource(p, max_age_h=24, now=pd.Timestamp("2026-01-08T12:00Z"))
    s = src.snapshot("Q1")
    assert s.market == pytest.approx(0.54) and s.volume == 400 and s.time_utc == "2026-01-08 09:00"
    assert s.change_7d_pts == pytest.approx(100 * (0.54 - (0.42 + 0.56) / 2))
    assert not s.stale and src.snapshot("Q2").volume is None
    assert CsvMarketSource(p, max_age_h=1, now=pd.Timestamp("2026-01-08T12:00Z")).snapshot("Q1").stale
    with pytest.raises(RuntimeError):
        src.snapshot("Q9")
    assert venue_price(0.3, None, None) == 0.3 and venue_price(None, 0.2, 0.4) == pytest.approx(0.3)


def test_settings_defaults_and_errors(tmp_path):
    s = settings.from_dict({"models": {"research": "r", "forecasters": ["f"]}}, tmp_path)
    assert s.method.w_market_lo == 0.67 and s.path("runs") == tmp_path / "runs"
    with pytest.raises(ValueError, match="unknown setting"):
        settings.from_dict({"method": {"w_markt_lo": 0.5}})
    with pytest.raises(ValueError, match="unknown section"):
        settings.from_dict({"secrets": {}})
    with pytest.raises(ValueError):
        settings.from_dict({"method": {"w_market_lo": 0.9, "w_market_hi": 0.8}})
    with pytest.raises(ValueError):
        settings.from_dict({"web_search": {"mode": "magic"}})
    assert settings.missing_models(settings.Settings(), True, True) == ["models.research", "models.forecasters", "models.move_forecasters"]


def test_example_config_loads():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    s = settings.load(root / "config.example.toml")
    qs = questions.load(s.path("questions"))
    assert qs.scenario_questions == ["LOWER", "UPPER"] and len(qs.moves) == 3


def test_question_validation():
    base = {"binary": [{"id": "A", "question": "?"}], "moves": [{"id": "M", "label": "x", "instrument": "X"}]}
    with pytest.raises(ValueError, match="scenario_questions"):
        questions.from_dict(base)
    assert questions.from_dict(base | {"scenario_questions": ["A"]}).moves[0].unit_label == "%"
    with pytest.raises(ValueError, match="unique"):
        questions.from_dict({"binary": [{"id": "A", "question": "?"}, {"id": "A", "question": "?"}]})
    with pytest.raises(ValueError, match="unit"):
        questions.from_dict({"binary": [{"id": "A", "question": "?"}], "scenario_questions": ["A"],
                             "moves": [{"id": "M", "label": "x", "instrument": "X", "unit": "dollars"}]})
    with pytest.raises(ValueError, match="unknown key"):
        questions.from_dict({"binaries": []})
