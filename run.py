"""Run the forecasting agent.

    python run.py --dry                                  offline run on the demo questions (fake model replies, no key)
    python run.py --config config.toml --run my-run      live run
    python run.py --config config.toml --run my-run --questions Q1,M2
    python run.py --config config.toml --run my-run --market-source mypackage.live:MySource
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from forecaster import envfile, runner, settings  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="TOML settings (default: config.toml, else config.example.toml)")
    ap.add_argument("--run", default="demo", help="run name: outputs go to runs/<name>/")
    ap.add_argument("--questions", default="all", help="comma-separated question ids, or all")
    ap.add_argument("--dry", action="store_true", help="fake model replies: no network, no key, no cost")
    ap.add_argument("--market-source", default=None, help="module:Class providing snapshot(qid); default: the CSV")
    a = ap.parse_args()
    cfg = Path(a.config) if a.config else next(p for p in (ROOT / "config.toml", ROOT / "config.example.toml") if p.exists())
    envfile.load_env(ROOT / ".env")
    s = settings.load(cfg)
    only = None if a.questions.lower() == "all" else [x.strip() for x in a.questions.split(",") if x.strip()]
    try:
        runner.run(s, a.run, dry=a.dry, only=only, source=runner.load_source(a.market_source, s) if a.market_source else None)
    except (ValueError, RuntimeError) as e:
        sys.exit(f"Not run: {e}")


if __name__ == "__main__":
    main()
