"""Score a finished run against what actually happened.

    python evaluate.py --run runs/my-run --outcomes outcomes.csv

outcomes.csv columns: question_id, outcome
  binary questions: outcome = 1 (YES) or 0 (NO); missing or empty = not resolved yet (skipped)
  move questions:   outcome = the actual move, in the question's unit

Binary score: gain over the market = [(market - outcome)^2 - (answer - outcome)^2] x 100 (positive = beat the market).
Move score:   min(|answer - actual| / typical move, 5) (lower is better), also shown for a forecast of zero.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecaster.calibration import SCORE_CAP  # noqa: E402
from forecaster.mathx import gain_vs_market  # noqa: E402
from forecaster.report import load_states  # noqa: E402


def evaluate(run_dir: Path, outcomes: dict[str, float]) -> dict:
    rows, gains, errs, zero = [], [], [], []
    for q, st in load_states(run_dir).items():
        if q not in outcomes:
            continue
        o, a = outcomes[q], (st.get("final") or {}).get("answer")
        if a is None:
            continue
        if st.get("kind") == "binary":
            g = gain_vs_market(a / 100, st["market"]["market"], int(o))
            gains.append(g)
            rows.append((q, "binary", a, round(100 * st["market"]["market"], 1), o, round(g, 3)))
        else:
            t = st["typical"]
            e, z = min(abs(a - o) / t, SCORE_CAP), min(abs(o) / t, SCORE_CAP)
            errs.append(e)
            zero.append(z)
            rows.append((q, "move", a, round(t, 4), o, round(e, 3)))
    return {"rows": rows, "binary_mean_gain": sum(gains) / len(gains) if gains else None, "binary_n": len(gains),
            "move_mean_error": sum(errs) / len(errs) if errs else None, "move_zero_error": sum(zero) / len(zero) if zero else None,
            "move_n": len(errs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder, for example runs/my-run")
    ap.add_argument("--outcomes", required=True)
    a = ap.parse_args()
    with open(a.outcomes, encoding="utf-8") as f:
        out = {r["question_id"]: float(r["outcome"]) for r in csv.DictReader(f) if (r.get("outcome") or "").strip()}
    res = evaluate(Path(a.run), out)
    print(f"{'question':10} {'kind':6} {'answer':>8} {'market/typical':>15} {'outcome':>8} {'score':>8}")
    for r in res["rows"]:
        print(f"{r[0]:10} {r[1]:6} {r[2]:>8} {r[3]:>15} {r[4]:>8} {r[5]:>8}")
    if res["binary_n"]:
        print(f"binary: mean gain over the market {res['binary_mean_gain']:+.3f} on {res['binary_n']} resolved questions")
    if res["move_n"]:
        print(f"moves: mean error {res['move_mean_error']:.3f} typical moves (zero forecast: {res['move_zero_error']:.3f}) "
              f"on {res['move_n']} questions")


if __name__ == "__main__":
    main()
