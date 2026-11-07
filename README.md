# Market-anchored forecasting agent

An automated forecasting agent that combines language-model research with market prices. It answers two kinds of question:

- **Binary questions** ("Will X happen?"). The answer is a probability, as a whole percentage, that the YES outcome happens. The agent forms its own estimate without seeing the market, then blends it with prediction-market prices.
- **Move questions** ("How much will instrument I move between two closes?"). The answer is one number in the instrument's unit (percent, points or basis points). The agent combines statistics from comparable past events with scenario forecasts tied to the binary questions.

I first built it for a live forecasting challenge on election night of the 2026 US midterms: closest Senate and House races, control of each chamber, and next-day moves of stocks, bonds and commodities. This repository is a general version. The questions, models, provider, data and every number in the method are set in two files, and nothing is tied to one competition, provider or API.

The language models supply evidence, estimates and explanations. **Every final number is computed by code**, from rules that are written down and tested.

---

## Contents

1. [How it works: binary questions](#1-how-it-works-binary-questions)
2. [How it works: move questions](#2-how-it-works-move-questions)
3. [Design principles](#3-design-principles)
4. [Quick start](#4-quick-start)
5. [Configuration](#5-configuration-configtoml)
6. [Question file](#6-question-file)
7. [Data files and live data](#7-data-files-and-live-data)
8. [Outputs](#8-outputs)
9. [Scoring a run](#9-scoring-a-run)
10. [Tests](#10-tests)
11. [Cost, limits and known weaknesses](#11-cost-limits-and-known-weaknesses)
12. [Project layout](#12-project-layout)
13. [References](#13-references)

---

## 1. How it works: binary questions

For each binary question, `forecaster/binary.py` runs these steps:

| # | Step | Who | What happens | Why |
|---|---|---|---|---|
| 0 | Market snapshot | code | Reads the latest price on each venue (the bid-ask midpoint when both are available). The market reference **m** is the mean across venues, and the traded volume is summed. | This is the benchmark the answer is compared with. |
| 1 | Background briefing | research model, web search | One call per run on the common background shared by all questions. | Gives every forecast the same national or general picture. |
| 2 | Research angles | research model, web search | Two angles per question run in parallel: **structured evidence** (polls, data, expert ratings, published models, base rates) and **news, rules and timing**. You can change or add angles in the question file. | Different angles find different evidence. Retrieval quality drives forecasting accuracy (Halawi et al. 2024; AIA Forecaster 2025). |
| 3 | Gap-driven follow-ups | research model | For each angle, a call without search reads the dossier, says whether something important is missing and proposes at most 3 queries. A follow-up search covers only those queries. This repeats up to `research_rounds` times and stops early when the dossier is enough. | Iterative, agentic search beats a single pass (AIA Forecaster 2025). The gap check is limited to its own angle so the two dossiers do not duplicate each other. |
| 4 | Market filter | code | Deletes every line that mentions a prediction market or betting (configurable list of terms). | The forecasters must not see the market, or the blend in step 9 would count the market twice. |
| 5 | Evidence review | reviewer model, no search | Merges the dossiers into one evidence sheet: the latest dated value when sources conflict, a data table with weak items flagged, at most 10 news items ranked by importance and date, a list of conflicts, and claims without a date or source. It gives no probability. | A fast search model writes long, overlapping and sometimes contradictory text, and the reviewer cleans it. Keeping the reviewer away from probabilities keeps the forecasters independent. |
| 6 | Conflict search | research model, web search | When the reviewer lists conflicts that dates cannot settle, one targeted search runs and its result is appended as a CONFLICT CHECK section. | Settles disagreements with evidence instead of leaving them to each forecaster. |
| 7 | Independent forecasts | several models, ideally from different families | Each model reads the same evidence sheet and briefing, follows a base-rate → evidence → news method and returns `p_yes` (1 to 99). The market is hidden. | Ensembles of models are more accurate than one model (Schoenegger et al. 2024; AIA Forecaster 2025). |
| 8 | Pooling and supervisor | code, then supervisor model | Code takes the **median in log-odds** of the forecasts. A supervisor then reviews the forecasts against the evidence and may replace the pooled value **only** if it cites a concrete, dated fact with a link that a forecaster missed. Otherwise the pooled value stands. | The median is robust to one outlying model. The citation rule stops the supervisor from overriding on opinion (AIA Forecaster 2025). |
| 9 | Blend with the market | code | answer = round( w · m + (1 − w) · own ), clipped to [`answer_lo`, `answer_hi`]. | Market prices are hard to beat, so they keep most of the weight even against strong systems (Halawi et al. 2024; AIA Forecaster 2025). |
| 10 | Explanation | explainer model | Writes 2 to 4 sentences on why the number is reasonable. A reply that changes the number is rejected. | Every answer comes with a readable reason, and the number stays the code's. |

**Market weight.** The weight **w** rises with liquidity. It is linear in the logarithm of traded volume, from `w_market_lo` for the least-traded question of the run to `w_market_hi` for the most-traded:

```
w(q) = w_lo + (w_hi − w_lo) · (ln V(q) − ln V_min) / (ln V_max − ln V_min)
```

The defaults are 0.67 (the market weight on liquid markets in the AIA Forecaster report, Table 7) and 0.80 (the 4:1 crowd weight in Halawi et al. 2024). A question with no reported volume gets `w_lo`.

**No extremizing.** Some systems push pooled probabilities away from 50% (for example multiplying the log-odds by √3; Neyman and Roughgarden 2022). That is justified when forecasters hold different information. Here every forecaster reads the same evidence sheet, so their errors are correlated and extremizing would turn shared noise into confident bets. In my tests it also moved answers to the other side of the market from the forecasters themselves.

**Fallback.** If no forecast is valid, or the pipeline fails, the answer is the market price rounded (zero expected gain over the market), and the decision card says so.

## 2. How it works: move questions

The target is the move from one close to the next, for example from the close on election day to the close on the day after. Many scoring rules for such questions use **absolute error divided by a typical move**. For absolute error the best single number is the **median** of the forecast distribution, so the agent forecasts a median:

```
answer = typical move × (drift + scenario shift)        (drift and shift in typical-move units)
```

`forecaster/moves.py` and `forecaster/calibration.py` compute each part:

1. **Typical move.** The median absolute daily move over the last `typical_window` trading days (default 60) up to `typical_end`.
2. **Drift, from past events.** For every date in `past_events` the code measures the next-day move in typical-move units, each event with its own typical move (the window ends `event_window_lag_bdays` business days before that event). It then compares two candidate drifts, **zero** and the **median of the other events**, by a leave-one-event-out score, and keeps the one that would have scored better out of sample. Ties go to zero. Statistics only, with no language model.
3. **Scenarios.** The outcomes of 1 to 3 binary questions (`scenario_questions`) define every combination of YES and NO. Their probabilities come from this run's answers to those questions. Two couplings are available:
   - `comonotone` (default): one common swing drives every question, so the less likely YES happens only when the likelier one does. This suits outcomes driven by the same national mood, such as control of two chambers.
   - `independent`: product of the probabilities.
4. **Scenario moves.** Two or more models get the instrument's recent data, its past event-day moves, a research dossier and the briefing. Each returns the move it expects **in each scenario**, in typical-move units, bounded by the largest past event move.
5. **Demeaning.** Prices already reflect the expected outcome before it is known, so only the surprise moves them. The code subtracts the probability-weighted mean from the scenario moves.
6. **Mixture median.** Each scenario is a Student-t distribution centred on its demeaned move. The scale is set so that one typical move is the median absolute value, and the degrees of freedom are fitted by maximum likelihood on the price history (fat tails, about 3 to 5 for daily data). The answer uses the median of the probability-weighted mixture.
7. **Cross-instrument check.** A model checks whether the proposed moves are consistent with each other, for example equities and their volatility index usually moving in opposite directions. The result is logged and shown to the explainer, but it never changes a number.
8. **Explanation** as for binary questions. **Fallback**: drift only.

## 3. Design principles

- **Numbers come from code.** Models research, estimate and explain, and the policy that turns their output into an answer is deterministic and unit-tested.
- **Forecasters never see the market**, so their estimate is an independent signal and the blend does not count the market twice.
- **All data is read before the first model call**, so a data problem never leaves a half-finished run.
- **Every call is logged** as an append-only JSON line: prompt, reply, model, attempts, errors, time, token usage and the links cited. Every step of the policy is logged as an event. The API key is redacted from all logs.
- **Every question ends with an answer**, through fallback models per step, a cap on attempts and a policy fallback.
- **No back-testing of language-model forecasts** on events before the models' training cutoff: the models may remember the outcome (Lopez-Lira et al. 2025). Only the statistical parts (drift, degrees of freedom) are estimated on past events.

## 4. Quick start

Requires Python 3.11 or newer.

```bash
git clone https://github.com/<your-user>/<this-repo>.git
cd <this-repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --dry                # offline demo: fake model replies, no key, no cost
```

The dry run uses the **fictional** demo in `examples/demo/` (an invented country and instruments, with synthetic prices made by `examples/make_demo_data.py`). The answers are written to `runs/dry-demo/answers.csv` and the decision cards to `runs/dry-demo/cards/`.

To run with real models:

```bash
cp config.example.toml config.toml     # then set provider, models and paths
cp .env.example .env                   # then put your API key in it
python run.py --run my-first-run
python run.py --run test --questions D1,M1      # only some questions
```

`config.toml` and `.env` are git-ignored, so your key and your local settings never go into the repository.

## 5. Configuration (`config.toml`)

Every key is optional. Missing keys use the defaults in `forecaster/settings.py`, and an unknown key is an error, so a typo cannot pass silently.

| Section | Key | Default | Meaning |
|---|---|---|---|
| `provider` | `base_url` | `https://api.openai.com/v1` | Any endpoint that speaks the OpenAI chat-completions protocol: a model provider, a gateway, or a local server. |
| | `api_key_env` | `LLM_API_KEY` | **Name** of the environment variable that holds the key. |
| | `send_temperature`, `no_temperature_models` | `true`, `[]` | Some reasoning models reject a temperature: list their ids or prefixes. |
| | `extra_body` | `{}` | Extra request fields your provider needs. |
| `web_search` | `mode` | `none` | `none` (models use their own knowledge), `extra_body` (merge a provider-specific search payload into research calls) or `suffix` (append a suffix to the model id). |
| `models` | `research`, `forecasters`, `move_forecasters`, `reviewer`, `supervisor`, `explainer` (+ fallbacks) | – | Model ids exactly as your provider names them. A fast, cheap model is enough for research. Use several families for the forecasters. Leave the reviewer, supervisor or explainer empty to skip that step. |
| `method` | `context` | – | One paragraph of background inserted in every prompt. |
| | `research_rounds` | 2 | Follow-up search rounds per angle (0 = one pass). |
| | `w_market_lo`, `w_market_hi` | 0.67, 0.80 | Market weight range (section 1). |
| | `answer_lo`, `answer_hi` | 1, 99 | Range of binary answers. |
| | `market_terms` | betting and market words | Regular expressions for the market filter. |
| | `odds_max_age_h` | 24 | Older quotes are flagged as stale in the cards. |
| | `explain_final` | true | One explanation call per question. |
| | `cross_check_notes` | – | Expected relations between instruments, for the cross-check. |
| `moves` | `typical_window`, `typical_end` | 60, last date | Typical-move window. |
| | `past_events` | `[]` | Dates (YYYY-MM-DD) of comparable past events, used for the drift. |
| | `event_window_lag_bdays` | 2 | Business-day gap between a past event and the end of its typical-move window. |
| | `scenario_coupling` | `comonotone` | Or `independent`. |
| `limits` | `workers`, `call_attempts`, `step_deadline_s` | 8, 2, 300 | Parallelism, attempts per model, time per step. |
| | `max_calls` | 2000 | Kill switch: beyond it, optional calls (follow-ups, supervisor, explanations) are skipped. |
| | `max_tokens`, `temperature` | per role | Output caps (a cut reply is retried with twice the cap) and temperatures. |
| `paths` | `questions`, `market_prices`, `price_history`, `runs` | demo files | Relative to the config file. |

**Web search.** Search quality matters more than almost any other choice (AIA Forecaster 2025), and `mode = "none"` is much weaker for current events. Providers expose search in different ways, so check your provider's documentation. For example, OpenRouter (a gateway to many model families) uses `mode = "extra_body"` with `extra_body = { plugins = [ { id = "web" } ] }`. Links that the provider returns as citations, and links written in the reply, are captured in the logs and cards.

## 6. Question file

JSON; see `examples/demo/questions.json`.

```json
{
  "briefing_topic": "Common background for all binary questions (optional)",
  "binary": [
    {"id": "UPPER", "short": "Upper House",
     "question": "Will the Green Party control the Upper House?",
     "yes_means": "the Green Party holds at least 31 of 60 seats after the election",
     "notes": "facts the forecasters must not get wrong (rules, thresholds, candidates)"}
  ],
  "scenario_questions": ["LOWER", "UPPER"],
  "move_window": "from the close on election day to the close on the next trading day",
  "move_briefing_topic": "Background for all move questions (optional)",
  "moves": [
    {"id": "M1", "label": "Equity index", "instrument": "EQUITY", "unit": "pct", "decimals": 2, "limit": 20}
  ],
  "research_angles": {"evidence": "...", "news": "..."},
  "gap_scopes": {"evidence": "...", "news": "..."}
}
```

- `unit`: `pct` (percentage change of the close), `points` (difference of the close, for example a volatility index) or `bp` (difference × 100, for yields quoted in percent).
- `limit`: answers are clamped to ± limit.
- `research_angles` and `gap_scopes` are optional. Without them the two default angles in `forecaster/prompts.py` are used. Use them to tailor research to a domain, for example naming the poll aggregators and race raters for an election.

## 7. Data files and live data

**Market prices** (`paths.market_prices`): one row per quote. Several rows per question and venue are fine: the latest is used, and the one 7 days earlier gives the weekly change shown in the cards.

```
question_id,venue,time_utc,price,bid,ask,volume
UPPER,venue_a,2026-11-03T20:00:00+00:00,,0.44,0.46,400000
UPPER,venue_b,2026-11-03T20:00:00+00:00,0.45,,,700000
```

Prices are probabilities of YES in [0, 1]. Volume is any measure of recent trading, in the same unit for every question, because only ratios matter.

**Price history** (`paths.price_history`): daily closes.

```
date,instrument,close
2026-11-02,EQUITY,5123.4
```

**Plugging in live market data.** Write a class with a `snapshot(qid)` method that returns a `forecaster.markets.MarketSnapshot`, and pass it on the command line:

```python
# mysources.py
from forecaster.markets import MarketSnapshot, Venue

class MyLiveSource:
    def __init__(self, settings):
        ...                                   # set up clients, read your own keys from the environment

    def snapshot(self, qid: str) -> MarketSnapshot:
        price, volume, time_utc = ...         # fetch from your venues' public APIs
        return MarketSnapshot(qid, [Venue("venue_a", price, volume=volume, time_utc=time_utc)],
                              market=price, volume=volume, time_utc=time_utc)
```

```bash
python run.py --run live-1 --market-source mysources:MyLiveSource
```

Venue APIs change often, so this repository ships no live connectors. Check each venue's current documentation.

## 8. Outputs

Each run writes to `runs/<run name>/` (or `runs/dry-<name>/` for dry runs):

| File | Content |
|---|---|
| `answers.csv` | One row per question: answer, unit, market price or typical move, own estimate, and whether it came from the policy or a fallback. |
| `cards/index.md`, `cards/<id>.md` | One decision card per question: market, evidence review, each forecast, pooling, supervisor, policy arithmetic, the explanation and every source link. |
| `calls.jsonl` | Every model call: prompt, reply, model, attempt, errors, seconds, token usage, links. |
| `events.jsonl` | Every step: run start (settings, git commit, market weights, Student-t degrees of freedom), market snapshots, move data, aggregates, policies, finals, errors and a run summary. |
| `state/<id>.json` | The full state of each question. |

`python -m forecaster.calibration --config config.toml` prints the drift table and degrees of freedom without calling any model.

## 9. Scoring a run

```bash
python evaluate.py --run runs/my-run --outcomes outcomes.csv
```

`outcomes.csv` has the columns `question_id,outcome`: 1 or 0 for binary questions (leave it empty if unresolved) and the actual move for move questions. It reports:

- **Binary:** the gain over the market, [(m − outcome)² − (answer − outcome)²] × 100. Positive means the answer beat the market. Copying the market scores 0.
- **Moves:** min(|answer − actual| / typical move, 5), lower is better, next to the same score for a forecast of zero.

## 10. Tests

```bash
python -m pytest tests -q
```

The tests cover every function that touches a probability or a move (log-odds, pooling, blend, weights, scenario probabilities, mixture median, drift, degrees of freedom, units), the settings and question-file validation, the CSV market source, the request built for each web-search mode, key redaction in logs, the call cap, an offline end-to-end run, and an end-to-end run through a fake chat-completions client. That fake client checks that the forecasters never see market lines.

## 11. Cost, limits and known weaknesses

- **Calls per run.** With the defaults, a binary question needs about 10 to 20 calls (2 angles × up to 3 searches, gap checks, review, an optional conflict search, the forecasters, the supervisor and the explanation). A move question needs about 4. Web-search calls usually cost several times more than plain calls. Run a few questions with `--questions` before a full run, and use `max_calls` as a kill switch.
- **Shared evidence.** All forecasters read the same evidence sheet. The ensemble removes model noise, not research errors.
- **Lean towards YES.** Language models tend to forecast YES too often (Schoenegger et al. 2024). The market weight, at least 0.67 by default, limits any such bias to a third of its size.
- **Correlated outcomes.** Binary questions driven by one common factor, such as a national swing across many races, can all miss in the same direction. The market blend limits the damage, but nothing in the method diversifies it away.
- **Scenario moves come from language models.** They are bounded and demeaned, but they are judgments, not estimates. The drift is the only part fitted on data.
- **Comonotone coupling** is a modelling assumption, and `independent` is the other extreme. The truth usually lies between them.

## 12. Project layout

```
run.py                     command line: one run
evaluate.py                score a run against outcomes
config.example.toml        all settings, commented
.env.example               where the API key goes (copy to .env)
forecaster/
  settings.py              settings schema and validation
  questions.py             question file schema
  llm.py                   model gateway: OpenAI-compatible client, web search, retries, fallbacks, logs, dry mode
  markets.py               market snapshots (CSV source + interface for live sources)
  prices.py                price history, moves, typical move
  calibration.py           drift (leave-one-event-out), Student-t degrees of freedom
  scenarios.py             scenario probabilities, demeaning, mixture median
  mathx.py                 log-odds, pooling, blend, liquidity weights, scoring
  prompts.py               every prompt
  binary.py                pipeline for binary questions
  moves.py                 pipeline for move questions
  runner.py                orchestration of a run
  report.py                answers.csv and decision cards
examples/
  make_demo_data.py        builds the fictional demo data
  demo/                    questions.json, market_prices.csv, prices_daily.csv (all fictional)
tests/                     pytest suite
```

## 13. References

1. Alur, R., Stadie, B. et al. (2025). *AIA Forecaster: Technical Report.* [arXiv:2511.07678](https://arxiv.org/abs/2511.07678)
2. Halawi, D., Zhang, F., Yueh-Han, C., Steinhardt, J. (2024). *Approaching Human-Level Forecasting with Language Models.* [arXiv:2402.18563](https://arxiv.org/abs/2402.18563)
3. Neyman, E., Roughgarden, T. (2022). *Are You Smarter Than a Random Expert? The Robust Aggregation of Substitutable Signals.* [arXiv:2111.03153](https://arxiv.org/abs/2111.03153)
4. Lopez-Lira, A., Tang, Y., Zhu, M. (2025). *The Memorization Problem: Can We Trust LLMs' Economic Forecasts?* [arXiv:2504.14765](https://arxiv.org/abs/2504.14765)
5. Schoenegger, P., Tuminauskaite, I., Park, P. S., Bastos, R. V. S., Tetlock, P. E. (2024). *Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy.* [arXiv:2402.19379](https://arxiv.org/abs/2402.19379)

---

Author: Lorenzo Monteduro.
