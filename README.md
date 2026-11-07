# Multi-agent forecasting model

A forecasting agent that uses several language models in different roles and anchors its answers on market prices. I built it for a live challenge on the night of the 2026 US midterms, and this is the general version: you bring your own questions, models, data and API key.

It handles two kinds of question:

- **Yes/no questions** ("Will X happen?"). The answer is a probability, as a whole percentage.
- **Move questions** ("How much will this price move by the next close?"). The answer is one number: percent, points or basis points.

The models do the research, give estimates and explain themselves. **The final number is always computed by code**, using fixed rules that are tested.

## How it works

### Yes/no questions

1. **Read the market.** Take the latest price from each prediction market and average them. This is the reference.
2. **Research.** A fast model with web search looks at each question from two angles: hard evidence (polls, data, expert ratings, base rates) and news (the last few days, resolution rules, scheduled events). After each pass it checks what is missing and searches only for that, up to a set number of rounds.
3. **Hide the market.** Any line that mentions prediction markets or betting is removed from the research. The forecasters must not see the market, otherwise the market would be counted twice later.
4. **Review.** A stronger model merges the research into one clean evidence sheet: latest dated values win, weak sources are flagged, news is ranked, conflicts are listed. It never gives a probability. If conflicts remain, one targeted search tries to settle them.
5. **Forecast.** Several models, ideally from different families, read the same sheet and each gives a probability. They work independently.
6. **Pool.** Code takes the median in log-odds, which is robust to one model going off. A supervisor model then checks the forecasts against the evidence and may change the result only if it cites a specific, dated fact with a link.
7. **Blend with the market.** `answer = w × market + (1 − w) × own estimate`. The weight `w` runs from 0.67 for the least-traded question to 0.80 for the most-traded, following the log of traded volume. Markets are hard to beat, so they keep most of the weight.
8. **Explain.** A model writes a short reason. If its reply changes the number, it is rejected.

I deliberately don't push the pooled probability away from 50% ("extremizing"). All forecasters read the same evidence, so their errors are correlated, and extremizing would turn shared noise into overconfidence.

### Move questions

The best single answer under absolute-error scoring is the **median** of the outcome distribution, so the agent forecasts a median:

```
answer = typical move × (drift + scenario shift)
```

- **Typical move:** the median absolute daily move over the last 60 trading days.
- **Drift:** measured on past comparable events (for example, the day after earlier elections). The code compares "zero" with "the median of the other events" by leaving one event out at a time, and keeps whichever would have scored better.
- **Scenario shift:** the outcomes of one to three yes/no questions (for example, who controls each chamber) define scenarios. The models say how the instrument would move in each one. The code subtracts the expected move, since prices already reflect it, and takes the median of a Student-t mixture weighted by the scenario probabilities from the same run.
- **Cross-check:** a model looks for inconsistencies between instruments (for example, shares up and volatility up). It is logged, and it never changes a number.

## Quick start

Python 3.11 or newer.

```bash
git clone https://github.com/LorenzoMonteduro/multi-agent-forecasting-model.git
cd multi-agent-forecasting-model
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run.py --dry                # offline demo: fake replies, no key, no cost
```

The demo uses a fictional country with synthetic prices. Results go to `runs/dry-demo/`.

To use real models:

```bash
cp config.example.toml config.toml     # set your provider, models and data paths
cp .env.example .env                   # put your API key here
python run.py --run my-run
python run.py --run test --questions D1,M1      # only some questions
```

`config.toml` and `.env` are git-ignored, so your key never goes into the repository.

## Configuration

Everything is in `config.toml` (see `config.example.toml`, which is commented):

- **Provider:** any endpoint that speaks the OpenAI chat-completions protocol. You set the address and the *name* of the environment variable that holds the key.
- **Web search:** how research calls turn on search, since providers differ. For example, OpenRouter uses `extra_body = { plugins = [ { id = "web" } ] }`.
- **Models:** one for research, several for forecasting, plus reviewer, supervisor and explainer. Leave one empty to skip that step.
- **Method:** research rounds, the market weight range, answer range, the terms used to filter market mentions.
- **Moves:** typical-move window, dates of past events, how scenarios are coupled.
- **Limits:** parallel workers, retries, and a cap on the number of calls.

Questions are a JSON file (`examples/demo/questions.json` shows the format). Market prices and price history are CSV files with the columns described below. For live data, write a small class with a `snapshot(question_id)` method and pass it with `--market-source module:Class`.

```
market_prices.csv:  question_id, venue, time_utc, price, bid, ask, volume
prices_daily.csv:   date, instrument, close
```

## Outputs

Each run writes to `runs/<name>/`:

- `answers.csv`: one row per question.
- `cards/`: one readable decision card per question, with the market, the evidence, each forecast, the arithmetic and the sources.
- `calls.jsonl`: every model call with its prompt, reply and token usage.
- `events.jsonl`: every step of the policy, with the settings and git commit of the run.

To score a finished run against what actually happened:

```bash
python evaluate.py --run runs/my-run --outcomes outcomes.csv
```

For yes/no questions it reports the gain over the market; for moves, the error in typical moves compared with simply forecasting zero.

## Tests

```bash
python -m pytest tests -q
```

The tests cover the probability maths, scenario calculations, drift, configuration checks, the market reader, key redaction in the logs, and full runs both offline and through a fake model client.

## Project layout

```
run.py, evaluate.py        command-line entry points
config.example.toml        all settings
forecaster/                the model: pipelines, prompts, maths, model gateway, reports
examples/demo/             fictional demo data
tests/                     test suite
```

## References

1. Alur, Stadie et al. (2025). *AIA Forecaster: Technical Report.* [arXiv:2511.07678](https://arxiv.org/abs/2511.07678)
2. Halawi, Zhang, Yueh-Han, Steinhardt (2024). *Approaching Human-Level Forecasting with Language Models.* [arXiv:2402.18563](https://arxiv.org/abs/2402.18563)
3. Schoenegger, Tuminauskaite, Park, Bastos, Tetlock (2024). *Wisdom of the Silicon Crowd.* [arXiv:2402.19379](https://arxiv.org/abs/2402.19379)
4. Neyman, Roughgarden (2022). *Are You Smarter Than a Random Expert?* [arXiv:2111.03153](https://arxiv.org/abs/2111.03153)
5. Lopez-Lira, Tang, Zhu (2025). *The Memorization Problem: Can We Trust LLMs' Economic Forecasts?* [arXiv:2504.14765](https://arxiv.org/abs/2504.14765)

Author: Lorenzo Monteduro
