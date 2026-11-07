"""Every prompt the agent sends. Placeholders in {braces}; literal braces are doubled.
`context` (settings.method.context) describes the setting in one paragraph and is inserted everywhere."""
import json

END = "End your reply with ONE line of JSON and nothing after it:"
COMMON = (
    "Context: {context}\nCurrent time: {now} UTC. Be factual, date every fact you report, and say 'not found' "
    "rather than guessing. Never invent numbers, polls or sources."
)
NO_MARKETS = "Do NOT report prediction-market or betting odds (any exchange, bookmaker or betting site)."

# Research angles for binary questions: name -> what to look for. Override per question file ("research_angles").
RESEARCH_ANGLES = {
    "evidence": "STRUCTURED EVIDENCE. Report, with dates, source, sample size and method where available: polls or "
                "surveys and their averages, official data, expert ratings and published model forecasts with their "
                "probabilities, base rates from comparable past cases, and the fundamentals of the case.",
    "news": "NEWS, RULES AND TIMING (not polls). Report, with dates: news of the last 7 days, especially the last 72 hours; "
            "the exact resolution rule and anything unusual about it; scheduled events before resolution; "
            "early indicators already observable (for example early turnout, preliminary data).",
}
GAP_SCOPES = {
    "evidence": "Judge only the structured evidence (polls, averages, data, ratings, models, base rates, fundamentals). "
                "News, rules and timing are covered by another dossier: do not ask for them.",
    "news": "Judge only news of the last 7 days, the resolution rule, scheduled events and early indicators. "
            "Polls, ratings and models are covered by another dossier: do not ask for them.",
}

BRIEFING = COMMON + """

TASK: Using web search, write a briefing on the COMMON BACKGROUND shared by these questions:
{topic}
Cover, with dates and source URLs: the current state of play, the main drivers, notable news of the last 72 hours
that could affect many questions at once, and how large forecasting errors have historically been in similar
cases. """ + NO_MARKETS + """ End with the list of URLs you used."""

RESEARCH = COMMON + """

TASK: Research this question using web search: {question}
YES means: {yes_means}
Angle: {angle_text}
""" + NO_MARKETS + """ End with the list of URLs you used."""

GAPS = COMMON + """

QUESTION: {question}
Angle: {angle}
DOSSIER SO FAR:
{dossier}

TASK: You are checking a research dossier, without searching. {scope}
Within that scope only, is anything important missing or out of date for forecasting the outcome? If the dossier
is enough, say so. Otherwise give at most 3 specific search queries that would fill the gaps. """ + END + """
{{"enough": <true or false>, "missing": "<one sentence>", "queries": ["<query>", "<query>"]}}"""

FOLLOWUP = COMMON + """

TASK: Research this question using web search: {question}
Angle: {angle}. A first dossier already exists; do NOT repeat it. Find only what is missing:
{missing}
Suggested searches: {queries}
Report, with dates and sources, only NEW facts. """ + NO_MARKETS + """
If nothing new is found, say 'nothing new'. End with the list of URLs you used."""

REVIEW = COMMON + """

QUESTION: {question}
YES means: {yes_means}
{notes}

You receive raw research dossiers written by a fast search model (with any follow-up searches). They may overlap,
contradict each other, contain undated or weakly sourced claims, and evidence of very different quality.

{dossiers}

TASK: You are the evidence reviewer. Do NOT search, do NOT give any probability or verdict, and do NOT add facts
that are not in the dossiers. Write one clean EVIDENCE SHEET with these sections:

FUNDAMENTALS: the facts of the case, base rates, the resolution rule.
RATINGS AND MODELS: each expert rating or published model once, with its LATEST dated value (older value in brackets).
DATA: a table of polls, surveys or data points, newest first: date | source | sponsor or method | sample | result.
  Then one line naming the items that deserve less weight and why (small sample, partisan sponsor, old, unclear method).
NEWS: at most 10 items, ranked by importance, then by date (newest first). Each item: date | importance (HIGH, MEDIUM,
  LOW) | one-line fact | source | whether it came before or after the latest quality data point. Undated items last.
CONFLICTS: every disagreement you found and how you resolved it (the most recent dated source wins). Write
  UNRESOLVED for conflicts that dates cannot settle.
WEAK CLAIMS: claims with no date or no specific source.
Never mention prediction-market or betting odds; drop them.
""" + END + """
{{"unresolved": ["<conflict that dates cannot settle>"], "queries": ["<search query that would settle it>"]}}"""

CONFLICT = COMMON + """

TASK: Using web search, settle these open questions about: {question}
{unresolved}
Suggested searches: {queries}
Report only the facts that settle them, each with its date and source URL. """ + NO_MARKETS + """
If you cannot settle one, say 'not settled'."""

FORECAST = COMMON + """

QUESTION: {question}
YES means: {yes_means}
{notes}
Give the probability (1-99) that the outcome is YES.

{evidence}

BACKGROUND BRIEFING:
{briefing}

Method (write it out briefly):
1. Start from a base rate for cases like this one.
2. Update on the evidence, weighting recent, high-quality and well-sourced items more. If the CONFLICT CHECK section
   contradicts the sheet above it, the CONFLICT CHECK wins (it is newer).
3. Adjust only for concrete, dated news. Ignore any betting or prediction-market odds you may see.
4. State the probability. Avoid both overconfidence and hedging to 50.
""" + END + """
{{"p_yes": <integer 1-99>, "key_evidence": "<one sentence>"}}"""

SUPERVISE = COMMON + """

QUESTION: {question}
YES means: {yes_means}
Independent forecasts (they did not see any market price; range {spread:.0f} points):
{forecasts}

EVIDENCE THE FORECASTERS READ:
{evidence}

TASK: You are the supervisor. Check the forecasts against the evidence, find the specific fact(s) behind any
disagreement and decide which forecast is better supported. You may search the web. You may propose a different
probability ONLY if you cite a concrete, dated fact (with URL) that at least one forecaster missed or misread;
otherwise return null. """ + END + """
{{"adjusted_p_yes": <integer 1-99 or null>, "new_fact": "<fact or empty>", "source": "<URL or empty>", "reason": "<one sentence>"}}"""

EXPLAIN_BINARY = COMMON + """

QUESTION: {question}

OUR OWN ESTIMATE (independent of the market): {own_int}%
- forecasters (market hidden from them): {forecasts}
- pooled (log-odds median, after the supervisor): {pooled:.1f}%{supervisor}

MARKET REFERENCE (mean of {venues}): {market:.1f}%. Notes: {market_notes}

ANSWER (computed by code): {proposal}%
Rule: weighted average {w_market:.1%} market + {w_model:.1%} own estimate = {blend:.1f}%, rounded. The market weight
rises with the question's traded volume, from {w_lo:.0%} (least traded) to {w_hi:.0%} (most traded).

TASK: The answer is {proposal}; do not change it. Explain in 2-4 sentences what drives our own estimate, how it
compares with the market and why the weighted answer is reasonable. """ + END + """
{{"answer": {proposal}, "reason": "<one sentence>"}}"""

MOVE_BRIEFING = COMMON + """

TASK: Using web search, write a briefing for forecasting the move of several instruments {window}:
{topic}
Cover, with dates and source URLs: scheduled events in that window (data releases, central bank meetings, major
company news); what analysts and options markets expect, including the expected size of the move; which outcomes
are seen as positive or negative for each kind of instrument; and major other drivers this week.
End with the list of URLs you used."""

MOVE_RESEARCH = COMMON + """

TASK: Research, using web search, what could move the {label} {window}. Report with dates and URLs:
- how analysts expect it to react to each scenario outcome below, including any options-implied move;
- scheduled events in that window relevant to it, and other news of the last 72 hours;
- its recent trend and level.
Scenarios: {scenario_names}
Be concise. End with the list of URLs you used."""

MOVE_FORECAST = COMMON + """

INSTRUMENT: {label} ({qid}). Forecast target: its move {window}, in {unit}.
Typical daily move (median absolute daily move over the last {typical_window} trading days): {typical} {unit}.
Recent data: {recent}
Moves on the day after comparable past events ({unit}): {history}

SCENARIOS (probabilities from our own forecasts of the questions that decide them):
{scenarios}

RESEARCH (instrument):
{dossier}

BRIEFING:
{briefing}

TASK: For EACH scenario, the move you would expect if that scenario happened, in TYPICAL-MOVE UNITS
(1.0 = {typical} {unit}), between -{bound} and {bound} (the largest day-after move of the past events). Think about
the direction first, keep sizes modest unless the scenario is a real surprise, and include scheduled non-event
news from the briefing. """ + END + """
{{"scenario_moves": {template}, "reason": "<one sentence>"}}"""

CROSS_CHECK = COMMON + """

TASK: Cross-instrument consistency check of our proposed moves (answer units and typical-move units):
{table}
{notes}
List the instruments whose proposed move looks inconsistent with the others (this check is logged; it does not
change the numbers). """ + END + """
{{"flags": ["<question id>: <issue>"], "reason": "<one sentence>"}}"""

EXPLAIN_MOVE = COMMON + """

INSTRUMENT: {label} ({qid}). Move {window}, in {unit}. Typical daily move: {typical} {unit}.
The best single number for absolute-error scoring is the MEDIAN of the outcome distribution.

DECOMPOSITION (computed by code, in typical-move units):
- drift: {drift_u:+.2f} ({drift_note})
- scenario shift (median of the probability-weighted scenario mixture with Student-t noise, demeaned because prices
  already reflect the expected outcome): {scen_u:+.2f}; forecasters: {scen_detail}
- cross-instrument check: {cross_note}
= total {total_u:+.2f} typical moves = ANSWER {proposal} {unit}

TASK: The answer is {proposal}; do not change it. Explain in 2-3 sentences what drives it. """ + END + """
{{"answer": {proposal}, "reason": "<one sentence>"}}"""


def jdump(x: object) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)
