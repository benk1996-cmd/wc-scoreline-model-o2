# World Cup scoreline model — free & open-source starter

A runnable pipeline that forecasts a match **scoreline** and an accompanying
**reliability score**, built entirely on free / keyless data. It ingests
international results, computes World Football Elo, engineers leak-free
pre-match features, and fits a recency-weighted, neutral-venue Dixon-Coles
model that emits a full score grid plus a transparent confidence number.

## Quick start

### Option 1: Interactive dashboard (recommended)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a browser dashboard at http://localhost:8501 with:
- All upcoming fixtures + predictions
- Filters by team, reliability threshold
- Interactive 1X2 and score probability charts
- Reliability breakdown per match

See [FRONTEND.md](FRONTEND.md) to deploy to the public web for free (Streamlit Cloud).

### Option 2: Command-line model (fastest)

```bash
pip install -r requirements.txt
python -m scripts.run_pipeline
```

Ingests all data, predicts every upcoming fixture, saves results to
`data/processed/world_cup_predictions.csv`.

### Option 3: REST API (for custom front-ends)

```bash
pip install -r requirements.txt flask flask-cors
python api.py
```

Then make JSON requests to `http://localhost:5000/api/predictions` or
`/api/predict/Brazil/Croatia`. See [api.py](api.py) and [FRONTEND.md](FRONTEND.md).

### One-off commands

```bash
python -m scripts.predict_match "Brazil" "Croatia"           # single-fixture
python -m scripts.build_dataset                             # ingest only
```

Sample output (run live, 28 fixtures currently upcoming in the 2026 tournament):

```
STEP 1/2 -- updating and ingesting data pipelines
results spine   : 49,477 matches (49,449 played, through 2026-06-27)

STEP 2/2 -- forecasting upcoming World Cup fixtures
28/28 fixtures forecast successfully -> data/processed/world_cup_predictions.csv

      date     home_team  away_team predicted_score  p_home  p_draw  p_away  reliability
2026-06-23      Portugal Uzbekistan             1-0  0.6717  0.2258  0.1025        0.462
2026-06-23      Colombia   DR Congo             1-0  0.5431  0.3026  0.1543        0.394
2026-06-23       England      Ghana             2-0  0.7432  0.1909  0.0659        0.518
2026-06-27        Jordan  Argentina             0-2  0.0309  0.1055  0.8636        0.631
...
```

Note the reliability spread: 0.34 for genuine toss-ups (e.g. Egypt vs Iran) up to
0.63 for heavily lopsided matchups (Jordan vs Argentina) — the score is reading
the matchup, not defaulting to a flat number.

## The one design rule

A feature is only allowed if it is **knowable before kickoff**. Anything derived
from the match itself (final xG, possession, shots) is training-only and must
never enter the inference feature set. `schema.py` tags every column `[pre]` or
`[post]`. This is why the rich event data is used for *priors*, not match-day
inputs — see below.

## Data sources (all free)

| Layer | Source | Access | Keyless |
|---|---|---|---|
| Results + fixtures (spine) | martj42/international_results | raw CSV | ✅ |
| Team strength | World Football Elo, computed from results | local | ✅ |
| Live WC fixtures/results | football-data.org v4 (free tier) | REST, `X-Auth-Token` | needs free key |
| Event data (touches/xG/360) | StatsBomb Open Data via `statsbombpy` | repo pull | ✅ |
| Weather | Open-Meteo (archive + forecast) | REST | ✅ |
| Lineups | API-Football (free tier) | REST | needs free key |

The keyless path (results → Elo → features → model) runs with no signup. The
two keyed feeds are optional enrichment; set `FOOTBALL_DATA_API_KEY` and
`API_FOOTBALL_KEY` in your shell to enable them.

> **Network note:** `weather.py` and `lineups.py` call `open-meteo.com` and
> `api-football.com` respectively. If you run this behind a restrictive
> egress firewall (e.g. a locked-down CI runner or sandbox), those calls will
> fail closed — `kickoff_temp_c` comes back empty and `lineup_known` stays
> `False` — rather than crashing the run. Add those hosts to your allowlist to
> enable them.

> Why compute Elo instead of scraping eloratings.net? Reproducible, keyless, and
> it yields the *pre-match* rating for every historical fixture for free.

> Why is event data only a prior? StatsBomb data doesn't exist for the fixture
> you're predicting, and the free tier of the live APIs doesn't carry shots/xG.
> So team xG profiles (`ingest/statsbomb.py`) inform strength priors rather than
> serving as match-day features.

## The reliability score

`model.py` combines three interpretable factors into a 0–1 number:

1. **outcome_confidence** — how peaked the 1X2 distribution is (`1 − normalised
   entropy`). A coin-flip match scores low.
2. **data_sufficiency** — whether both teams have enough recent matches.
3. **lineup_factor** — a small penalty until the confirmed XI is available.

It is a **starting heuristic, not a calibrated probability**. Before trusting
it, back-test: bucket predictions by reliability and check that higher buckets
really do score better on the Ranked Probability Score. `penaltyblog.metrics`
and `penaltyblog.backtest` give you RPS and a rolling back-test harness.

## Where to go next

- **Blend Elo / xG priors into the model.** Dixon-Coles here uses only
  goals+teams. Use `features.py` outputs (elo_diff, form, xG profiles) as a
  second model and ensemble, or move to a goals regression on those features.
- **Calibrate reliability** against held-out tournaments.
- **Add team-name aliases** in `normalize.py` as you connect new sources;
  `normalize.unmatched()` prints the misses.
- **Schedule `run_pipeline.py`** (cron / CI) during the tournament to keep
  predictions current as results land and new fixtures are confirmed.

## Layout

```
config.py                  paths, settings, env-based keys
README.md                  this file
FRONTEND.md                guide to running/deploying the dashboard
requirements.txt           dependencies (includes streamlit, plotly, etc.)
app.py                     Streamlit dashboard ← run with: streamlit run app.py
api.py                     Flask REST API ← run with: python api.py
wcmodel/
  normalize.py             team-name canonicalisation (the join glue)
  schema.py                canonical match table + [pre]/[post] tags
  features.py              leak-free pre-match feature builder
  model.py                 Dixon-Coles wrapper → scoreline + reliability
  pipeline.py              ingest_all() + predict_upcoming_world_cup()
  web.py                   data helpers for the dashboard (load, filter, parse)
  ingest/
    results_history.py     martj42 results (spine)
    elo.py                 World Football Elo from results
    weather.py             Open-Meteo (kickoff temp, upcoming fixtures only)
    fixtures.py            football-data.org v4
    statsbomb.py           StatsBomb event aggregates (priors, static archive)
    lineups.py             API-Football lineup availability (upcoming fixtures only)
scripts/
  run_pipeline.py          main entrypoint: ingest everything, forecast all upcoming WC fixtures
  build_dataset.py         ingest-only (no predictions) — the build step alone
  predict_match.py         CLI forecast for one specific fixture
```

### What gets refreshed each run, and why
`ingest_all()` re-pulls and rebuilds the results spine, Elo, and pre-match
features every time — that's what the model is actually fit on, and it changes
as new results come in. StatsBomb's open event data is **not** re-pulled on
every run: it's a static historical archive used to build team-strength priors
offline, and it has no new rows for a fixture that hasn't been played yet, so
re-fetching it every run would just be a slow no-op. Weather and lineup
availability **are** pulled fresh every run, but only for the handful of
upcoming fixtures being forecast, since that's the only place they add value.

Data is CC0/open; respect each provider's terms for redistribution.
