"""Top-level orchestration: the two functions the model is built around.

  ingest_all()                 -- refresh every data pipeline that feeds the
                                   model and rebuild the unified feature table.
  predict_upcoming_world_cup() -- forecast scoreline + reliability for every
                                   upcoming FIFA World Cup fixture.

Scope note on "all data pipelines": results history, Elo, and the engineered
features are refreshed every run because they are exactly what the model fits
on and they change as new results come in. StatsBomb open event data is NOT
re-pulled here -- it's a static historical archive used to build priors
offline (see ingest/statsbomb.py); it has no new rows for a fixture that
hasn't been played yet, so re-fetching it on every run would just be a slow
no-op. Weather and lineup-availability ARE pulled fresh here, but only for the
small number of upcoming fixtures being forecast, since that's the only place
they're useful.
"""
from __future__ import annotations
import json
import pandas as pd

import config
from .ingest import results_history as rh
from .ingest import elo
from .ingest import weather
from .ingest import lineups
from . import features
from .model import ScorelineModel


def ingest_all() -> pd.DataFrame:
    """Refresh the results spine, recompute Elo, rebuild pre-match features.
    Writes data/processed/matches.parquet and returns the dataframe."""
    matches = rh.fetch()
    matches, _ratings = elo.compute(matches)
    matches = features.add_features(matches, form_window=config.FORM_WINDOW)
    matches.to_parquet(config.PROC / "matches.parquet", index=False)
    return matches


def _upcoming_world_cup(matches: pd.DataFrame) -> pd.DataFrame:
    """Exact-match on the tournament label -- 'FIFA World Cup qualification'
    and similar are deliberately excluded (confirmed against live data: the
    spine uses 'FIFA World Cup' for the tournament proper)."""
    is_wc = matches["tournament"] == "FIFA World Cup"
    unplayed = matches["home_score"].isna() | matches["away_score"].isna()
    return matches[is_wc & unplayed].sort_values("date").reset_index(drop=True)


def predict_upcoming_world_cup(matches: pd.DataFrame | None = None,
                               check_lineups: bool = True,
                               attach_weather: bool = True) -> pd.DataFrame:
    """Fit the Dixon-Coles model on the current training window and forecast
    every upcoming FIFA World Cup fixture. Returns one row per fixture with
    the scoreline, 1X2, reliability score, and its components.
    `matches` should already carry the [pre] feature columns from ingest_all();
    if omitted, the last saved matches.parquet is loaded."""
    if matches is None:
        matches = pd.read_parquet(config.PROC / "matches.parquet")
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])

    window = matches[matches["date"] >= pd.Timestamp(config.TRAIN_FROM)]
    played = window.dropna(subset=["home_score", "away_score"])
    model = ScorelineModel(xi=config.DIXON_COLES_XI, max_goals=config.MAX_GOALS).fit(played)

    fixtures = _upcoming_world_cup(matches)
    rows = []
    for r in fixtures.itertuples(index=False):
        date_str = r.date.date().isoformat()
        row = {
            "date": date_str, "home_team": r.home_team, "away_team": r.away_team,
            "venue_city": r.city, "venue_country": r.country, "neutral": bool(r.neutral),
        }

        lineup_known = False
        if check_lineups:
            lineup_known = lineups.check(r.home_team, r.away_team, date_str)

        try:
            pred = model.predict(r.home_team, r.away_team,
                                 neutral=bool(r.neutral), lineup_known=lineup_known)
        except KeyError as e:
            row.update({"status": f"skipped -- {e}"})
            rows.append(row)
            continue

        h, a = pred.scoreline
        row.update({
            "status": "ok",
            "predicted_score": f"{h}-{a}",
            "p_scoreline": round(pred.p_scoreline, 4),
            "exp_goals_home": pred.exp_goals[0], "exp_goals_away": pred.exp_goals[1],
            "p_home": round(pred.p_home, 4), "p_draw": round(pred.p_draw, 4),
            "p_away": round(pred.p_away, 4),
            "top_alt_scores": "; ".join(f"{s} ({p:.0%})" for s, p in pred.top_scores[1:4]),
            "reliability": pred.reliability,
            "reliability_components": json.dumps(pred.components),
        })
        if attach_weather:
            row["kickoff_temp_c"] = weather.temperature(r.city, date_str)
        rows.append(row)

    return pd.DataFrame(rows)
