"""Data fetching and prep for the Streamlit dashboard.

Decouples the UI from the model's pipeline logic — the dashboard calls these,
not the pipeline functions directly.
"""
from __future__ import annotations
import json
import datetime as _dt
import pandas as pd
import numpy as np

import config

PRED_PATH = config.PROC / "world_cup_predictions.csv"


def load_static_predictions() -> pd.DataFrame | None:
    """FAST. Read the last saved predictions from disk. Never runs the pipeline.
    Returns None if no predictions have been generated yet."""
    if not PRED_PATH.exists():
        return None
    df = pd.read_csv(PRED_PATH)
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    return df


def last_updated() -> str | None:
    """When the saved predictions file was last written (local time), or None."""
    if not PRED_PATH.exists():
        return None
    ts = _dt.datetime.fromtimestamp(PRED_PATH.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")


def refresh_predictions(attach_weather: bool = True) -> pd.DataFrame:
    """SLOW (~30s). Re-ingest all data pipelines, recompute Elo + features,
    re-predict every upcoming fixture, and overwrite the saved CSV.
    Only call this from an explicit user action, never on page load."""
    # imported here so a plain static load never pays the pipeline import cost
    from .pipeline import ingest_all, predict_upcoming_world_cup
    matches = ingest_all()
    preds = predict_upcoming_world_cup(matches, check_lineups=False,
                                       attach_weather=attach_weather)
    # predict_upcoming_world_cup already writes the CSV via run_pipeline; ensure
    # the file exists here too in case this is called directly.
    preds.to_csv(PRED_PATH, index=False)
    return preds[preds["status"] == "ok"].copy()


def load_or_refresh_predictions(force_refresh: bool = False) -> pd.DataFrame:
    """Back-compat shim. Prefer load_static_predictions() / refresh_predictions()."""
    if force_refresh:
        return refresh_predictions()
    df = load_static_predictions()
    return df if df is not None else refresh_predictions()


def parse_reliability_components(json_str: str) -> dict:
    """Parse the JSON reliability breakdown."""
    try:
        return json.loads(json_str)
    except (ValueError, TypeError):
        return {}


def get_fixture_details(preds: pd.DataFrame, idx: int) -> dict:
    """Extract full details for a single fixture by row index."""
    if idx >= len(preds):
        return {}
    r = preds.iloc[idx]
    return {
        "date": r["date"],
        "home": r["home_team"],
        "away": r["away_team"],
        "venue": f"{r['venue_city']}, {r['venue_country']}",
        "neutral": bool(r["neutral"]),
        "predicted_score": r["predicted_score"],
        "p_scoreline": r["p_scoreline"],
        "exp_goals": (r["exp_goals_home"], r["exp_goals_away"]),
        "p_1x2": (r["p_home"], r["p_draw"], r["p_away"]),
        "reliability": r["reliability"],
        "components": parse_reliability_components(r.get("reliability_components", "{}")),
        "top_alts": r.get("top_alt_scores", ""),
        "temp": r.get("kickoff_temp_c"),
    }


def summary_stats(preds: pd.DataFrame) -> dict:
    """Quick summary stats for the dashboard header."""
    return {
        "total_fixtures": len(preds),
        "avg_reliability": preds["reliability"].mean(),
        "date_range": f"{preds['date'].min()} — {preds['date'].max()}",
        "strong_favorites": len(preds[preds["p_home"] > 0.70]) + len(preds[preds["p_away"] > 0.70]),
        "toss_ups": len(preds[(preds["p_home"] > 0.40) & (preds["p_home"] < 0.60)]),
    }


def match_score_grid(home_exp: float, away_exp: float, max_goals: int = 5) -> np.ndarray:
    """Generate a simple Poisson probability grid for visualization.
    (Not the actual model's grid — just a visual aid based on xG.
     The real grid comes from the model's grid_obj.)"""
    from scipy.stats import poisson
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[h, a] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
    return grid / grid.sum()  # normalize
