"""Build the leak-free, pre-match feature table.

Everything here is knowable before kickoff: pre-match Elo (from ingest.elo),
rest days, and rolling form computed only from *earlier* matches. Features are
joined back to matches on a unique match id so the row count is preserved
exactly (a naive (date, team) join fans out when a team plays twice on a date).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def add_features(matches: pd.DataFrame, form_window: int = 10) -> pd.DataFrame:
    """Return matches with engineered pre-match feature columns.
    Assumes ingest.elo.compute() has already added home_elo_pre/away_elo_pre."""
    m = matches.reset_index(drop=True).copy()
    m["_mid"] = np.arange(len(m))

    # explode to one row per team per match, tagged with match id + side
    home = pd.DataFrame({"_mid": m["_mid"], "side": "home", "date": m["date"],
                         "team": m["home_team"], "gf": m["home_score"], "ga": m["away_score"]})
    away = pd.DataFrame({"_mid": m["_mid"], "side": "away", "date": m["date"],
                         "team": m["away_team"], "gf": m["away_score"], "ga": m["home_score"]})
    long = (pd.concat([home, away], ignore_index=True)
            .sort_values(["team", "date", "_mid"]).reset_index(drop=True))

    g = long.groupby("team", sort=False)
    long["rest_days"] = (long["date"] - g["date"].shift(1)).dt.days
    # rolling form from strictly earlier matches only (shift(1))
    long["form_gf"] = (g["gf"].shift(1)
                       .rolling(form_window, min_periods=1).mean()
                       .reset_index(level=0, drop=True))
    long["form_ga"] = (g["ga"].shift(1)
                       .rolling(form_window, min_periods=1).mean()
                       .reset_index(level=0, drop=True))
    long["n_matches"] = g.cumcount()  # matches seen before this one

    feat_cols = ["rest_days", "form_gf", "form_ga", "n_matches"]
    out = m
    for side in ("home", "away"):
        side_feat = (long[long["side"] == side]
                     .set_index("_mid")[feat_cols]
                     .add_prefix(f"{side}_"))
        out = out.merge(side_feat, left_on="_mid", right_index=True, how="left")

    assert len(out) == len(matches), "feature join changed row count"

    home_adv = np.where(out["neutral"].fillna(False), 0.0, 100.0)
    out["elo_diff"] = out["home_elo_pre"] - out["away_elo_pre"] + home_adv
    return out.drop(columns="_mid")
