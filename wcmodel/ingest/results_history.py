"""Spine source: every men's international 1872->present.

Source : github.com/martj42/international_results (CC0, updated continuously)
Access : raw CSV over https -- keyless, no rate limit.
Note   : the file already contains *scheduled* future fixtures (incl. the live
         World Cup) with null scores, so it doubles as a fixtures feed.
"""
from __future__ import annotations
import pandas as pd
from ..normalize import canonical, register_known

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def fetch(url: str = URL) -> pd.DataFrame:
    """Download and tidy the results table into the canonical match schema."""
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df["home_team"] = df["home_team"].map(canonical)
    df["away_team"] = df["away_team"].map(canonical)
    df["neutral"] = df["neutral"].astype("boolean")
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
    register_known(pd.concat([df["home_team"], df["away_team"]]).unique())
    cols = ["date", "home_team", "away_team", "tournament",
            "city", "country", "neutral", "home_score", "away_score"]
    return df[cols].sort_values("date").reset_index(drop=True)


def played(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a final score (training set)."""
    return df.dropna(subset=["home_score", "away_score"]).copy()


def upcoming(df: pd.DataFrame) -> pd.DataFrame:
    """Scheduled fixtures with no score yet (inference targets)."""
    return df[df["home_score"].isna() | df["away_score"].isna()].copy()
