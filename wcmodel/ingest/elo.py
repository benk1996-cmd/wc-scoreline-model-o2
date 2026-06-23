"""World Football Elo, computed from the results history.

Why compute it instead of scraping eloratings.net? It is fully reproducible,
keyless, has no fragile-scrape dependency, and gives us the *pre-match* rating
for every historical fixture for free -- exactly the leak-free feature we want.
The formula follows the public World Football Elo Ratings method
(eloratings.net): K scaled by match importance, a goal-difference multiplier,
and a home-advantage term that is switched off at neutral venues.

Output: the input frame plus `home_elo_pre` / `away_elo_pre` (ratings BEFORE
each match) and a `ratings` dict of final ratings.
"""
from __future__ import annotations
import pandas as pd

START = 1500.0
HOME_ADV = 100.0

# match-importance K, matched on substrings of the `tournament` column
def _k(tournament: str) -> float:
    t = str(tournament).lower()
    if "world cup" in t and "qual" not in t:
        return 60.0
    if any(x in t for x in ("euro", "copa am", "african cup", "asian cup",
                            "gold cup", "nations league")) and "qual" not in t:
        return 50.0
    if "qual" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 30.0


def _g(margin: int) -> float:
    """Goal-difference multiplier."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    if margin == 3:
        return 1.75
    return 1.75 + (margin - 3) / 8.0


def compute(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk matches chronologically, returning the frame with pre-match Elo
    columns and the final ratings dict."""
    df = matches.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    hp, ap = [], []
    for r in df.itertuples(index=False):
        rh = ratings.get(r.home_team, START)
        ra = ratings.get(r.away_team, START)
        hp.append(rh)
        ap.append(ra)
        # skip rating update for unplayed fixtures
        if pd.isna(r.home_score) or pd.isna(r.away_score):
            continue
        adv = 0.0 if bool(r.neutral) else HOME_ADV
        dr = (rh + adv) - ra
        we_home = 1.0 / (10 ** (-dr / 400.0) + 1.0)
        hs, as_ = int(r.home_score), int(r.away_score)
        w_home = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = _k(r.tournament) * _g(abs(hs - as_))
        delta = k * (w_home - we_home)
        ratings[r.home_team] = rh + delta
        ratings[r.away_team] = ra - delta
    df["home_elo_pre"] = hp
    df["away_elo_pre"] = ap
    return df, ratings
