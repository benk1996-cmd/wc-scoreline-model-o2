"""StatsBomb Open Data -- free event data (touches, passes, shots, xG, 360).

statsbombpy reads the public repo directly; no key needed. Men's FIFA World Cup
is competition_id 43 (seasons include 2018, 2022 and several historical
editions); the open set also covers EURO 2020/2024, Copa America 2024 and
AFCON 2023.

Event data does NOT exist for the fixture you are about to predict, so it can't
be a match-day input. Use it to build *priors*: team-level attacking/defensive
xG profiles that you can blend into the Dixon-Coles strength estimates or feed
a richer model later. `team_match_xg()` returns one row per team per match.
"""
from __future__ import annotations
import pandas as pd
from ..normalize import canonical

WC_COMPETITION_ID = 43


def list_competitions() -> pd.DataFrame:
    from statsbombpy import sb
    return sb.competitions()


def team_match_xg(competition_id: int, season_id: int) -> pd.DataFrame:
    """xG for & against per team per match in a tournament season."""
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    out = []
    for mid in matches["match_id"]:
        ev = sb.events(match_id=mid)
        shots = ev[ev["type"] == "Shot"]
        if shots.empty:
            continue
        xg = shots.groupby("team")["shot_statsbomb_xg"].sum()
        teams = list(xg.index)
        if len(teams) != 2:
            continue
        a, b = teams
        out.append({"match_id": mid, "team": canonical(a),
                    "opponent": canonical(b), "xg_for": xg[a], "xg_against": xg[b]})
        out.append({"match_id": mid, "team": canonical(b),
                    "opponent": canonical(a), "xg_for": xg[b], "xg_against": xg[a]})
    return pd.DataFrame(out)


def team_xg_profile(competition_id: int = WC_COMPETITION_ID,
                    season_ids: list[int] | None = None) -> pd.DataFrame:
    """Average xG for/against per team across the given seasons -> a prior."""
    from statsbombpy import sb
    if season_ids is None:
        comps = sb.competitions()
        season_ids = comps[comps["competition_id"] == competition_id]["season_id"].tolist()
    frames = [team_match_xg(competition_id, s) for s in season_ids]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return df.groupby("team")[["xg_for", "xg_against"]].mean().reset_index()
