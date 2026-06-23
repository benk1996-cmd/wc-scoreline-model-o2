"""Live World Cup fixtures & results via football-data.org (v4 REST, free tier).

Free tier (as of 2026): 12 competitions incl. the FIFA World Cup ('WC') and
European Championship ('EC'); 10 requests/minute; fixtures, results, standings
and scorers only -- NO lineups or match stats on the free plan.

Set FOOTBALL_DATA_API_KEY in your environment. The martj42 spine already
carries scheduled fixtures, so treat this as the *authoritative, low-latency*
result/fixture feed during the tournament rather than your only source.
"""
from __future__ import annotations
import os
import requests
import pandas as pd
from ..normalize import canonical

BASE = "https://api.football-data.org/v4"


def fetch_competition_matches(code: str = "WC") -> pd.DataFrame:
    key = os.getenv("FOOTBALL_DATA_API_KEY", "")
    if not key:
        raise RuntimeError("Set FOOTBALL_DATA_API_KEY (free key at football-data.org).")
    r = requests.get(f"{BASE}/competitions/{code}/matches",
                     headers={"X-Auth-Token": key}, timeout=30)
    r.raise_for_status()
    rows = []
    for m in r.json().get("matches", []):
        ft = m.get("score", {}).get("fullTime", {})
        rows.append({
            "date": pd.to_datetime(m["utcDate"]).tz_localize(None),
            "home_team": canonical(m["homeTeam"].get("name")),
            "away_team": canonical(m["awayTeam"].get("name")),
            "tournament": "FIFA World Cup",
            "stage": m.get("stage"),
            "status": m.get("status"),
            "neutral": True,  # World Cup: treat all as neutral except host games
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
        })
    return pd.DataFrame(rows)
