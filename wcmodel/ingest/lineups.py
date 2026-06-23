"""Lineups & availability -- the painful free gap.

There is no free, keyless API for confirmed XIs. API-Football's free tier
(100 requests/day) carries World Cup lineups but they only land ~1h pre-kickoff
and the quota is tiny. This module is opt-in: with no key set it always reports
lineup_known=False and the reliability score widens accordingly (see model.py).
Set API_FOOTBALL_KEY to enable best-effort checks.
"""
from __future__ import annotations
import os
import requests

BASE = "https://v3.football.api-sports.io"
WC_LEAGUE_ID = 1  # API-Football's FIFA World Cup league id


def find_fixture_id(home: str, away: str, date: str) -> int | None:
    """Best-effort lookup of the API-Football fixture id for a (home, away, date)
    World Cup match, by scanning that day's fixtures. Returns None on any miss
    (wrong team-name spelling between sources, no key, network error, etc.) --
    callers must treat None as 'unknown', never as 'no lineup'."""
    key = os.getenv("API_FOOTBALL_KEY", "")
    if not key:
        return None
    try:
        r = requests.get(f"{BASE}/fixtures",
                         headers={"x-apisports-key": key},
                         params={"league": WC_LEAGUE_ID, "date": str(date)[:10]},
                         timeout=20)
        for m in r.json().get("response", []):
            h = m["teams"]["home"]["name"]
            a = m["teams"]["away"]["name"]
            if {h, a} == {home, away}:
                return m["fixture"]["id"]
    except Exception:
        return None
    return None


def lineup_available(fixture_id: int) -> bool:
    key = os.getenv("API_FOOTBALL_KEY", "")
    if not key or fixture_id is None:
        return False
    try:
        r = requests.get(f"{BASE}/fixtures/lineups",
                         headers={"x-apisports-key": key},
                         params={"fixture": fixture_id}, timeout=20)
        return bool(r.json().get("response"))
    except Exception:
        return False


def check(home: str, away: str, date: str) -> bool:
    """Convenience: fixture lookup + lineup check in one call. Always False
    if API_FOOTBALL_KEY isn't set -- this is the expected, honest default."""
    fid = find_fixture_id(home, away, date)
    return lineup_available(fid) if fid is not None else False
