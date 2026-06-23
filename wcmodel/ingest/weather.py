"""Kickoff weather via Open-Meteo -- free, keyless, generous limits.

Three endpoints, all keyless:
  * geocoding : resolve a host city -> lat/lon
  * archive   : historical daily weather (for the training set)
  * forecast  : up to ~16 days ahead (for upcoming fixtures)

Network note: open-meteo.com must be reachable. Results are cached on disk so
you only pay the request once per (city, date).
"""
from __future__ import annotations
import requests

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

_GEOCODE_CACHE: dict = {}  # city -> (lat, lon) | None, shared across calls in-process


def geocode(city: str, cache: dict | None = None) -> tuple[float, float] | None:
    if cache is not None and city in cache:
        return tuple(cache[city]) if cache[city] else None
    try:
        r = requests.get(GEOCODE, params={"name": city, "count": 1}, timeout=20)
        res = r.json().get("results")
        out = (res[0]["latitude"], res[0]["longitude"]) if res else None
    except Exception:
        out = None
    if cache is not None:
        cache[city] = list(out) if out else None
    return out


def temperature(city: str, date: str, lookahead_days: int = 16) -> float | None:
    """Mean 2m temperature (deg C) for `date` at `city`. Picks archive vs
    forecast automatically. `date` is 'YYYY-MM-DD'."""
    loc = geocode(city, cache=_GEOCODE_CACHE)
    if not loc:
        return None
    lat, lon = loc
    import datetime as _dt
    today = _dt.date.today()
    d = _dt.date.fromisoformat(str(date)[:10])
    base = ARCHIVE if d < today else FORECAST
    params = {"latitude": lat, "longitude": lon,
              "daily": "temperature_2m_mean", "timezone": "UTC",
              "start_date": str(date)[:10], "end_date": str(date)[:10]}
    try:
        vals = requests.get(base, params=params, timeout=20).json()["daily"]["temperature_2m_mean"]
        return float(vals[0]) if vals and vals[0] is not None else None
    except Exception:
        return None
