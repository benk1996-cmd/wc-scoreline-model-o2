"""Central configuration. API keys are read from the environment so nothing
secret lives in the repo. Only the live feeds (football-data.org, API-Football)
need keys; the historical training pipeline is fully keyless.
"""
import os
from pathlib import Path

# --- paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
for _p in (DATA, RAW, PROC):
    _p.mkdir(parents=True, exist_ok=True)

# --- API keys (optional; set in your shell) --------------------------------
# export FOOTBALL_DATA_API_KEY="..."     # https://www.football-data.org (free tier)
# export API_FOOTBALL_KEY="..."          # https://www.api-football.com (free tier)
FOOTBALL_DATA_API_KEY = os.getenv("FOOTBALL_DATA_API_KEY", "")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")

# --- modelling settings ----------------------------------------------------
# Only fit on reasonably recent internationals: squads/managers churn, so old
# matches are weak evidence. The recency weight (xi) handles fine-grained decay.
TRAIN_FROM = "2015-01-01"
DIXON_COLES_XI = 0.0018      # ~half-life of roughly one year; raise to forget faster
FORM_WINDOW = 10             # matches used for rolling form features
MAX_GOALS = 15               # scoreline grid is (MAX_GOALS+1) x (MAX_GOALS+1)
