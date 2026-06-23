"""The canonical match table every source is joined into.

Join keys
---------
* match identity  : (date, home_team, away_team)  -- after name normalisation
* team time-series: (team, date)                  -- for Elo / form lookups

Design rule (read this before adding a feature): a column is only useful for
*forecasting* if it is knowable BEFORE kickoff. Anything derived from the match
result itself (final xG, possession, shots) belongs to the TRAINING side only
and must never leak into the inference feature set. Columns below are tagged
[pre]  = available before kickoff (safe as a model input)
[post] = known only after the match (label / training-only)
"""

# --- match table -----------------------------------------------------------
MATCH_COLUMNS = {
    "date": "datetime64[ns]",     # [pre]
    "home_team": "string",        # [pre] canonical name
    "away_team": "string",        # [pre] canonical name
    "tournament": "string",       # [pre]
    "city": "string",             # [pre]
    "country": "string",          # [pre]
    "neutral": "boolean",         # [pre]
    "home_score": "Int64",        # [post] label
    "away_score": "Int64",        # [post] label
}

# --- engineered pre-match features (built in features.py) ------------------
FEATURE_COLUMNS = {
    "home_elo_pre": "float64",     # [pre] Elo before this match
    "away_elo_pre": "float64",     # [pre]
    "elo_diff": "float64",         # [pre] home_elo_pre - away_elo_pre (+ home adj)
    "home_rest_days": "float64",   # [pre] days since each team last played
    "away_rest_days": "float64",   # [pre]
    "home_form_gf": "float64",     # [pre] rolling goals-for per match
    "home_form_ga": "float64",     # [pre] rolling goals-against per match
    "away_form_gf": "float64",     # [pre]
    "away_form_ga": "float64",     # [pre]
    "home_n_matches": "Int64",     # [pre] sample depth -> feeds reliability score
    "away_n_matches": "Int64",     # [pre]
    "temp_c": "float64",           # [pre] kickoff temperature (Open-Meteo)
    "lineup_known": "boolean",     # [pre] confirmed XI available yet?
}

PRE_MATCH_FEATURES = list(FEATURE_COLUMNS)
