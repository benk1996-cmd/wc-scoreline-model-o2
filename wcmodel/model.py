"""Scoreline forecaster: a recency-weighted, neutral-venue-aware Dixon-Coles
model (via penaltyblog) wrapped to emit an exact scoreline, the full score
grid, 1X2 probabilities, and a transparent reliability score.

Dixon-Coles is the right baseline for this task: it directly models each side's
goal *distribution*, so a most-likely scoreline and its probability fall out
naturally -- and it gives you a principled benchmark before any heavier ML.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import penaltyblog as pb


@dataclass
class Prediction:
    home_team: str
    away_team: str
    scoreline: tuple[int, int]          # most-likely exact score
    p_scoreline: float                  # its probability
    exp_goals: tuple[float, float]      # model goal expectations
    p_home: float
    p_draw: float
    p_away: float
    top_scores: list[tuple[str, float]] # top-k exact scores
    reliability: float                  # 0..1, see _reliability()
    components: dict = field(default_factory=dict)


def _np(s):
    return np.asarray(s).copy()


class ScorelineModel:
    def __init__(self, xi: float = 0.0018, max_goals: int = 15):
        self.xi = xi
        self.max_goals = max_goals
        self.model: pb.models.DixonColesGoalModel | None = None
        self.teams: set[str] = set()
        self._depth: dict[str, int] = {}

    def fit(self, played: pd.DataFrame) -> "ScorelineModel":
        """`played` needs: date, home_team, away_team, home_score, away_score,
        neutral. Recency weights are derived from `date`."""
        d = played.dropna(subset=["home_score", "away_score"]).copy()
        d["date"] = pd.to_datetime(d["date"])
        weights = _np(pb.models.dixon_coles_weights(d["date"], xi=self.xi)).astype(float)
        self.model = pb.models.DixonColesGoalModel(
            _np(d["home_score"].astype(int)),
            _np(d["away_score"].astype(int)),
            _np(d["home_team"]),
            _np(d["away_team"]),
            weights=weights,
            neutral_venue=_np(d["neutral"].fillna(False).astype(int)),
        )
        self.model.fit()
        self.teams = set(d["home_team"]) | set(d["away_team"])
        self._depth = (pd.concat([d["home_team"], d["away_team"]])
                       .value_counts().to_dict())
        return self

    def predict(self, home: str, away: str, neutral: bool = True,
                lineup_known: bool = False, top_k: int = 5) -> Prediction:
        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if home not in self.teams or away not in self.teams:
            missing = [t for t in (home, away) if t not in self.teams]
            raise KeyError(f"Unknown team(s) for this training window: {missing}")

        grid_obj = self.model.predict(home, away, max_goals=self.max_goals,
                                      neutral_venue=neutral)
        grid = np.asarray(grid_obj.grid)
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        hda = np.asarray(grid_obj.home_draw_away, dtype=float)

        flat = [((a, b), float(grid[a, b]))
                for a in range(grid.shape[0]) for b in range(grid.shape[1])]
        flat.sort(key=lambda x: -x[1])
        top = [(f"{a}-{b}", round(p, 4)) for (a, b), p in flat[:top_k]]

        rel, comp = self._reliability(hda, home, away, lineup_known)
        return Prediction(
            home_team=home, away_team=away,
            scoreline=(int(i), int(j)), p_scoreline=float(grid[i, j]),
            exp_goals=(round(float(grid_obj.home_goal_expectation), 2),
                       round(float(grid_obj.away_goal_expectation), 2)),
            p_home=float(hda[0]), p_draw=float(hda[1]), p_away=float(hda[2]),
            top_scores=top, reliability=rel, components=comp,
        )

    def _reliability(self, hda, home, away, lineup_known):
        """Transparent 0..1 confidence. Three interpretable factors:
          * outcome_confidence: how peaked the 1X2 distribution is
            (1 - normalised entropy). A coin-flip match -> low.
          * data_sufficiency: do we have enough recent matches for both teams?
          * lineup_factor: small penalty when the XI isn't confirmed yet.
        This is a starting heuristic -- CALIBRATE it against held-out results
        (see backtest in README) before trusting the numbers."""
        p = np.clip(hda, 1e-9, 1)
        entropy = -(p * np.log(p)).sum() / np.log(len(p))
        outcome_conf = 1.0 - float(entropy)
        depth = min(self._depth.get(home, 0), self._depth.get(away, 0))
        data_suff = min(depth / 30.0, 1.0)          # saturates at 30 matches
        lineup_factor = 1.0 if lineup_known else 0.85
        reliability = (0.6 * outcome_conf + 0.4 * data_suff) * lineup_factor
        return round(float(reliability), 3), {
            "outcome_confidence": round(outcome_conf, 3),
            "data_sufficiency": round(data_suff, 3),
            "lineup_known": bool(lineup_known),
            "min_recent_matches": int(depth),
        }
