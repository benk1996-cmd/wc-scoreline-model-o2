"""Predict a single fixture's scoreline and reliability.

    python -m scripts.predict_match "Brazil" "Croatia"
    python -m scripts.predict_match "Spain" "Germany" --venue home

Trains the Dixon-Coles model on the recent window from the prebuilt dataset
(run build_dataset.py first) and prints the forecast.
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config
from wcmodel.model import ScorelineModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--venue", choices=["neutral", "home"], default="neutral")
    ap.add_argument("--lineup-known", action="store_true")
    args = ap.parse_args()

    path = config.PROC / "matches.parquet"
    if not path.exists():
        sys.exit("Run `python -m scripts.build_dataset` first.")
    df = pd.read_parquet(path)
    df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(config.TRAIN_FROM)]
    played = df.dropna(subset=["home_score", "away_score"])

    model = ScorelineModel(xi=config.DIXON_COLES_XI, max_goals=config.MAX_GOALS).fit(played)
    pred = model.predict(args.home, args.away,
                         neutral=(args.venue == "neutral"),
                         lineup_known=args.lineup_known)

    h, a = pred.scoreline
    print(f"\n{pred.home_team} vs {pred.away_team}  ({args.venue})")
    print("-" * 48)
    print(f"most likely score : {h}-{a}   (p={pred.p_scoreline:.1%})")
    print(f"expected goals    : {pred.exp_goals[0]} - {pred.exp_goals[1]}")
    print(f"outcome 1X2       : {pred.p_home:.1%} / {pred.p_draw:.1%} / {pred.p_away:.1%}")
    print(f"other likely      : {', '.join(s for s, _ in pred.top_scores[1:4])}")
    print(f"reliability       : {pred.reliability:.2f}   {pred.components}")


if __name__ == "__main__":
    main()
