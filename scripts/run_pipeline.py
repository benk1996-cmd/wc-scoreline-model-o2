"""Run the model end to end:
  1. Update and ingest all data pipelines feeding the model.
  2. Generate scoreline predictions + reliability for every upcoming
     FIFA World Cup fixture.

    python -m scripts.run_pipeline
    python -m scripts.run_pipeline --no-lineups --no-weather   # skip network extras
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from wcmodel.pipeline import ingest_all, predict_upcoming_world_cup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lineups", action="store_true",
                    help="skip the API-Football lineup-availability check")
    ap.add_argument("--no-weather", action="store_true",
                    help="skip the Open-Meteo kickoff-temperature lookup")
    args = ap.parse_args()

    print("=" * 60)
    print("STEP 1/2 -- updating and ingesting data pipelines")
    print("=" * 60)
    matches = ingest_all()
    n_played = matches["home_score"].notna().sum()
    print(f"results spine   : {len(matches):,} matches "
         f"({n_played:,} played, through {matches['date'].max().date()})")
    print(f"feature columns : {[c for c in matches.columns if c.endswith(('_pre','_gf','_ga','rest_days','n_matches','elo_diff'))]}")

    print()
    print("=" * 60)
    print("STEP 2/2 -- forecasting upcoming World Cup fixtures")
    print("=" * 60)
    preds = predict_upcoming_world_cup(
        matches,
        check_lineups=not args.no_lineups,
        attach_weather=not args.no_weather,
    )

    out_path = config.PROC / "world_cup_predictions.csv"
    preds.to_csv(out_path, index=False)

    if preds.empty:
        print("No upcoming FIFA World Cup fixtures found in the spine right now.")
        return

    ok = preds[preds["status"] == "ok"]
    skipped = preds[preds["status"] != "ok"]
    print(f"{len(ok)}/{len(preds)} fixtures forecast successfully -> {out_path}")
    if len(skipped):
        print(f"skipped ({len(skipped)}):")
        for _, r in skipped.iterrows():
            print(f"  {r.date}  {r.home_team} vs {r.away_team}  -- {r.status}")

    if len(ok):
        cols = ["date", "home_team", "away_team", "predicted_score",
               "p_home", "p_draw", "p_away", "reliability"]
        print()
        print(ok[cols].to_string(index=False))


if __name__ == "__main__":
    main()
