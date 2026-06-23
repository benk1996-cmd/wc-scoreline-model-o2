"""Build the unified, model-ready dataset from the keyless sources.

    python -m scripts.build_dataset

Produces data/processed/matches.parquet with the canonical schema plus
pre-match Elo and engineered features. Weather/lineups are left as optional
enrichment (they need network/keys) -- wire them in via the ingest modules.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from wcmodel.ingest import results_history as rh
from wcmodel.ingest import elo
from wcmodel import features


def main():
    print("1/4  pulling international results (martj42)...")
    matches = rh.fetch()
    print(f"     {len(matches):,} matches, through {matches['date'].max().date()}")

    print("2/4  computing World Football Elo...")
    matches, ratings = elo.compute(matches)
    top = sorted(ratings.items(), key=lambda x: -x[1])[:5]
    print("     top 5 now:", ", ".join(f"{t} {r:.0f}" for t, r in top))

    print("3/4  engineering pre-match features...")
    matches = features.add_features(matches, form_window=config.FORM_WINDOW)

    print("4/4  saving...")
    out = config.PROC / "matches.parquet"
    matches.to_parquet(out, index=False)
    print(f"     wrote {out}  ({len(matches):,} rows, {matches.shape[1]} cols)")


if __name__ == "__main__":
    main()
