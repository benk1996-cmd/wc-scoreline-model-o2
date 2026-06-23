# Performance: static-by-default, refresh-on-demand

## Why it was slow

Streamlit reruns the whole script on every interaction (click, slider, scroll).
The old app ran the full data pipeline — pull ~49k results from GitHub and
recompute Elo from scratch (~30s) — as part of loading the page, so every
interaction risked paying that cost.

## How it works now

The dashboard is split into two clearly separated paths:

1. **Page load = static read (instant).** `app.py` calls
   `web.load_static_predictions()`, which only reads the last saved
   `data/processed/world_cup_predictions.csv`. It never runs the pipeline, so
   the page appears immediately no matter how large the underlying data is.
   The sidebar shows when those predictions were last updated.

2. **Refresh = explicit button (slow, ~30s).** Clicking **"Run refresh now"**
   in the sidebar calls `web.refresh_predictions()`, which re-ingests every
   data pipeline, recomputes Elo and features, re-predicts every fixture, and
   overwrites the CSV. Only then does the displayed data change.

So the cost is paid only when you actually ask for fresh numbers.

| Action | Before | Now |
|--------|--------|-----|
| Open the page | ~30s | instant |
| Click a fixture / move a filter | ~30s | instant |
| Click "Run refresh now" | ~30s | ~30s (expected — it's doing real work) |

## Keeping the deployed URL fresh

For the static view to show data on Streamlit Cloud, the predictions CSV must
be committed to the repo (the large `matches.parquet` stays git-ignored). The
`.gitignore` is set up for exactly this:

```
data/processed/*
!data/processed/world_cup_predictions.csv
```

To refresh the public site without anyone clicking the button, the included
GitHub Action (`.github/workflows/update-predictions.yml`) runs the pipeline on
a schedule and commits the updated CSV; Streamlit Cloud redeploys automatically.

## Tuning

`load_static()` in `app.py` is wrapped in `@st.cache_data(ttl=3600)` so repeated
reads within a session are free; the refresh button clears that cache. Adjust
`ttl` if you want the in-session cache to expire sooner or later.
