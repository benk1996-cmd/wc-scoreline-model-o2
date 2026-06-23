# Front-End Guide: World Cup Predictions Dashboard

The model ships with a **Streamlit dashboard** (`app.py`) that visualizes all upcoming World Cup fixtures with predictions, reliability scores, and probability distributions. It's the easiest way to interact with the model.

## Option 1: Run Streamlit locally (recommended for testing)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser. You'll see:

- **Summary header:** total fixtures, average reliability, date range
- **Filters:** by team, reliability threshold, sort order
- **Fixture table:** one row per match with predicted score, 1X2, expected goals, reliability
- **Interactive details:** click "Details" on any match to see:
  - Pie chart of 1X2 odds
  - Heatmap of possible final scores (Poisson approx)
  - Reliability component breakdown (outcome confidence, data sufficiency, lineup status)
  - Kickoff temperature (if available)

**How it works:**
- The dashboard calls `wcmodel.web.load_or_refresh_predictions()`, which:
  - Checks for cached CSV in `data/processed/world_cup_predictions.csv`
  - If missing/stale, runs the full pipeline (ingest + predict)
  - Returns the results as a DataFrame
- All UI is built with Streamlit widgets (no HTML/CSS needed)
- Charts use Plotly for interactivity

## Option 2: Deploy to Streamlit Cloud (free, public URL)

**Setup (one-time):**

1. Push your repo to GitHub:
   ```bash
   git init
   git add .
   git commit -m "initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/wc-scoreline-model
   git push -u origin main
   ```

2. Go to https://streamlit.io/cloud and sign in with GitHub
3. Click "New app" → select your repo, branch, and `app.py`
4. Streamlit deploys automatically. You get a public URL like:
   ```
   https://your-username-wc-model.streamlit.app
   ```

**Free tier includes:**
- Up to 3 deployed apps
- Shared CPU (~1 vCPU), 2GB RAM
- Auto-sleeps after 15 min of inactivity
- Auto-wakes on next visit

**Cost:** $0/month if you stay within limits. Upgrade to Pro if you need dedicated resources.

---

## Option 3: Flask REST API + custom front-end (more control)

If you want to decouple the UI from the model, build a Flask API:

```python
# api.py
from flask import Flask, jsonify
from wcmodel.pipeline import predict_upcoming_world_cup, ingest_all
import pandas as pd

app = Flask(__name__)

@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Return all upcoming World Cup predictions as JSON."""
    matches = ingest_all()
    preds = predict_upcoming_world_cup(matches)
    return jsonify(preds.to_dict(orient="records"))

@app.route("/api/predict/<home>/<away>", methods=["GET"])
def predict_fixture(home, away):
    """Predict a single fixture by team name."""
    from wcmodel.model import ScorelineModel
    import config
    matches = pd.read_parquet(config.PROC / "matches.parquet")
    played = matches.dropna(subset=["home_score", "away_score"])
    model = ScorelineModel().fit(played)
    pred = model.predict(home, away, neutral=True)
    return jsonify({
        "scoreline": pred.scoreline,
        "p_scoreline": pred.p_scoreline,
        "p_1x2": [pred.p_home, pred.p_draw, pred.p_away],
        "reliability": pred.reliability,
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
```

Then build a custom front-end with React, Vue, or plain HTML/JS that calls `/api/predictions` and renders it however you like.

**Pros:**
- Complete control over UI/UX
- Can build mobile app, custom charts, etc.
- Decoupled from Streamlit limitations
- Can scale to multiple users

**Cons:**
- More boilerplate (need to write API + front-end)
- Need to manage deployment separately (Heroku, AWS, etc.)

---

## Option 4: Static HTML + JavaScript (no server)

Since the model outputs a CSV, you can build a static site that reads it:

```html
<!DOCTYPE html>
<html>
<head>
    <title>World Cup Predictions</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>World Cup 2026 Predictions</h1>
    <div id="table"></div>
    <script>
        fetch('data/processed/world_cup_predictions.csv')
            .then(r => r.text())
            .then(csv => {
                // Parse CSV and render table
                const rows = csv.split('\n').map(r => r.split(','));
                // ... build HTML table
            });
    </script>
</body>
</html>
```

**Pros:**
- No server needed (works on GitHub Pages, Netlify, etc.)
- Pure client-side (no dependencies)
- Fast for static data

**Cons:**
- CORS issues when reading local CSV from browser
- No dynamic refresh (user must manually update CSV)
- Limited interactivity without a server

---

## Recommended: Streamlit + GitHub Actions (automated updates)

Combine Streamlit Cloud with a GitHub Actions workflow that runs `python -m scripts.run_pipeline` every 6 hours:

```yaml
# .github/workflows/update-predictions.yml
name: Update Predictions
on:
  schedule:
    - cron: "0 */6 * * *"  # every 6 hours
  workflow_dispatch:        # or manually trigger

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: python -m scripts.run_pipeline
      - name: Commit and push
        run: |
          git config user.name "CI"
          git add data/processed/world_cup_predictions.csv
          git commit -m "auto-update predictions" || true
          git push
```

This way:
- Predictions update automatically every 6 hours
- Streamlit Cloud picks up the new CSV automatically
- No manual intervention needed during the tournament

---

## Architecture summary

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit Cloud (app.py)                               │
│  https://your-app.streamlit.app                         │
│                                                         │
│  ├─ Load predictions from data/processed/ CSV           │
│  ├─ Render table + filters + charts                     │
│  └─ User clicks "Refresh" → runs ingest_all +           │
│     predict_upcoming_world_cup() locally                │
└─────────────────────────────────────────────────────────┘
                           ↑
                     Reads from Git
                           ↑
┌─────────────────────────────────────────────────────────┐
│  GitHub (your fork of wc-scoreline-model)               │
│                                                         │
│  ├─ app.py (Streamlit code)                             │
│  ├─ wcmodel/ (model package)                            │
│  └─ data/processed/                                     │
│     └─ world_cup_predictions.csv                        │
│        (updated by GitHub Actions every 6h)             │
└─────────────────────────────────────────────────────────┘
                           ↑
                     Updated by CI
                           ↑
┌─────────────────────────────────────────────────────────┐
│  GitHub Actions (.github/workflows/update-predictions.yml) │
│  Runs `python -m scripts.run_pipeline` on schedule       │
│  Commits CSV back to the repo                            │
└─────────────────────────────────────────────────────────┘
```

---

## File structure with front-end

```
wc-scoreline-model/
├── app.py                      ← Streamlit dashboard (THE ENTRY POINT)
├── api.py                      ← (optional) Flask REST API
├── FRONTEND.md                 ← This file
├── requirements.txt            ← Now includes streamlit, plotly
├── wcmodel/
│   ├── web.py                  ← Helper functions for the dashboard
│   ├── pipeline.py
│   └── ...
├── .github/
│   └── workflows/
│       └── update-predictions.yml  ← (optional) GitHub Actions automation
└── data/processed/
    └── world_cup_predictions.csv   ← Read by the dashboard
```

---

## Deployment checklist

**To deploy Streamlit Cloud:**

- [ ] Push to GitHub
- [ ] Add `app.py` to repo root
- [ ] Add `streamlit` to `requirements.txt`
- [ ] Go to streamlit.io/cloud, sign in, create app
- [ ] Select repo, branch, and `app.py`
- [ ] Done — public URL in ~1 minute

**To add automated updates:**

- [ ] Create `.github/workflows/update-predictions.yml`
- [ ] Set GitHub Actions secrets if you use API keys (optional)
- [ ] Push to GitHub
- [ ] Predictions update automatically on schedule

---

## Customization ideas

- **Color by outcome confidence:** Use 🟢🟡🔴 emoji or bar color
- **Add betting odds:** Integrate The Odds API (paid) to show market vs model comparison
- **Historical accuracy:** Track predictions vs actual results as tournament progresses
- **Group stage simulator:** Let user pick winners to see tournament projection
- **Mobile app:** React Native or Flutter (read the Flask API)
- **Slack bot:** Post updated predictions to a channel every 6 hours

Questions? See `wcmodel/web.py` for the data layer and `app.py` for UI examples.
