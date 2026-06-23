"""Optional Flask REST API for the World Cup model.

Use this if you want to build a custom front-end (React, Vue, mobile, etc.)
that doesn't use Streamlit.

Run with:
  pip install flask flask-cors
  python api.py

Then make requests:
  GET /api/predictions
  GET /api/predictions?sort=reliability
  GET /api/predictions?team=Brazil
  GET /api/predict/Brazil/Croatia
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from datetime import datetime

from wcmodel.web import load_or_refresh_predictions, parse_reliability_components
from wcmodel.model import ScorelineModel
import config

app = Flask(__name__)
CORS(app)  # allow cross-origin requests (for front-ends on different domains)


@app.route("/api/health", methods=["GET"])
def health():
    """Liveness check."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    """Get all upcoming World Cup predictions.
    
    Query params:
      - sort: 'reliability' (desc), 'date' (asc), 'home_win' (desc) [default: date]
      - team: filter by home or away team name
      - min_reliability: 0.0-1.0 [default: 0.0]
    """
    preds = load_or_refresh_predictions(force_refresh=False)
    
    # Filter by team
    team = request.args.get("team")
    if team:
        preds = preds[
            (preds["home_team"] == team) | (preds["away_team"] == team)
        ]
    
    # Filter by reliability
    min_rel = float(request.args.get("min_reliability", 0.0))
    preds = preds[preds["reliability"] >= min_rel]
    
    # Sort
    sort = request.args.get("sort", "date")
    if sort == "reliability":
        preds = preds.sort_values("reliability", ascending=False)
    elif sort == "home_win":
        preds = preds.sort_values("p_home", ascending=False)
    else:
        preds = preds.sort_values("date")
    
    # Convert to JSON-friendly format
    rows = []
    for _, r in preds.iterrows():
        rows.append({
            "date": r["date"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "venue": {"city": r["venue_city"], "country": r["venue_country"]},
            "neutral": bool(r["neutral"]),
            "prediction": {
                "scoreline": r["predicted_score"],
                "p_scoreline": round(r["p_scoreline"], 4),
                "exp_goals": [r["exp_goals_home"], r["exp_goals_away"]],
            },
            "outcome_1x2": {
                "p_home": round(r["p_home"], 4),
                "p_draw": round(r["p_draw"], 4),
                "p_away": round(r["p_away"], 4),
            },
            "reliability": {
                "score": r["reliability"],
                "components": parse_reliability_components(r.get("reliability_components", "{}")),
            },
            "alternatives": r.get("top_alt_scores", ""),
            "kickoff_temp_c": r.get("kickoff_temp_c"),
        })
    
    return jsonify({
        "count": len(rows),
        "data": rows,
    })


@app.route("/api/predict/<home>/<away>", methods=["GET"])
def predict_single(home, away):
    """Predict a single fixture by team names (URL-encoded).
    
    Example: /api/predict/Brazil/Croatia?neutral=true&lineup_known=false
    """
    neutral = request.args.get("neutral", "true").lower() == "true"
    lineup_known = request.args.get("lineup_known", "false").lower() == "true"
    
    try:
        matches = pd.read_parquet(config.PROC / "matches.parquet")
        matches["date"] = pd.to_datetime(matches["date"])
        window = matches[matches["date"] >= pd.Timestamp(config.TRAIN_FROM)]
        played = window.dropna(subset=["home_score", "away_score"])
        
        model = ScorelineModel(xi=config.DIXON_COLES_XI).fit(played)
        pred = model.predict(home, away, neutral=neutral, lineup_known=lineup_known)
        
        return jsonify({
            "home_team": pred.home_team,
            "away_team": pred.away_team,
            "prediction": {
                "scoreline": f"{pred.scoreline[0]}-{pred.scoreline[1]}",
                "p_scoreline": round(pred.p_scoreline, 4),
                "exp_goals": pred.exp_goals,
            },
            "outcome_1x2": {
                "p_home": round(pred.p_home, 4),
                "p_draw": round(pred.p_draw, 4),
                "p_away": round(pred.p_away, 4),
            },
            "reliability": pred.reliability,
            "reliability_components": pred.components,
            "top_alternatives": pred.top_scores[:5],
        })
    except KeyError as e:
        return jsonify({"error": f"Team not found: {e}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh", methods=["POST"])
def refresh():
    """Force a full refresh of predictions (re-ingest + re-predict).
    
    This is slow (~30 seconds) because it pulls all results and recomputes Elo.
    """
    try:
        preds = load_or_refresh_predictions(force_refresh=True)
        return jsonify({
            "status": "ok",
            "fixtures_refreshed": len(preds),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats():
    """Quick summary stats."""
    preds = load_or_refresh_predictions()
    return jsonify({
        "total_fixtures": len(preds),
        "avg_reliability": round(preds["reliability"].mean(), 3),
        "date_range": {
            "from": preds["date"].min(),
            "to": preds["date"].max(),
        },
        "strong_favorites": len(preds[(preds["p_home"] > 0.70) | (preds["p_away"] > 0.70)]),
        "toss_ups": len(preds[(preds["p_home"] > 0.40) & (preds["p_home"] < 0.60)]),
    })


if __name__ == "__main__":
    print("""
    Flask API running on http://localhost:5000
    
    Try:
      curl http://localhost:5000/api/predictions
      curl http://localhost:5000/api/stats
      curl http://localhost:5000/api/predict/Brazil/Croatia
    """)
    app.run(debug=True, port=5000)
