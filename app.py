"""Streamlit dashboard for World Cup scoreline predictions.

Run with:
  streamlit run app.py

Then open http://localhost:8501 in your browser.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from wcmodel import web


# Read the last saved predictions instantly. This NEVER runs the pipeline, so
# the page loads immediately regardless of how big the data is.
@st.cache_data(ttl=3600)
def load_static():
    return web.load_static_predictions()


@st.cache_resource
def get_scipy_poisson():
    """Pre-import scipy.stats.poisson (lightweight, but cache for consistency)."""
    from scipy.stats import poisson
    return poisson


def render_1x2_pie(p_home, p_draw, p_away, home, away):
    """Simple pie chart of 1X2 odds."""
    fig = go.Figure(data=[go.Pie(
        labels=[f"{home}\n{p_home:.0%}", f"Draw\n{p_draw:.0%}", f"{away}\n{p_away:.0%}"],
        values=[p_home, p_draw, p_away],
        marker=dict(colors=["#1f77b4", "#aec7e8", "#ff7f0e"]),
        textposition="auto",
    )])
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def render_score_grid(exp_home, exp_away):
    """Heatmap of possible final scores (Poisson approximation for viz)."""
    poisson = get_scipy_poisson()
    max_goals = 6
    grid = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            grid[h, a] = poisson.pmf(h, exp_home) * poisson.pmf(a, exp_away)
    grid = grid / grid.sum()
    
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=[str(i) for i in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        colorscale="Blues",
        hovertemplate="Home %{y} - Away %{x}<br>p=%{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        title="Score Probability Grid (Poisson approx)",
        xaxis_title="Away Goals", yaxis_title="Home Goals",
        height=400, margin=dict(l=50, r=50, t=50, b=50),
    )
    return fig


def main():
    st.set_page_config(page_title="World Cup Predictions", layout="wide")
    st.title("⚽ World Cup 2026 Scoreline Predictions")
    
    # Sidebar: refresh controls
    with st.sidebar:
        st.header("Data")
        updated = web.last_updated()
        if updated:
            st.caption(f"Predictions last updated:\n**{updated}**")
        else:
            st.caption("No predictions saved yet.")

        st.markdown("---")
        st.markdown("**Refresh data pipelines**")
        st.caption("Re-pulls results, recomputes Elo, re-predicts every fixture. "
                   "Takes ~30s — only needed when new results have come in.")
        if st.button("🔄 Run refresh now", use_container_width=True, type="primary"):
            with st.spinner("Updating data pipelines and re-predicting… (~30s)"):
                web.refresh_predictions()
            load_static.clear()      # invalidate the cached static read
            st.success("Predictions refreshed.")
            st.rerun()

        st.markdown("---")
        if st.button("View model details", use_container_width=True):
            st.session_state.show_details = not st.session_state.get("show_details", False)

    # Load the last saved predictions instantly (no pipeline run on page load)
    preds = load_static()

    if preds is None:
        st.info("No saved predictions found yet. Click **Run refresh now** in the "
                "sidebar to generate them for the first time (takes ~30s).")
        return
    if preds.empty:
        st.warning("No upcoming World Cup fixtures in the saved predictions.")
        return
    
    # Summary header
    stats = web.summary_stats(preds)
    cols = st.columns(5)
    cols[0].metric("Fixtures", stats["total_fixtures"])
    cols[1].metric("Avg Reliability", f"{stats['avg_reliability']:.2f}")
    cols[2].metric("Strong Favorites", stats["strong_favorites"])
    cols[3].metric("Toss-ups", stats["toss_ups"])
    cols[4].metric("Date Range", stats["date_range"].replace(" — ", "\n"))
    
    st.divider()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        min_rel = st.slider("Min reliability", 0.0, 1.0, 0.0, step=0.1)
    with col2:
        selected_team = st.selectbox(
            "Filter by team", 
            ["All teams"] + sorted(preds["home_team"].unique().tolist()),
        )
    with col3:
        sort_by = st.selectbox("Sort by", ["Date", "Reliability", "Home Win %"])
    
    # Apply filters
    filtered = preds[preds["reliability"] >= min_rel].copy()
    if selected_team != "All teams":
        filtered = filtered[
            (filtered["home_team"] == selected_team) | (filtered["away_team"] == selected_team)
        ]
    
    # Sort
    if sort_by == "Reliability":
        filtered = filtered.sort_values("reliability", ascending=False)
    elif sort_by == "Home Win %":
        filtered = filtered.sort_values("p_home", ascending=False)
    else:
        filtered = filtered.sort_values("date")
    
    st.subheader(f"Upcoming Fixtures ({len(filtered)})")
    
    # Display as table with interactivity
    for idx, r in filtered.iterrows():
        with st.container(border=True):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1.5])
            
            with col1:
                h, a = r["predicted_score"].split("-")
                st.write(f"**{r['date']}**")
                st.write(f"{r['home_team']} vs {r['away_team']}")
                st.caption(f"{r['venue_city']}, {r['venue_country']}")
            
            with col2:
                st.metric("Predicted", r["predicted_score"], f"p={r['p_scoreline']:.1%}")
            
            with col3:
                st.metric(
                    "1X2",
                    f"{r['p_home']:.0%} / {r['p_draw']:.0%} / {r['p_away']:.0%}",
                    f"xG: {r['exp_goals_home']:.1f}-{r['exp_goals_away']:.1f}",
                )
            
            with col4:
                rel_color = "🟢" if r["reliability"] >= 0.5 else "🟡" if r["reliability"] >= 0.35 else "🔴"
                st.metric(f"{rel_color} Reliability", f"{r['reliability']:.2f}")
            
            # Expandable details
            if st.button("Details", key=f"btn_{idx}"):
                st.session_state[f"detail_{idx}"] = not st.session_state.get(f"detail_{idx}", False)
            
            if st.session_state.get(f"detail_{idx}", False):
                st.divider()
                detail_col1, detail_col2 = st.columns(2)
                
                with detail_col1:
                    st.plotly_chart(
                        render_1x2_pie(r["p_home"], r["p_draw"], r["p_away"], r["home_team"], r["away_team"]),
                        use_container_width=True,
                    )
                
                with detail_col2:
                    st.plotly_chart(
                        render_score_grid(r["exp_goals_home"], r["exp_goals_away"]),
                        use_container_width=True,
                    )
                
                # Reliability breakdown
                comp = web.parse_reliability_components(r.get("reliability_components", "{}"))
                st.markdown("**Reliability Breakdown**")
                rb_col1, rb_col2, rb_col3 = st.columns(3)
                rb_col1.write(f"Outcome confidence: **{comp.get('outcome_confidence', 0):.2f}**")
                rb_col2.write(f"Data sufficiency: **{comp.get('data_sufficiency', 0):.2f}**")
                rb_col3.write(f"Lineup known: **{comp.get('lineup_known', False)}**")
                
                # Weather
                if not pd.isna(r.get("kickoff_temp_c")):
                    st.write(f"Kickoff temp: **{r['kickoff_temp_c']:.1f}°C**")
                
                # Top alternatives
                if pd.notna(r.get("top_alt_scores")):
                    st.write(f"Top alternatives: {r['top_alt_scores']}")
    
    st.divider()
    
    # Footer
    if st.session_state.get("show_details", False):
        st.markdown("""
### Model Details

**Dixon-Coles model** with recency weighting on international match results (2015+).

**Features used:**
- World Football Elo (home + away, pre-match)
- Rolling form (goals for/against, 10-match window)
- Rest days since last match
- Home advantage (not applied at neutral venues)
- Sample depth (feeds reliability score)
- Kickoff temperature (Open-Meteo, forecast only)

**Reliability score factors:**
1. **Outcome confidence** — entropy of the 1X2 distribution (how "peaked" vs spread)
2. **Data sufficiency** — how many recent matches each team has (saturates at 30)
3. **Lineup known** — confirmed XI available (~1h pre-kickoff)

**Not used:** Betting odds, event-level data, tracking data, head-to-head history.

See README.md and wcmodel/pipeline.py for details.
        """)
    
    # Auto-refresh every 5 minutes (if running headless/scheduled)
    import time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()


if __name__ == "__main__":
    main()
