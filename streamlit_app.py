import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")


# Database connection
@st.cache_resource
def init_connection():
    return create_engine(st.secrets["db_connection"])


# Basic data loading functions
@st.cache_data
def get_teams():
    engine = init_connection()
    query = """
    SELECT DISTINCT team 
    FROM nhl24_matchups_with_situations 
    WHERE team IS NOT NULL 
    ORDER BY team
    """
    return pd.read_sql(query, engine)['team'].tolist()


def get_team_stats(team):
    query = """
    SELECT 
        AVG("goalsFor") as goals_for,
        AVG("goalsAgainst") as goals_against,
        AVG("xGoalsPercentage") as xg_percentage,
        AVG("corsiPercentage") as corsi_percentage,
        AVG("fenwickPercentage") as fenwick_percentage,
        AVG("highDangerShotsFor") as high_danger_shots,
        AVG("shotsOnGoalFor") as shots_on_goal,
        AVG("takeawaysFor") - AVG("giveawaysFor") as puck_management,
        AVG("faceOffsWonFor") as faceoffs_won
    FROM nhl24_matchups_with_situations
    WHERE team = %(team)s
    GROUP BY team
    """
    engine = init_connection()
    return pd.read_sql(query, engine, params={'team': team}).iloc[0]


# Simple UI
st.title("NHL Game Predictor 🏒")

# Team selection
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", get_teams(), key="home")
    home_odds = st.number_input("Home Odds", min_value=1.0, value=2.0, step=0.1)

with col2:
    away_team = st.selectbox("Away Team", get_teams(), key="away")
    away_odds = st.number_input("Away Odds", min_value=1.0, value=2.0, step=0.1)

draw_odds = st.number_input("Draw Odds", min_value=1.0, value=3.5, step=0.1)

# Make prediction button
if st.button("Analyze Match"):
    if home_team == away_team:
        st.error("Please select different teams")
    else:
        try:
            # Get team stats
            home_stats = get_team_stats(home_team)
            away_stats = get_team_stats(away_team)

            # Display enhanced stats comparison
            st.header("Team Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"📊 {home_team}")
                metrics = {
                    "Goals For": ("goals_for", "⚽"),
                    "Goals Against": ("goals_against", "🥅"),
                    "Expected Goals %": ("xg_percentage", "📈"),
                    "Corsi %": ("corsi_percentage", "🎯"),
                    "Fenwick %": ("fenwick_percentage", "🏒"),
                    "High Danger Shots": ("high_danger_shots", "⚡"),
                    "Shots on Goal": ("shots_on_goal", "🎪"),
                    "Puck Management": ("puck_management", "🏃"),
                    "Faceoffs Won": ("faceoffs_won", "🎮")
                }

                for label, (key, emoji) in metrics.items():
                    value = home_stats[key]
                    if "percentage" in key or "%" in label:
                        st.metric(f"{emoji} {label}", f"{value:.1f}%")
                    else:
                        st.metric(f"{emoji} {label}", f"{value:.1f}")

            with col2:
                st.subheader(f"📊 {away_team}")
                for label, (key, emoji) in metrics.items():
                    value = away_stats[key]
                    if "percentage" in key or "%" in label:
                        st.metric(f"{emoji} {label}", f"{value:.1f}%")
                    else:
                        st.metric(f"{emoji} {label}", f"{value:.1f}")

            # Comparison indicators
            st.subheader("📊 Key Matchup Indicators")
            comparison_cols = st.columns(3)

            # Offensive comparison
            with comparison_cols[0]:
                offensive_diff = home_stats['goals_for'] - away_stats['goals_for']
                st.metric("Offensive Strength",
                          f"{home_team if offensive_diff > 0 else away_team}",
                          f"{abs(offensive_diff):.1f} goals difference",
                          delta_color="normal" if offensive_diff > 0 else "inverse")

            # Possession comparison
            with comparison_cols[1]:
                corsi_diff = home_stats['corsi_percentage'] - away_stats['corsi_percentage']
                st.metric("Possession Control",
                          f"{home_team if corsi_diff > 0 else away_team}",
                          f"{abs(corsi_diff):.1f}% difference",
                          delta_color="normal" if corsi_diff > 0 else "inverse")

            # Quality chances comparison
            with comparison_cols[2]:
                xg_diff = home_stats['xg_percentage'] - away_stats['xg_percentage']
                st.metric("Quality Chances",
                          f"{home_team if xg_diff > 0 else away_team}",
                          f"{abs(xg_diff):.1f}% difference",
                          delta_color="normal" if xg_diff > 0 else "inverse")

            # Debug data in expander
            with st.expander("Debug Data"):
                st.json({
                    "home_stats": dict(home_stats),
                    "away_stats": dict(away_stats)
                })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Debug info:")
            st.write(f"Home team: {home_team}")
            st.write(f"Away team: {away_team}")

# Add informative footer
st.markdown("---")
st.markdown("""
    ℹ️ **How to use:**
    1. Select home and away teams
    2. Enter current betting odds
    3. Click "Analyze Match" for predictions
""")