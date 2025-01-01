# app.py
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

# Feature calculation functions
def calculate_xg_score(home_stats, away_stats):
    xg_home = home_stats['xGoalsPercentage']
    xg_away = away_stats['xGoalsPercentage']
    xg_diff = xg_home - xg_away
    return xg_diff / 100  # Normalize to 0-1 range

def calculate_form_score(home_stats, away_stats):
    home_form = home_stats['recent_win_rate']
    away_form = away_stats['recent_win_rate']
    form_diff = home_form - away_form
    return (form_diff + 1) / 2  # Normalize to 0-1 range

def calculate_possession_score(home_stats, away_stats):
    corsi_diff = home_stats['corsiPercentage'] - away_stats['corsiPercentage']
    fenwick_diff = home_stats['fenwickPercentage'] - away_stats['fenwickPercentage']
    return ((corsi_diff + fenwick_diff) / 2) / 100  # Normalize to 0-1 range


def calculate_h2h_score(h2h_stats):
    if h2h_stats['games_played'] == 0:
        return 0.5

    win_rate = h2h_stats['home_team_wins'] / h2h_stats['games_played']
    return win_rate


def calculate_market_score(home_odds, away_odds, draw_odds):
    home_prob = 1 / home_odds
    away_prob = 1 / away_odds
    draw_prob = 1 / draw_odds
    total = home_prob + away_prob + draw_prob

    # Normalize probabilities
    home_prob_norm = home_prob / total
    away_prob_norm = away_prob / total

    market_confidence = abs(home_prob_norm - away_prob_norm)
    return market_confidence


def get_team_stats(team):
    """Get team statistics with validation"""
    query = """
   WITH recent_games AS (
       SELECT 
           team,
           games_played,
           COALESCE("goalsFor", 0) as "goalsFor",
           COALESCE("goalsAgainst", 0) as "goalsAgainst",
           COALESCE("xGoalsPercentage", 50) as "xGoalsPercentage",
           COALESCE("corsiPercentage", 50) as "corsiPercentage",
           COALESCE("fenwickPercentage", 50) as "fenwickPercentage"
       FROM nhl24_matchups_with_situations
       WHERE team = %(team)s
       ORDER BY games_played DESC
       LIMIT 10
   )
   SELECT 
       MAX(games_played) as games_played,
       AVG("goalsFor") as "goalsFor",
       AVG("goalsAgainst") as "goalsAgainst",
       AVG("xGoalsPercentage") as "xGoalsPercentage",
       AVG("corsiPercentage") as "corsiPercentage",
       AVG("fenwickPercentage") as "fenwickPercentage",
       COUNT(CASE WHEN "goalsFor" > "goalsAgainst" THEN 1 END)::float / 
           NULLIF(COUNT(*), 0) as recent_win_rate
   FROM recent_games
   """
    engine = init_connection()
    result = pd.read_sql(query, engine, params={'team': team})
    return result.iloc[0] if not result.empty else None


def validate_data_quality(home_stats, away_stats, h2h_stats):
    """Validate data meets minimum quality thresholds"""
    MIN_GAMES = 5
    MIN_H2H = 0

    if home_stats is None or away_stats is None:
        return False, "Missing team stats"

    # Convert to float to avoid pandas Series boolean ambiguity
    if float(home_stats['games_played']) < MIN_GAMES or float(away_stats['games_played']) < MIN_GAMES:
        return False, "Insufficient recent games"

    if float(h2h_stats['games_played']) < MIN_H2H:
        return False, "Insufficient head-to-head history"

    return True, "Data quality checks passed"


def get_head_to_head_stats(home_team, away_team):
    """Get H2H stats with validation"""
    query = """
   WITH game_results AS (
       SELECT 
           game_date,
           home_team,
           away_team,
           CASE WHEN home_team_score > away_team_score THEN 1 ELSE 0 END as home_win
       FROM nhl24_results
       WHERE (home_team = %(home)s AND away_team = %(away)s)
          OR (home_team = %(away)s AND away_team = %(home)s)
       ORDER BY game_date DESC
       LIMIT 5
   )
   SELECT 
       COUNT(*) as games_played,
       SUM(home_win) as home_team_wins
   FROM game_results
   """
    engine = init_connection()
    result = pd.read_sql(query, engine, params={'home': home_team, 'away': away_team})
    return result.iloc[0] if not result.empty else pd.Series({'games_played': 0, 'home_team_wins': 0})


def calculate_prediction(home_team, away_team, home_odds, away_odds, draw_odds):
    # Get stats
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    h2h_stats = get_head_to_head_stats(home_team, away_team)

    # Validate data
    is_valid, message = validate_data_quality(home_stats, away_stats, h2h_stats)
    if not is_valid:
        return None, message

    # Calculate weighted features
    features = {
        'xg_score': calculate_xg_score(home_stats, away_stats) * 0.25,
        'form_score': calculate_form_score(home_stats, away_stats) * 0.20,
        'h2h_score': calculate_h2h_score(h2h_stats) * 0.20,
        'possession_score': calculate_possession_score(home_stats, away_stats) * 0.15,
        'market_score': calculate_market_score(home_odds, away_odds, draw_odds) * 0.10
    }

    # Calculate final prediction
    total_score = sum(features.values())

    # Convert to probabilities
    home_prob = total_score
    away_prob = 1 - total_score
    draw_prob = 0.2  # Base draw probability

    # Normalize
    total = home_prob + away_prob + draw_prob
    probabilities = [away_prob / total, draw_prob / total, home_prob / total]

    return probabilities, "Valid prediction"


def display_prediction_results(probabilities, home_team, away_team, home_odds, away_odds, draw_odds):
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    # Calculate implied probabilities for value comparison
    implied_probs = [
        1 / away_odds / (1 / home_odds + 1 / away_odds + 1 / draw_odds),
        1 / draw_odds / (1 / home_odds + 1 / away_odds + 1 / draw_odds),
        1 / home_odds / (1 / home_odds + 1 / away_odds + 1 / draw_odds)
    ]

    with col1:
        value = probabilities[0] - implied_probs[0]
        st.metric(
            f"{away_team} (Away)",
            f"{probabilities[0]:.1%}",
            f"Value: {value:+.1%}",
            delta_color="normal" if value > 0 else "off"
        )

    with col2:
        value = probabilities[1] - implied_probs[1]
        st.metric(
            "Draw",
            f"{probabilities[1]:.1%}",
            f"Value: {value:+.1%}",
            delta_color="normal" if value > 0 else "off"
        )

    with col3:
        value = probabilities[2] - implied_probs[2]
        st.metric(
            f"{home_team} (Home)",
            f"{probabilities[2]:.1%}",
            f"Value: {value:+.1%}",
            delta_color="normal" if value > 0 else "off"
        )


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


def display_team_stats(home_stats, away_stats, home_team, away_team):
    st.header("Team Statistics")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{home_team}")
        st.metric("xG%", f"{home_stats['xGoalsPercentage']:.1f}%")
        st.metric("Corsi%", f"{home_stats['corsiPercentage']:.1f}%")
        st.metric("Recent Win Rate", f"{home_stats['recent_win_rate'] * 100:.1f}%")

    with col2:
        st.subheader(f"{away_team}")
        st.metric("xG%", f"{away_stats['xGoalsPercentage']:.1f}%")
        st.metric("Corsi%", f"{away_stats['corsiPercentage']:.1f}%")
        st.metric("Recent Win Rate", f"{away_stats['recent_win_rate'] * 100:.1f}%")

def main():
    st.title("NHL Game Predictor 🏒")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", get_teams())
        home_odds = st.number_input("Home Odds", min_value=1.01, value=2.0, step=0.05)

    with col2:
        away_team = st.selectbox("Away Team", get_teams())
        away_odds = st.number_input("Away Odds", min_value=1.01, value=2.0, step=0.05)

    draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.5, step=0.05)

    if st.button("Get Prediction", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams")
            return

        probabilities, message = calculate_prediction(
            home_team, away_team,
            home_odds, away_odds, draw_odds
        )

        if probabilities is None:
            st.error(message)
        else:
            display_prediction_results(
                probabilities,
                home_team, away_team,
                home_odds, away_odds, draw_odds
            )

            # Show team stats
            home_stats = get_team_stats(home_team)
            away_stats = get_team_stats(away_team)
            display_team_stats(home_stats, away_stats, home_team, away_team)

        st.markdown("---")
        st.markdown("""
               <div style='text-align: center'>
                   <p>NHL Game Predictor v5.0 | Quality-First Predictions</p>
               </div>
               """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
