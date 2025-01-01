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


@st.cache_resource
def load_model_from_drive():
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )

        service = build('drive', 'v3', credentials=credentials)
        file_id = st.secrets["drive_folder_id"]
        model_name = "nhl_game_predictor_ensemble_v4.1_balanced.pkl"

        query = f"name='{model_name}' and '{file_id}' in parents"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])

        if not files:
            st.error("Model file not found in Drive")
            return None

        request = service.files().get_media(fileId=files[0]['id'])
        file_handle = io.BytesIO()
        downloader = MediaIoBaseDownload(file_handle, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()

        file_handle.seek(0)
        return joblib.load(file_handle)

    except Exception as e:
        st.error(f"Error loading model from Google Drive: {str(e)}")
        return None

def prepare_model_features(home_stats, away_stats, h2h_stats, home_odds, away_odds, draw_odds):
    """Prepare features in format expected by model"""
    
    # Safe conversion helper
    def safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
            
    features = {
        # H2H features
        'h2h_home_win_pct': safe_float(h2h_stats['home_team_wins']) / max(safe_float(h2h_stats['games_played']), 1),
        'h2h_games_played': safe_float(h2h_stats['games_played']),
        'h2h_avg_total_goals': 5.5,  # Default if no h2h games
        'draw_implied_prob_normalized': (1 / draw_odds) / ((1 / home_odds) + (1 / away_odds) + (1 / draw_odds)),

        # Recent performance ratios
        'relative_recent_wins_ratio': safe_float(home_stats['recent_win_rate']) / max(safe_float(away_stats['recent_win_rate']), 0.001),
        'relative_recent_goals_avg_ratio': safe_float(home_stats['goalsFor']) / max(safe_float(away_stats['goalsFor']), 0.001),
        'relative_recent_goals_allowed_ratio': safe_float(home_stats['goalsAgainst']) / max(safe_float(away_stats['goalsAgainst']), 0.001),

        # Goalie ratios
        'relative_goalie_save_pct_ratio': 1.0,
        'relative_goalie_games_ratio': 1.0,

        # Team scoring ratios 
        'relative_team_goals_per_game_ratio': safe_float(home_stats['goalsFor']) / max(safe_float(away_stats['goalsFor']), 0.001),
        'relative_team_top_scorer_goals_ratio': 1.0,

        # Market ratio
        'relative_implied_prob_normalized_ratio': ((1/home_odds) / ((1/home_odds) + (1/away_odds) + (1/draw_odds))) / 
                                                max((1/away_odds) / ((1/home_odds) + (1/away_odds) + (1/draw_odds)), 0.001),

        # Advanced stats ratios
        'relative_xGoalsPercentage_ratio': safe_float(home_stats['xGoalsPercentage']) / max(safe_float(away_stats['xGoalsPercentage']), 0.001),
        'relative_corsiPercentage_ratio': safe_float(home_stats['corsiPercentage']) / max(safe_float(away_stats['corsiPercentage']), 0.001),
        'relative_fenwickPercentage_ratio': safe_float(home_stats['fenwickPercentage']) / max(safe_float(away_stats['fenwickPercentage']), 0.001)
    }

    return pd.DataFrame([features])

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

    if home_stats is None or away_stats is None:
        return False, "Missing team stats"

    # Convert to float to avoid pandas Series boolean ambiguity
    if float(home_stats['games_played']) < MIN_GAMES or float(away_stats['games_played']) < MIN_GAMES:
        return False, "Insufficient recent games"

    return True, "Data quality checks passed"

def calculate_h2h_importance(h2h_stats):
    """Adjust feature weights based on h2h data availability"""
    if float(h2h_stats['games_played']) >= 2:
        return 0.20  # Full h2h weight when we have enough games
    return 0.0  # No h2h weight when insufficient data


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


def calculate_prediction(home_team, away_team, home_odds, away_odds, draw_odds, model):
    """Make prediction using trained model"""
    # Get stats
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    h2h_stats = get_head_to_head_stats(home_team, away_team)

    # Validate data
    is_valid, message = validate_data_quality(home_stats, away_stats, h2h_stats)
    if not is_valid:
        return None, message

    # Prepare features
    features_df = prepare_model_features(home_stats, away_stats, h2h_stats, home_odds, away_odds, draw_odds)

    # Make prediction
    probabilities = model.predict_proba(features_df)[0]

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

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model_from_drive()

    if model is None:
        st.error("Failed to load model. Please check the connection.")
        return

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
            home_odds, away_odds, draw_odds,
            model
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

            # Add h2h info if available
            h2h_stats = get_head_to_head_stats(home_team, away_team)
            if float(h2h_stats['games_played']) > 0:
                st.header("Head to Head History")
                st.write(f"Previous Meetings: {int(h2h_stats['games_played'])}")
                home_wins = int(h2h_stats['home_team_wins'])
                away_wins = int(h2h_stats['games_played'] - h2h_stats['home_team_wins'])
                st.write(f"{home_team} Wins: {home_wins}")
                st.write(f"{away_team} Wins: {away_wins}")

            # Add feature explanation
            with st.expander("Feature Importance Details"):
                st.write("Model considers:")
                st.write("- Recent team performance metrics (xG%, Corsi%, Win Rate)")
                st.write("- Head-to-head history (if available)")
                st.write("- Market odds and implied probabilities")

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>NHL Game Predictor v5.0 | Quality-First Predictions</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
