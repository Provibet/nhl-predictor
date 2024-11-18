import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")


# Model loading with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load('nhl_game_predictor_ensemble_v4.1_balanced.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


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


def get_team_recent_stats(team, days_back=30):
    engine = init_connection()
    cutoff_date = datetime.now() - timedelta(days=days_back)

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
        AVG("faceOffsWonFor") as faceoffs_won,
        AVG("scoreAdjustedTotalShotCreditFor") as shot_credit,
        AVG("penaltiesFor") as penalties_for,
        AVG("penaltiesAgainst") as penalties_against
    FROM nhl24_matchups_with_situations
    WHERE team = %(team)s
    AND date >= %(cutoff_date)s
    GROUP BY team
    """
    return pd.read_sql(query, engine, params={'team': team, 'cutoff_date': cutoff_date}).iloc[0]


def get_head_to_head_stats(home_team, away_team):
    engine = init_connection()
    query = """
    WITH game_results AS (
        SELECT 
            game_date,
            home_team,
            away_team,
            home_team_score,
            away_team_score,
            CASE 
                WHEN home_team_score > away_team_score THEN 'home'
                WHEN away_team_score > home_team_score THEN 'away'
                ELSE 'draw'
            END as winner
        FROM nhl24_results
        WHERE (home_team = %(home_team)s AND away_team = %(away_team)s)
           OR (home_team = %(away_team)s AND away_team = %(home_team)s)
        ORDER BY game_date DESC
        LIMIT 5
    )
    SELECT 
        COUNT(*) as total_games,
        SUM(CASE WHEN winner = 'home' THEN 1 ELSE 0 END) as home_wins,
        SUM(CASE WHEN winner = 'away' THEN 1 ELSE 0 END) as away_wins,
        SUM(CASE WHEN winner = 'draw' THEN 1 ELSE 0 END) as draws,
        AVG(home_team_score + away_team_score) as avg_total_goals
    FROM game_results
    """
    return pd.read_sql(query, engine, params={'home_team': home_team, 'away_team': away_team}).iloc[0]


def prepare_prediction_features(home_team, away_team, home_stats, away_stats, h2h_stats):
    """Prepare features for model prediction"""
    features = pd.DataFrame({
        # Team performance metrics
        'home_xGoalsPercentage': [home_stats['xg_percentage']],
        'away_xGoalsPercentage': [away_stats['xg_percentage']],
        'home_corsiPercentage': [home_stats['corsi_percentage']],
        'away_corsiPercentage': [away_stats['corsi_percentage']],
        'home_fenwickPercentage': [home_stats['fenwick_percentage']],
        'away_fenwickPercentage': [away_stats['fenwick_percentage']],

        # Recent form
        'home_recent_goals_for': [home_stats['goals_for']],
        'home_recent_goals_against': [home_stats['goals_against']],
        'away_recent_goals_for': [away_stats['goals_for']],
        'away_recent_goals_against': [away_stats['goals_against']],

        # Head to head features
        'h2h_home_win_pct': [
            h2h_stats['home_wins'] / h2h_stats['total_games'] if h2h_stats['total_games'] > 0 else 0.5],
        'h2h_games_played': [h2h_stats['total_games']],
        'h2h_avg_total_goals': [h2h_stats['avg_total_goals']],

        # Market implied probabilities
        'home_implied_prob_normalized': [0.4],  # Default values, will be updated with odds
        'away_implied_prob_normalized': [0.4],
        'draw_implied_prob_normalized': [0.2]
    })

    return features


def calculate_implied_probs(home_odds, away_odds, draw_odds):
    """Calculate normalized implied probabilities from odds"""
    home_imp = 1 / home_odds
    away_imp = 1 / away_odds
    draw_imp = 1 / draw_odds

    # Normalize
    total = home_imp + away_imp + draw_imp
    return home_imp / total, away_imp / total, draw_imp / total


def make_prediction(model, features):
    """Make prediction and get probabilities"""
    try:
        probabilities = model.predict_proba(features)[0]
        return {
            'home_win': probabilities[2],
            'away_win': probabilities[0],
            'draw': probabilities[1]
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


def display_prediction_results(probabilities, home_team, away_team, home_odds, away_odds, draw_odds):
    """Display prediction results with visualizations"""
    st.header("🎯 Prediction Results")

    # Calculate betting values
    home_value = probabilities['home_win'] - (1 / home_odds)
    away_value = probabilities['away_win'] - (1 / away_odds)
    draw_value = probabilities['draw'] - (1 / draw_odds)

    # Display probabilities
    cols = st.columns(3)
    with cols[0]:
        st.metric("Home Win Probability", f"{probabilities['home_win'] * 100:.1f}%",
                  f"Value: {home_value * 100:+.1f}%")
    with cols[1]:
        st.metric("Draw Probability", f"{probabilities['draw'] * 100:.1f}%",
                  f"Value: {draw_value * 100:+.1f}%")
    with cols[2]:
        st.metric("Away Win Probability", f"{probabilities['away_win'] * 100:.1f}%",
                  f"Value: {away_value * 100:+.1f}%")

    # Create probability visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f'{home_team}\nHome Win', 'Draw', f'{away_team}\nAway Win']
    probs = [probabilities['home_win'], probabilities['draw'], probabilities['away_win']]
    colors = ['#2ecc71', '#95a5a6', '#e74c3c']

    bars = ax.bar(labels, probs, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Predicted Outcome Probabilities')

    # Add percentage labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height * 100:.1f}%',
                ha='center', va='bottom')

    st.pyplot(fig)

    # Betting recommendations
    st.subheader("💰 Betting Analysis")
    value_threshold = 0.05  # 5% value threshold

    value_dict = {
        "Home Win": (home_value, home_odds),
        "Draw": (draw_value, draw_odds),
        "Away Win": (away_value, away_odds)
    }

    best_value = max(value_dict.items(), key=lambda x: x[1][0])

    if best_value[1][0] > value_threshold:
        st.success(f"Best Value Bet: {best_value[0]} @ {best_value[1][1]:.2f}")
        st.write(f"Expected Value: {best_value[1][0] * 100:.1f}%")
    else:
        st.warning("No significant betting value found in this match")


# Main app layout
st.title("NHL Game Predictor 🏒")

# Load model
model = load_model()

if model is None:
    st.error("Failed to load prediction model. Please check the model file.")
else:
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
                home_stats = get_team_recent_stats(home_team)
                away_stats = get_team_recent_stats(away_team)
                h2h_stats = get_head_to_head_stats(home_team, away_team)

                # Prepare features
                features = prepare_prediction_features(
                    home_team, away_team, home_stats, away_stats, h2h_stats
                )

                # Update market probabilities
                home_imp, away_imp, draw_imp = calculate_implied_probs(home_odds, away_odds, draw_odds)
                features['home_implied_prob_normalized'] = home_imp
                features['away_implied_prob_normalized'] = away_imp
                features['draw_implied_prob_normalized'] = draw_imp

                # Make prediction
                prediction = make_prediction(model, features)

                if prediction:
                    # Display results
                    display_prediction_results(
                        prediction, home_team, away_team,
                        home_odds, away_odds, draw_odds
                    )

                    # Display team stats
                    st.header("📊 Team Statistics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader(f"{home_team} (Home)")
                        st.metric("Goals For", f"{home_stats['goals_for']:.2f}")
                        st.metric("xG%", f"{home_stats['xg_percentage']:.1f}%")
                        st.metric("Corsi%", f"{home_stats['corsi_percentage']:.1f}%")

                    with col2:
                        st.subheader(f"{away_team} (Away)")
                        st.metric("Goals For", f"{away_stats['goals_for']:.2f}")
                        st.metric("xG%", f"{away_stats['xg_percentage']:.1f}%")
                        st.metric("Corsi%", f"{away_stats['corsi_percentage']:.1f}%")

                    # H2H Stats
                    st.header("🤼 Head-to-Head Stats")
                    st.metric("Previous Meetings", int(h2h_stats['total_games']))
                    st.metric("Average Total Goals", f"{h2h_stats['avg_total_goals']:.1f}")

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
    3. Click "Analyze Match" for predictions and betting analysis

    📈 The model considers recent form, head-to-head history, and market odds to make predictions.
""")