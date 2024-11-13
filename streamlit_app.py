import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from datetime import datetime


def add_confidence_indicators(st, home_prob, away_prob, draw_prob):
    """Add clear visual confidence indicators"""
    st.markdown("### 🎯 Confidence Analysis")

    # Calculate overall confidence level
    max_prob = max(home_prob, away_prob, draw_prob)

    # Create confidence meter
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Confidence Level:**")
    with col2:
        if max_prob >= 0.90:
            st.markdown("🟢 **VERY HIGH** (90%+)")
        elif max_prob >= 0.80:
            st.markdown("🟡 **HIGH** (80-90%)")
        elif max_prob >= 0.70:
            st.markdown("🟠 **MEDIUM** (70-80%)")
        else:
            st.markdown("🔴 **LOW** (<70%)")


def add_betting_recommendations(st, home_team, away_team, home_prob, away_prob, draw_prob,
                                home_odds, away_odds, draw_odds, stake):
    """Add clear betting recommendations with reasoning"""
    st.markdown("### 💰 Betting Recommendations")

    # Calculate EVs
    home_ev = (home_prob * (home_odds - 1) * stake) - ((1 - home_prob) * stake)
    away_ev = (away_prob * (away_odds - 1) * stake) - ((1 - away_prob) * stake)
    draw_ev = (draw_prob * (draw_odds - 1) * stake) - ((1 - draw_prob) * stake)

    # Calculate ROI percentages
    home_roi = (home_ev / stake) * 100
    away_roi = (away_ev / stake) * 100
    draw_roi = (draw_ev / stake) * 100

    # Create recommendation box
    best_bet = max(
        ("Home", home_ev, home_roi, home_prob, home_odds, home_team),
        ("Away", away_ev, away_roi, away_prob, away_odds, away_team),
        ("Draw", draw_ev, draw_roi, draw_prob, draw_odds, "Draw"),
        key=lambda x: x[1]
    )

    if best_bet[1] > 0:  # If best EV is positive
        recommendation_box = st.container()
        with recommendation_box:
            st.markdown(
                f"""
                <div style='background-color: #1a472a; padding: 20px; border-radius: 10px;'>
                    <h4 style='color: white; margin-top: 0;'>🎯 RECOMMENDED BET</h4>
                    <p style='color: white; font-size: 18px;'><strong>{best_bet[5]}</strong> ({best_bet[0]})</p>
                    <ul style='color: white;'>
                        <li>Expected Value: £{best_bet[1]:.2f}</li>
                        <li>ROI: {best_bet[2]:.1f}%</li>
                        <li>Win Probability: {best_bet[3]:.1%}</li>
                        <li>Odds: {best_bet[4]:.2f}</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ No positive expected value bets found. Consider skipping this game.")

    # Add detailed value breakdown
    st.markdown("#### Value Breakdown")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"{home_team} (Home)",
            f"£{home_ev:.2f} EV",
            f"{home_roi:.1f}% ROI",
            delta_color="normal" if home_ev > 0 else "off"
        )

    with col2:
        st.metric(
            "Draw",
            f"£{draw_ev:.2f} EV",
            f"{draw_roi:.1f}% ROI",
            delta_color="normal" if draw_ev > 0 else "off"
        )

    with col3:
        st.metric(
            f"{away_team} (Away)",
            f"£{away_ev:.2f} EV",
            f"{away_roi:.1f}% ROI",
            delta_color="normal" if away_ev > 0 else "off"
        )


def add_risk_assessment(st, home_prob, away_prob, draw_prob, h2h_stats, home_stats, away_stats):
    """Add risk assessment indicators"""
    st.markdown("### ⚠️ Risk Assessment")

    # Calculate risk factors
    risk_factors = []

    # Check probability spread
    prob_spread = max(home_prob, away_prob, draw_prob) - min(home_prob, away_prob, draw_prob)
    if prob_spread < 0.15:
        risk_factors.append("Close probability spread indicates uncertain outcome")

    # Check recent form consistency
    if abs(home_stats['recent_goals_for'] - away_stats['recent_goals_for']) < 0.5:
        risk_factors.append("Teams showing similar recent form")

    # Check head-to-head history
    if h2h_stats['games_played'] < 3:
        risk_factors.append("Limited head-to-head history")

    # Display risk factors
    if risk_factors:
        for factor in risk_factors:
            st.warning(f"⚠️ {factor}")
    else:
        st.success("✅ No significant risk factors identified")


def add_key_insights(st, home_team, away_team, home_stats, away_stats, h2h_stats):
    """Add key insights section"""
    st.markdown("### 🔍 Key Insights")

    insights = []

    # Form comparison
    if home_stats['recent_goals_for'] > away_stats['recent_goals_for']:
        insights.append(f"✅ {home_team} showing stronger recent scoring form")
    elif away_stats['recent_goals_for'] > home_stats['recent_goals_for']:
        insights.append(f"✅ {away_team} showing stronger recent scoring form")

    # Defense comparison
    if home_stats['recent_goals_against'] < away_stats['recent_goals_against']:
        insights.append(f"✅ {home_team} showing stronger defensive form")
    elif away_stats['recent_goals_against'] < home_stats['recent_goals_against']:
        insights.append(f"✅ {away_team} showing stronger defensive form")

    # H2H dominance
    if h2h_stats['games_played'] > 0:
        home_win_rate = h2h_stats['home_team_wins'] / h2h_stats['games_played']
        if home_win_rate > 0.6:
            insights.append(f"✅ {home_team} has strong H2H record ({home_win_rate:.0%} win rate)")
        elif home_win_rate < 0.4:
            insights.append(f"✅ {away_team} has strong H2H record ({(1 - home_win_rate):.0%} win rate)")

    # Display insights
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("No clear advantages identified between teams")


# Page config
st.set_page_config(
    page_title="NHL Game Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize database connection
@st.cache_resource
def init_connection():
    return create_engine(st.secrets["db_connection"])


# Load model from Drive
@st.cache_resource
def load_model():
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

# Basic helper functions
@st.cache_data
def get_teams():
    engine = init_connection()
    query = """
    SELECT DISTINCT team 
    FROM nhl24_matchups_with_situations 
    WHERE team IS NOT NULL 
    ORDER BY team
    """
    teams = pd.read_sql(query, engine)
    return teams['team'].tolist()

def safe_get(stats, key, default=0.0):
    try:
        val = stats.get(key, default)
        if val is None or pd.isna(val):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

# Stats collection functions
def get_team_stats(team):
    engine = init_connection()
    query = f"""
    SELECT 
        AVG("goalsFor") as recent_goals_for,
        AVG("goalsAgainst") as recent_goals_against,
        AVG("xGoalsPercentage") as recent_xgoals_pct,
        AVG("corsiPercentage") as recent_corsi_pct,
        AVG("fenwickPercentage") as recent_fenwick_pct,
        AVG("shotsOnGoalFor") as recent_shots_for,
        AVG("shotsOnGoalAgainst") as recent_shots_against,
        AVG("highDangerShotsFor") as recent_hd_shots_for,
        AVG("highDangerGoalsFor") as recent_hd_goals_for,
        AVG("xGoalsFor") as recent_xgoals_for,
        AVG("xGoalsAgainst") as recent_xgoals_against,
        COUNT(*) as games_played
    FROM nhl24_matchups_with_situations
    WHERE team = '{team}'
    """
    return pd.read_sql(query, engine).iloc[0]

def get_head_to_head_stats(home_team, away_team):
    engine = init_connection()
    query = f"""
    WITH h2h_games AS (
        SELECT 
            home_team,
            away_team,
            CAST(home_team_score AS INTEGER) as home_score,
            CAST(away_team_score AS INTEGER) as away_score,
            CAST(home_team_score AS INTEGER) + CAST(away_team_score AS INTEGER) as total_goals
        FROM nhl24_results
        WHERE (home_team = '{home_team}' AND away_team = '{away_team}')
           OR (home_team = '{away_team}' AND away_team = '{home_team}')
        ORDER BY game_date DESC
        LIMIT 10
    )
    SELECT 
        COALESCE(AVG(total_goals), 0) as avg_total_goals,
        COALESCE(STDDEV(total_goals), 0) as std_total_goals,
        COUNT(*) as games_played,
        COALESCE(SUM(CASE 
            WHEN home_team = '{home_team}' AND home_score > away_score THEN 1
            WHEN away_team = '{home_team}' AND away_score > home_score THEN 1
            ELSE 0 
        END), 0) as home_team_wins,
        COALESCE(SUM(CASE 
            WHEN home_team = '{away_team}' AND home_score > away_score THEN 1
            WHEN away_team = '{away_team}' AND away_score > home_score THEN 1
            ELSE 0 
        END), 0) as away_team_wins,
        COALESCE(AVG(CASE WHEN home_team = '{home_team}' THEN home_score ELSE away_score END), 0) as home_team_avg_goals,
        COALESCE(AVG(CASE WHEN home_team = '{away_team}' THEN home_score ELSE away_score END), 0) as away_team_avg_goals
    FROM h2h_games
    """
    return pd.read_sql(query, engine).iloc[0]

def get_form_guide(team, last_n=5):
    engine = init_connection()
    query = f"""
    SELECT 
        game_date,
        CASE 
            WHEN home_team = '{team}' THEN away_team
            ELSE home_team
        END as opponent,
        CASE 
            WHEN home_team = '{team}' THEN CAST(home_team_score AS INTEGER)
            ELSE CAST(away_team_score AS INTEGER)
        END as goals_for,
        CASE 
            WHEN home_team = '{team}' THEN CAST(away_team_score AS INTEGER)
            ELSE CAST(home_team_score AS INTEGER)
        END as goals_against,
        CASE 
            WHEN (home_team = '{team}' AND CAST(home_team_score AS INTEGER) > CAST(away_team_score AS INTEGER)) OR
                 (away_team = '{team}' AND CAST(away_team_score AS INTEGER) > CAST(home_team_score AS INTEGER))
            THEN 'W'
            ELSE 'L'
        END as result
    FROM nhl24_results
    WHERE home_team = '{team}' OR away_team = '{team}'
    ORDER BY game_date DESC
    LIMIT {last_n}
    """
    return pd.read_sql(query, engine)

# Analysis helper functions
def calculate_ev(probability, odds, stake):
    try:
        stake = float(stake)
        potential_profit = stake * (float(odds) - 1)
        ev = (probability * potential_profit) - ((1 - probability) * stake)
        roi_percentage = (potential_profit / stake) * 100
        return ev, roi_percentage
    except (ValueError, TypeError):
        return 0, 0

def format_team_stats(team_name, stats):
    try:
        return f"{team_name} Statistics:\n\n" + \
               f"Goals For/Game: {stats['recent_goals_for']:.2f}\n" + \
               f"Goals Against/Game: {stats['recent_goals_against']:.2f}\n" + \
               f"Expected Goals For: {stats['recent_xgoals_for']:.2f}\n" + \
               f"Expected Goals Against: {stats['recent_xgoals_against']:.2f}\n" + \
               f"Corsi %: {stats['recent_corsi_pct']:.1f}%\n" + \
               f"Fenwick %: {stats['recent_fenwick_pct']:.1f}%\n" + \
               f"Games Played: {stats['games_played']}"
    except Exception as e:
        return f"Error formatting stats for {team_name}: {str(e)}"


def prepare_features(home_team, away_team, home_odds, away_odds, draw_odds):
    """Prepare features for prediction"""
    try:
        # Get team stats
        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)
        h2h_stats = get_head_to_head_stats(home_team, away_team)

        # Calculate implied probabilities
        home_implied_prob = 1 / float(home_odds) if float(home_odds) != 0 else 0.33
        away_implied_prob = 1 / float(away_odds) if float(away_odds) != 0 else 0.33
        draw_implied_prob = 1 / float(draw_odds) if float(draw_odds) != 0 else 0.33
        market_efficiency = home_implied_prob + away_implied_prob + draw_implied_prob

        # Create features dictionary with exact column names
        features = {
            # Team Performance Metrics
            'home_goalsFor': safe_get(home_stats, 'recent_goals_for', 2.5),
            'home_goalsAgainst': safe_get(home_stats, 'recent_goals_against', 2.5),
            'away_goalsFor': safe_get(away_stats, 'recent_goals_for', 2.5),
            'away_goalsAgainst': safe_get(away_stats, 'recent_goals_against', 2.5),
            'home_shotsOnGoalFor': safe_get(home_stats, 'recent_shots_for', 30),
            'home_shotsOnGoalAgainst': safe_get(home_stats, 'recent_shots_against', 30),
            'away_shotsOnGoalFor': safe_get(away_stats, 'recent_shots_for', 30),
            'away_shotsOnGoalAgainst': safe_get(away_stats, 'recent_shots_against', 30),
            'home_xGoalsPercentage': safe_get(home_stats, 'recent_xgoals_pct', 50.0),
            'away_xGoalsPercentage': safe_get(away_stats, 'recent_xgoals_pct', 50.0),
            'home_corsiPercentage': safe_get(home_stats, 'recent_corsi_pct', 50.0),
            'away_corsiPercentage': safe_get(away_stats, 'recent_corsi_pct', 50.0),
            'home_fenwickPercentage': safe_get(home_stats, 'recent_fenwick_pct', 50.0),
            'away_fenwickPercentage': safe_get(away_stats, 'recent_fenwick_pct', 50.0),

            # Recent Form Features
            'home_recent_wins': safe_get(home_stats, 'recent_wins', 0.5),
            'home_recent_goals_for': safe_get(home_stats, 'recent_goals_for', 2.5),
            'home_recent_goals_against': safe_get(home_stats, 'recent_goals_against', 2.5),
            'away_recent_wins': safe_get(away_stats, 'recent_wins', 0.5),
            'away_recent_goals_for': safe_get(away_stats, 'recent_goals_for', 2.5),
            'away_recent_goals_against': safe_get(away_stats, 'recent_goals_against', 2.5),

            # Head-to-Head Features
            'h2h_home_wins': safe_get(h2h_stats, 'home_team_wins', 0) / max(h2h_stats['games_played'], 1),
            'h2h_home_goals': safe_get(h2h_stats, 'home_team_avg_goals', 2.5),
            'h2h_away_goals': safe_get(h2h_stats, 'away_team_avg_goals', 2.5),

            # Relative Strength Metrics
            'relative_goalsFor': (
                        safe_get(home_stats, 'recent_goals_for', 2.5) - safe_get(away_stats, 'recent_goals_for', 2.5)),
            'relative_shotsOnGoalFor': (
                        safe_get(home_stats, 'recent_shots_for', 30) - safe_get(away_stats, 'recent_shots_for', 30)),
            'relative_xGoalsPercentage': (
                        safe_get(home_stats, 'recent_xgoals_pct', 50.0) - safe_get(away_stats, 'recent_xgoals_pct',
                                                                                   50.0)),
            'relative_corsiPercentage': (
                        safe_get(home_stats, 'recent_corsi_pct', 50.0) - safe_get(away_stats, 'recent_corsi_pct',
                                                                                  50.0)),

            # Historical and Form-Based Features
            'home_historical_advantage': safe_get(h2h_stats, 'home_team_wins', 0) / max(h2h_stats['games_played'], 1),
            'home_recent_form': safe_get(home_stats, 'recent_goals_for', 2.5),
            'away_recent_form': safe_get(away_stats, 'recent_goals_for', 2.5),

            # Market Features
            'home_implied_prob_normalized': home_implied_prob / market_efficiency,
            'away_implied_prob_normalized': away_implied_prob / market_efficiency,
            'draw_implied_prob_normalized': draw_implied_prob / market_efficiency,
            'market_efficiency': market_efficiency
        }

        return pd.DataFrame([features]), home_stats, away_stats, h2h_stats

    except Exception as e:
        st.error(f"Failed to prepare features: {str(e)}")
        raise


def create_visualization(home_team, away_team, home_stats, away_stats, probabilities, h2h_stats):
    """Create visualization charts"""
    try:
        fig = plt.Figure(figsize=(12, 8), dpi=100)

        # Create 2x2 subplots
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Prediction Probability Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        labels = [
            f'{away_team}\n{probabilities[0]:.1%}',
            f'{home_team}\n{probabilities[1]:.1%}',
            f'Draw\n{probabilities[2]:.1%}'
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        ax1.pie(probabilities, labels=labels, colors=colors, autopct='%1.1f%%')
        ax1.set_title('Prediction Probabilities')

        # 2. Recent Form Comparison Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['Goals', 'xGoals', 'HD Goals']
        home_values = [
            float(home_stats['recent_goals_for']),
            float(home_stats['recent_xgoals_for']),
            float(home_stats['recent_hd_goals_for'])
        ]
        away_values = [
            float(away_stats['recent_goals_for']),
            float(away_stats['recent_xgoals_for']),
            float(away_stats['recent_hd_goals_for'])
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width / 2, home_values, width, label=home_team, color='#66b3ff')
        ax2.bar(x + width / 2, away_values, width, label=away_team, color='#ff9999')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_title('Recent Offensive Metrics')

        # 3. Advanced Stats Radar Chart
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        categories = ['Goals/Game', 'Shots/Game', 'HD Chances', 'HD Goals', 'Win Rate']

        # Scale factors
        SHOTS_SCALE = 100 / 40
        GOALS_SCALE = 100 / 5
        HD_SCALE = 100 / 20
        HDG_SCALE = 100 / 5

        # Get values with proper scaling
        home_values = [
            float(safe_get(home_stats, 'recent_goals_for', 3)) * GOALS_SCALE,
            float(safe_get(home_stats, 'recent_shots_for', 30)) * SHOTS_SCALE,
            float(safe_get(home_stats, 'recent_hd_shots_for', 10)) * HD_SCALE,
            float(safe_get(home_stats, 'recent_hd_goals_for', 2)) * HDG_SCALE,
            float(safe_get(home_stats, 'recent_wins', 0.5)) * 100
        ]

        away_values = [
            float(safe_get(away_stats, 'recent_goals_for', 3)) * GOALS_SCALE,
            float(safe_get(away_stats, 'recent_shots_for', 30)) * SHOTS_SCALE,
            float(safe_get(away_stats, 'recent_hd_shots_for', 10)) * HD_SCALE,
            float(safe_get(away_stats, 'recent_hd_goals_for', 2)) * HDG_SCALE,
            float(safe_get(away_stats, 'recent_wins', 0.5)) * 100
        ]

        # Normalize values
        home_values = np.clip(home_values, 0, 100)
        away_values = np.clip(away_values, 0, 100)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        home_values = np.concatenate((home_values, [home_values[0]]))
        away_values = np.concatenate((away_values, [away_values[0]]))

        ax3.plot(angles, home_values, 'o-', label=home_team, color='#66b3ff', linewidth=2)
        ax3.fill(angles, home_values, alpha=0.25, color='#66b3ff')
        ax3.plot(angles, away_values, 'o-', label=away_team, color='#ff9999', linewidth=2)
        ax3.fill(angles, away_values, alpha=0.25, color='#ff9999')

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_rticks([20, 40, 60, 80, 100])
        ax3.set_rlabel_position(0)
        ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        ax3.set_title('Key Performance Metrics', pad=15)
        ax3.grid(True)

        # 4. H2H Results Summary
        ax4 = fig.add_subplot(gs[1, 1])
        if h2h_stats['games_played'] > 0:
            h2h_labels = [f'{home_team}\nWins', f'{away_team}\nWins', 'Draws']
            h2h_values = [
                h2h_stats['home_team_wins'],
                h2h_stats['away_team_wins'],
                h2h_stats['games_played'] - h2h_stats['home_team_wins'] - h2h_stats['away_team_wins']
            ]
            ax4.bar(h2h_labels, h2h_values, color=['#66b3ff', '#ff9999', '#99ff99'])
            ax4.set_title('Head-to-Head Results')
        else:
            ax4.text(0.5, 0.5, 'No H2H Data Available',
                     horizontalalignment='center',
                     verticalalignment='center')
            ax4.set_xticks([])
            ax4.set_yticks([])

        fig.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Failed to create visualization: {str(e)}")
        raise


def main():
    st.title("NHL Game Predictor 🏒")

    try:
        # Load model and get teams
        with st.spinner("Loading model and data..."):
            model = load_model()
            teams = get_teams()

        # Create main columns
        col1, col2 = st.columns(2)

        # Team Selection
        with col1:
            st.subheader("Home Team")
            home_team = st.selectbox("Select Home Team", teams, key="home")
            home_odds = st.number_input("Home Team Odds", min_value=1.0, value=2.0, step=0.1)

        with col2:
            st.subheader("Away Team")
            away_team = st.selectbox("Select Away Team", teams, key="away")
            away_odds = st.number_input("Away Team Odds", min_value=1.0, value=2.0, step=0.1)

        # Additional inputs
        st.subheader("Additional Information")
        col3, col4 = st.columns(2)

        with col3:
            draw_odds = st.number_input("Draw Odds", min_value=1.0, value=3.5, step=0.1)
        with col4:
            stake = st.number_input("Stake (£)", min_value=0.0, value=10.0, step=1.0)

        # Make prediction button
        if st.button("Make Prediction", type="primary"):
            if home_team == away_team:
                st.error("Please select different teams")
            else:
                with st.spinner("Analyzing the matchup..."):
                    # Get features and stats
                    features, home_stats, away_stats, h2h_stats = prepare_features(
                        home_team, away_team, home_odds, away_odds, draw_odds
                    )

                    # Get model prediction
                    win_probability = model.predict_proba(features)[0]

                    # Calculate probabilities
                    home_prob = win_probability[1]
                    away_prob = win_probability[0] * 0.8  # Adjust for draw
                    draw_prob = win_probability[0] * 0.2  # Allocate portion to draw

                    # Normalize probabilities
                    total_prob = home_prob + away_prob + draw_prob
                    home_prob /= total_prob
                    away_prob /= total_prob
                    draw_prob /= total_prob

                    add_confidence_indicators(st, home_prob, away_prob, draw_prob)

                    add_betting_recommendations(
                        st, home_team, away_team,
                        home_prob, away_prob, draw_prob,
                        home_odds, away_odds, draw_odds,
                        stake
                    )

                    # Create tabs for different views
                    tab1, tab2, tab3, tab4, tab5= st.tabs([
                        "Prediction Results",
                        "Team Stats",
                        "Head to Head",
                        "Betting Analysis",
                        "Risk Assessment"
                    ])

                    with tab1:
                        # Show prediction results
                        st.markdown("### Prediction Results")

                        # Create metrics for probabilities
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric("Home Win", f"{home_prob:.1%}")
                        with col6:
                            st.metric("Draw", f"{draw_prob:.1%}")
                        with col7:
                            st.metric("Away Win", f"{away_prob:.1%}")

                        # Show confidence indicator
                        predicted_outcome = max(
                            [("Home Win", home_prob),
                             ("Draw", draw_prob),
                             ("Away Win", away_prob)],
                            key=lambda x: x[1]
                        )

                        confidence = predicted_outcome[1]
                        if confidence >= 0.65:
                            st.success("🌟 HIGH CONFIDENCE PREDICTION 🌟")
                        else:
                            st.info("Standard Confidence Prediction")

                        # Create visualization
                        fig = create_visualization(
                            home_team, away_team, home_stats, away_stats,
                            [away_prob, home_prob, draw_prob], h2h_stats
                        )
                        st.pyplot(fig)

                    with tab2:
                        # Show team stats
                        col8, col9 = st.columns(2)
                        with col8:
                            st.markdown(f"### {home_team} Statistics")
                            st.write(format_team_stats(home_team, home_stats))
                        with col9:
                            st.markdown(f"### {away_team} Statistics")
                            st.write(format_team_stats(away_team, away_stats))

                    with tab3:
                        # Show head to head stats
                        st.markdown("### Head to Head Statistics")
                        if h2h_stats['games_played'] > 0:
                            col10, col11 = st.columns(2)
                            with col10:
                                st.metric("Total Games", int(h2h_stats['games_played']))
                                st.metric(f"{home_team} Wins",
                                          int(h2h_stats['home_team_wins']))
                            with col11:
                                st.metric("Average Total Goals",
                                          f"{h2h_stats['avg_total_goals']:.1f}")
                                st.metric(f"{away_team} Wins",
                                          int(h2h_stats['away_team_wins']))
                        else:
                            st.info("No recent head-to-head matches found")

                    with tab4:
                        # Show betting analysis
                        st.markdown("### Betting Analysis")
                        if stake > 0:
                            # Calculate EV for each outcome
                            home_ev, home_roi = calculate_ev(home_prob, home_odds, stake)
                            draw_ev, draw_roi = calculate_ev(draw_prob, draw_odds, stake)
                            away_ev, away_roi = calculate_ev(away_prob, away_odds, stake)

                            # Create metrics for each bet
                            col12, col13, col14 = st.columns(3)
                            with col12:
                                st.metric("Home EV", f"£{home_ev:.2f}")
                                st.metric("Home ROI", f"{home_roi:.1f}%")
                            with col13:
                                st.metric("Draw EV", f"£{draw_ev:.2f}")
                                st.metric("Draw ROI", f"{draw_roi:.1f}%")
                            with col14:
                                st.metric("Away EV", f"£{away_ev:.2f}")
                                st.metric("Away ROI", f"{away_roi:.1f}%")

                            # Show best value bet
                            best_bet = max(
                                [("Home", home_ev, home_roi, home_odds),
                                 ("Draw", draw_ev, draw_roi, draw_odds),
                                 ("Away", away_ev, away_roi, away_odds)],
                                key=lambda x: x[1]
                            )

                            if best_bet[1] > 0:
                                st.success(
                                    f"Best Value Bet: {best_bet[0]} "
                                    f"(EV: £{best_bet[1]:.2f}, "
                                    f"ROI: {best_bet[2]:.1f}%, "
                                    f"Odds: {best_bet[3]:.2f})"
                                )
                            else:
                                st.warning("No positive EV bets available")
                        else:
                            st.warning("Please enter a stake amount for betting analysis")

                    with tab5:
                        add_risk_assessment(st, home_prob, away_prob, draw_prob,
                                         h2h_stats, home_stats, away_stats)
                        add_key_insights(st, home_team, away_team,
                                       home_stats, away_stats, h2h_stats)

        # Add footer with additional information
        st.markdown("---")
        st.markdown(
            """
            ℹ️ **How to use this predictor:**
            1. Select home and away teams
            2. Enter the current odds from your bookmaker
            3. Set your stake amount
            4. Click "Make Prediction" for detailed analysis
            """
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Refresh App"):
            st.experimental_rerun()


if __name__ == "__main__":
    main()