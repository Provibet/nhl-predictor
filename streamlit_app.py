import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sqlalchemy.sql import text as sql_text
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from datetime import datetime, date, timedelta
import pytz
from typing import Dict, List, Tuple
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style for better visualizations
plt.style.use('ggplot')

def add_betting_recommendations(st, home_team, away_team, home_prob, away_prob, draw_prob,
                                home_odds, away_odds, draw_odds, stake, confidence_score):
    """Add betting recommendations with a more conservative, edge-based approach"""
    st.markdown("### 💰 Betting Recommendations")

    # Calculate implied probabilities and market efficiency
    home_implied_prob = 1 / home_odds
    away_implied_prob = 1 / away_odds
    draw_implied_prob = 1 / draw_odds
    market_efficiency = home_implied_prob + away_implied_prob + draw_implied_prob

    # Define the underdog confidence thresholds and required odds differentials
    underdog_thresholds = {
        30: 2.95,
        31: 0.75,
        32: 8.83,
        33: 7.17,
        34: 5.75,
        35: 4.55,
        36: 3.55,
        37: 2.73,
        38: 2.07,
        39: 1.55,
        40: 1.15,
        41: 0.85,
        42: 0.63,
        43: 0.47,
        44: 0.35,
        45: 0.25
    }

    # Check for underdog override
    def check_underdog_override(underdog_prob, underdog_odds, favorite_odds):
        underdog_percentage = underdog_prob * 100

        # Find the applicable threshold
        for threshold, required_diff in underdog_thresholds.items():
            if underdog_percentage <= threshold:
                # Check if odds differential is sufficient
                actual_diff = underdog_odds - favorite_odds
                if actual_diff > required_diff:
                    return True
        return False

    # Determine if we have an underdog override situation
    override_bet = None
    if home_implied_prob > away_implied_prob:  # Home team is favorite
        if check_underdog_override(away_prob, away_odds, home_odds):
            override_bet = "Away"
    else:  # Away team is favorite
        if check_underdog_override(home_prob, home_odds, away_odds):
            override_bet = "Home"

    if override_bet:
        # Create override recommendation
        betting_options = {
            "Home": {
                "team": home_team,
                "odds": home_odds,
                "prob": home_prob,
                "implied_prob": home_implied_prob,
                "edge": 0,
                "stake": stake if override_bet == "Home" else 0
            },
            "Away": {
                "team": away_team,
                "odds": away_odds,
                "prob": away_prob,
                "implied_prob": away_implied_prob,
                "edge": 0,
                "stake": stake if override_bet == "Away" else 0
            },
            "Draw": {
                "team": "Draw",
                "odds": draw_odds,
                "prob": draw_prob,
                "implied_prob": draw_implied_prob,
                "edge": 0,
                "stake": 0
            }
        }

        best_bet = (override_bet, betting_options[override_bet])

        # Display override recommendation
        st.markdown(
            f"""
            <div style='background-color: #4a1c7c; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; margin-top: 0;'>🎯 UNDERDOG VALUE BET</h4>
                <p style='color: white; font-size: 18px;'>
                    <strong>{best_bet[1]['team']}</strong> ({best_bet[0]})
                </p>
                <ul style='color: white;'>
                    <li>Odds: {best_bet[1]['odds']:.2f}</li>
                    <li>Model Probability: {best_bet[1]['prob']:.1%}</li>
                    <li>Recommended Stake: £{best_bet[1]['stake']:.2f}</li>
                    <li>Override Type: Underdog Value Play</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.info(
            "ℹ️ This bet is recommended based on underdog value criteria, overriding standard model recommendation.")

        return {
            "best_bet": best_bet[0],
            "best_edge": 0,
            "recommended_stake": best_bet[1]["stake"],
            "all_edges": {k: v["edge"] for k, v in betting_options.items()},
            "market_efficiency": market_efficiency
        }

    # If no override, continue with original logic
    # Normalize implied probabilities to account for overround
    home_implied_prob_norm = home_implied_prob / market_efficiency
    away_implied_prob_norm = away_implied_prob / market_efficiency
    draw_implied_prob_norm = draw_implied_prob / market_efficiency

    # Calculate edges (difference between model probability and normalized implied probability)
    home_edge = home_prob - home_implied_prob_norm
    away_edge = away_prob - away_implied_prob_norm
    draw_edge = draw_prob - draw_implied_prob_norm

    # Determine stake based on confidence and edge
    def get_recommended_stake(edge: float, conf_score: float, base_stake: float) -> float:
        if conf_score < 0.25:  # Low confidence
            return 0
        elif conf_score < 0.40:  # Medium confidence
            if edge > 0.10:
                return base_stake * 0.5
            return 0
        else:  # High confidence
            if edge > 0.15:
                return base_stake
            elif edge > 0.10:
                return base_stake * 0.75
            elif edge > 0.05:
                return base_stake * 0.5
            return 0

    # Calculate recommended stakes
    recommended_stakes = {
        "Home": get_recommended_stake(home_edge, confidence_score, stake),
        "Away": get_recommended_stake(away_edge, confidence_score, stake),
        "Draw": get_recommended_stake(draw_edge, confidence_score, stake)
    }

    # Prepare betting options with all relevant information
    betting_options = {
        "Home": {
            "team": home_team,
            "odds": home_odds,
            "prob": home_prob,
            "implied_prob": home_implied_prob_norm,
            "edge": home_edge,
            "stake": recommended_stakes["Home"]
        },
        "Away": {
            "team": away_team,
            "odds": away_odds,
            "prob": away_prob,
            "implied_prob": away_implied_prob_norm,
            "edge": away_edge,
            "stake": recommended_stakes["Away"]
        },
        "Draw": {
            "team": "Draw",
            "odds": draw_odds,
            "prob": draw_prob,
            "implied_prob": draw_implied_prob_norm,
            "edge": draw_edge,
            "stake": recommended_stakes["Draw"]
        }
    }

    # Find best bet (highest edge with recommended stake > 0)
    valid_bets = {k: v for k, v in betting_options.items() if v["stake"] > 0}
    if valid_bets:
        best_bet = max(valid_bets.items(), key=lambda x: x[1]["edge"])

        # Display recommendation
        confidence_color = (
            "#1a472a" if confidence_score >= 0.4
            else "#2a4d1a" if confidence_score >= 0.25
            else "#4d1a1a"
        )

        st.markdown(
            f"""
            <div style='background-color: {confidence_color}; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; margin-top: 0;'>🎯 RECOMMENDED BET</h4>
                <p style='color: white; font-size: 18px;'>
                    <strong>{best_bet[1]['team']}</strong> ({best_bet[0]})
                </p>
                <ul style='color: white;'>
                    <li>Odds: {best_bet[1]['odds']:.2f}</li>
                    <li>Model Probability: {best_bet[1]['prob']:.1%}</li>
                    <li>Edge: {best_bet[1]['edge']:.1%}</li>
                    <li>Confidence Score: {confidence_score:.2f}</li>
                    <li>Recommended Stake: £{best_bet[1]['stake']:.2f}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Add warning messages based on confidence
        if confidence_score < 0.25:
            st.warning("⚠️ Low model confidence - consider skipping this bet")
        elif confidence_score < 0.4:
            st.info("ℹ️ Medium confidence - consider reducing stake")
    else:
        st.warning("⚠️ No significant edges found - no bet recommended")

    # Add detailed edge analysis
    st.markdown("#### Edge Analysis")
    col1, col2, col3 = st.columns(3)

    def display_edge_analysis(col, bet_type, data):
        with col:
            edge_color = (
                "red" if data["edge"] <= 0
                else "orange" if data["edge"] < 0.05
                else "green"
            )
            col.metric(
                f"{data['team']} ({bet_type})",
                f"Edge: {data['edge']:.1%}",
                f"Stake: £{data['stake']:.2f}",
                delta_color="normal" if data["stake"] > 0 else "off"
            )

    display_edge_analysis(col1, "Home", betting_options["Home"])
    display_edge_analysis(col2, "Draw", betting_options["Draw"])
    display_edge_analysis(col3, "Away", betting_options["Away"])

    # Add methodology explanation
    with st.expander("ℹ️ How are recommendations calculated?"):
        st.markdown("""
        Our betting recommendations are based on three key factors:

        1. **Price Edge**: The difference between our model's predicted probability and the market-implied probability
        2. **Model Confidence**: How certain the model is about its prediction
        3. **Conservative Staking**: Stakes are adjusted based on both edge size and confidence level

        Stake recommendations:
        - High confidence (≥0.40):
            * Large edge (>15%): Full stake
            * Medium edge (10-15%): 75% stake
            * Small edge (5-10%): 50% stake
        - Medium confidence (0.25-0.40):
            * Large edge (>10%): 50% stake
            * Otherwise: No bet
        - Low confidence (<0.25):
            * No bet recommended

        This conservative approach helps manage risk and ensure we only bet when we have a clear advantage.
        """)

    return {
        "best_bet": best_bet[0] if valid_bets else None,
        "best_edge": best_bet[1]["edge"] if valid_bets else 0,
        "recommended_stake": best_bet[1]["stake"] if valid_bets else 0,
        "all_edges": {k: v["edge"] for k, v in betting_options.items()},
        "market_efficiency": market_efficiency
    }

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


def get_form_guide(home_team, away_team, last_n=5):
    """Get head-to-head form guide between two teams"""
    engine = init_connection()
    query = f"""
    SELECT 
        game_date,
        home_team,
        away_team,
        CAST(home_team_score AS INTEGER) as home_score,
        CAST(away_team_score AS INTEGER) as away_score,
        CASE 
            WHEN CAST(home_team_score AS INTEGER) > CAST(away_team_score AS INTEGER) THEN home_team
            WHEN CAST(away_team_score AS INTEGER) > CAST(home_team_score AS INTEGER) THEN away_team
            ELSE 'Draw'
        END as winner
    FROM nhl24_results
    WHERE (home_team = '{home_team}' AND away_team = '{away_team}')
       OR (home_team = '{away_team}' AND away_team = '{home_team}')
    ORDER BY game_date DESC
    LIMIT {last_n}
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        # Add formatted result column
        df['result'] = df.apply(lambda row:
                                f"{row['home_team']} {row['home_score']} - {row['away_score']} {row['away_team']}",
                                axis=1
                                )

        # Format date
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # Select and rename columns
        return df[['game_date', 'result', 'winner']].rename(columns={
            'game_date': 'Date',
            'result': 'Score',
            'winner': 'Winner'
        })

    return pd.DataFrame()


def get_team_form(team, last_n=5):
    """Get recent form for a single team"""
    engine = init_connection()
    query = f"""
    SELECT 
        game_date,
        CASE 
            WHEN home_team = '{team}' THEN away_team
            ELSE home_team
        END as opponent,
        CASE 
            WHEN home_team = '{team}' 
            THEN home_team || ' ' || home_team_score || ' - ' || away_team_score || ' ' || away_team
            ELSE home_team || ' ' || home_team_score || ' - ' || away_team_score || ' ' || away_team
        END as result,
        CASE 
            WHEN (home_team = '{team}' AND CAST(home_team_score AS INTEGER) > CAST(away_team_score AS INTEGER)) OR
                 (away_team = '{team}' AND CAST(away_team_score AS INTEGER) > CAST(home_team_score AS INTEGER))
            THEN 'W'
            WHEN (home_team = '{team}' AND CAST(home_team_score AS INTEGER) < CAST(away_team_score AS INTEGER)) OR
                 (away_team = '{team}' AND CAST(away_team_score AS INTEGER) < CAST(home_team_score AS INTEGER))
            THEN 'L'
            ELSE 'D'
        END as outcome
    FROM nhl24_results
    WHERE home_team = '{team}' OR away_team = '{team}'
    ORDER BY game_date DESC
    LIMIT {last_n}
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        # Format date
        df['game_date'] = pd.to_datetime(df['game_date']).dt.strftime('%Y-%m-%d')

        # Select and rename columns
        return df[['game_date', 'opponent', 'result', 'outcome']].rename(columns={
            'game_date': 'Date',
            'opponent': 'Opponent',
            'result': 'Score',
            'outcome': 'Result'
        })

    return pd.DataFrame()

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
            'home_recent_wins': safe_get(home_stats, 'recent_wins', 0.5),
            'home_recent_goals_for': safe_get(home_stats, 'recent_goals_for', 2.5),
            'home_recent_goals_against': safe_get(home_stats, 'recent_goals_against', 2.5),
            'away_recent_wins': safe_get(away_stats, 'recent_wins', 0.5),
            'away_recent_goals_for': safe_get(away_stats, 'recent_goals_for', 2.5),
            'away_recent_goals_against': safe_get(away_stats, 'recent_goals_against', 2.5),
            'h2h_home_wins': safe_get(h2h_stats, 'home_team_wins', 0) / max(h2h_stats['games_played'], 1),
            'h2h_home_goals': safe_get(h2h_stats, 'home_team_avg_goals', 2.5),
            'h2h_away_goals': safe_get(h2h_stats, 'away_team_avg_goals', 2.5),
            'relative_goalsFor': (safe_get(home_stats, 'recent_goals_for', 2.5) -
                                  safe_get(away_stats, 'recent_goals_for', 2.5)),
            'relative_shotsOnGoalFor': (safe_get(home_stats, 'recent_shots_for', 30) -
                                        safe_get(away_stats, 'recent_shots_for', 30)),
            'relative_xGoalsPercentage': (safe_get(home_stats, 'recent_xgoals_pct', 50.0) -
                                          safe_get(away_stats, 'recent_xgoals_pct', 50.0)),
            'relative_corsiPercentage': (safe_get(home_stats, 'recent_corsi_pct', 50.0) -
                                         safe_get(away_stats, 'recent_corsi_pct', 50.0)),
            'home_historical_advantage': safe_get(h2h_stats, 'home_team_wins', 0) / max(
                h2h_stats['games_played'], 1),
            'home_recent_form': safe_get(home_stats, 'recent_goals_for', 2.5),
            'away_recent_form': safe_get(away_stats, 'recent_goals_for', 2.5),
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


def calculate_confidence(probabilities, ensemble_model, features):
    """
    Calculate prediction confidence using probability spread and ensemble agreement
    """
    home_prob, away_prob, draw_prob = probabilities

    # Method 1: Check probability spread
    prob_spread = max(home_prob, away_prob, draw_prob) - min(home_prob, away_prob, draw_prob)

    # Method 2: Check ensemble agreement
    try:
        individual_predictions = [
            ensemble_model.estimators_[0].predict_proba(features)[0],  # XGBoost
            ensemble_model.estimators_[1].predict_proba(features)[0],  # LightGBM
            ensemble_model.estimators_[2].predict_proba(features)[0]  # CatBoost
        ]

        # Calculate agreement between models
        model_agreement = np.std([np.argmax(pred) for pred in individual_predictions])

        # Combine factors
        confidence_score = (prob_spread * 0.7) + ((1 - model_agreement) * 0.3)
    except:
        # Fallback if ensemble details aren't accessible
        confidence_score = prob_spread

    return confidence_score


def save_prediction_to_db(engine, prediction_data: dict):
    """Save prediction data to database"""
    try:
        # Create predictions table if it doesn't exist
        create_table_query = """
        CREATE TABLE IF NOT EXISTS nhl_prediction_tracking (
            id SERIAL PRIMARY KEY,
            prediction_time TIMESTAMP,
            game_date DATE,
            home_team VARCHAR(100),
            away_team VARCHAR(100),
            home_odds FLOAT,
            away_odds FLOAT,
            draw_odds FLOAT,
            home_prob FLOAT,
            away_prob FLOAT,
            draw_prob FLOAT,
            confidence_score FLOAT,
            best_bet VARCHAR(20),
            best_edge FLOAT,
            recommended_stake FLOAT,
            market_efficiency FLOAT,
            actual_result VARCHAR(20) NULL,
            actual_score VARCHAR(20) NULL,
            profit_loss FLOAT NULL,
            notes TEXT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        engine.execute(create_table_query)

        # Prepare data for insertion
        insert_query = """
        INSERT INTO nhl_prediction_tracking (
            prediction_time, game_date, home_team, away_team,
            home_odds, away_odds, draw_odds,
            home_prob, away_prob, draw_prob,
            confidence_score, best_bet, best_edge,
            recommended_stake, market_efficiency
        ) VALUES (
            %(prediction_time)s, %(game_date)s, %(home_team)s, %(away_team)s,
            %(home_odds)s, %(away_odds)s, %(draw_odds)s,
            %(home_prob)s, %(away_prob)s, %(draw_prob)s,
            %(confidence_score)s, %(best_bet)s, %(best_edge)s,
            %(recommended_stake)s, %(market_efficiency)s
        )
        """

        with engine.connect() as conn:
            conn.execute(insert_query, prediction_data)

        return True, "Prediction saved successfully!"
    except Exception as e:
        return False, f"Error saving prediction: {str(e)}"


def view_prediction_history():
    """View and update past predictions"""
    engine = init_connection()

    st.markdown("### Prediction History")

    # Get recent predictions
    query = """
    SELECT 
        id,
        prediction_time,
        game_date,
        home_team,
        away_team,
        best_bet,
        recommended_stake,
        actual_result,
        profit_loss
    FROM nhl_prediction_tracking
    ORDER BY game_date DESC, prediction_time DESC
    LIMIT 50
    """

    predictions = pd.read_sql(query, engine)

    if not predictions.empty:
        # Format the DataFrame for display
        display_df = predictions.copy()
        display_df['prediction_time'] = pd.to_datetime(display_df['prediction_time']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['game_date'] = pd.to_datetime(display_df['game_date']).dt.strftime('%Y-%m-%d')

        # Add styling
        def style_profit_loss(val):
            if pd.isna(val):
                return ''
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'

        styled_df = display_df.style.applymap(style_profit_loss, subset=['profit_loss'])

        # Display the DataFrame
        st.dataframe(styled_df)

        # Add update section
        st.markdown("### Update Result")
        selected_id = st.selectbox(
            "Select prediction to update",
            predictions['id'].tolist(),
            format_func=lambda
                x: f"ID {x}: {predictions[predictions['id'] == x]['home_team'].iloc[0]} vs {predictions[predictions['id'] == x]['away_team'].iloc[0]} ({predictions[predictions['id'] == x]['game_date'].iloc[0]})"
        )

        if selected_id:
            selected_pred = predictions[predictions['id'] == selected_id].iloc[0]

            col1, col2 = st.columns(2)
            with col1:
                result = st.selectbox(
                    "Result",
                    ['Home Win', 'Away Win', 'Draw']
                )
                score = st.text_input(
                    "Score (format: 3-2)",
                    placeholder="Enter final score"
                )

            with col2:
                if selected_pred['best_bet'] and selected_pred['recommended_stake']:
                    if result == 'Home Win' and selected_pred['best_bet'] == 'Home':
                        profit = float(selected_pred['recommended_stake']) * (float(selected_pred['home_odds']) - 1)
                    elif result == 'Away Win' and selected_pred['best_bet'] == 'Away':
                        profit = float(selected_pred['recommended_stake']) * (float(selected_pred['away_odds']) - 1)
                    elif result == 'Draw' and selected_pred['best_bet'] == 'Draw':
                        profit = float(selected_pred['recommended_stake']) * (float(selected_pred['draw_odds']) - 1)
                    else:
                        profit = -float(selected_pred['recommended_stake'])

                    st.metric(
                        "Profit/Loss",
                        f"£{profit:.2f}",
                        delta=f"{profit / float(selected_pred['recommended_stake']) * 100:.1f}% ROI"
                    )

            if st.button("Update Result"):
                update_query = """
                UPDATE nhl_prediction_tracking
                SET actual_result = %s,
                    actual_score = %s,
                    profit_loss = %s
                WHERE id = %s
                """

                try:
                    with engine.connect() as conn:
                        conn.execute(
                            update_query,
                            (result, score, profit, selected_id)
                        )
                    st.success("Result updated successfully!")
                except Exception as e:
                    st.error(f"Error updating result: {str(e)}")
    else:
        st.info("No predictions found in the database.")


def main():
    st.title("NHL Game Predictor 🏒")

    try:
        # Load model and get teams
        with st.spinner("Loading model and data..."):
            model = load_model()
            teams = get_teams()

        # Add sidebar navigation
        st.sidebar.markdown("### Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Make Prediction", "View History"]
        )

        if page == "View History":
            view_prediction_history()
            return

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

                    # Calculate confidence score
                    confidence_score = calculate_confidence(
                        [home_prob, away_prob, draw_prob],
                        model,
                        features
                    )

                    # Get betting recommendations
                    betting_info = add_betting_recommendations(
                        st, home_team, away_team,
                        home_prob, away_prob, draw_prob,
                        home_odds, away_odds, draw_odds,
                        stake,
                        confidence_score
                    )

                    # Store this prediction info for later tracking
                    prediction_data = {
                        'prediction_time': datetime.now(),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': home_odds,
                        'away_odds': away_odds,
                        'draw_odds': draw_odds,
                        'home_prob': home_prob,
                        'away_prob': away_prob,
                        'draw_prob': draw_prob,
                        'confidence_score': confidence_score,
                        'best_bet': betting_info['best_bet'],
                        'best_edge': betting_info['best_edge'],
                        'recommended_stake': betting_info['recommended_stake'],
                        'market_efficiency': betting_info['market_efficiency']
                    }

                    # Show prediction results
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
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

                        # Create a confidence meter
                        st.markdown("### 🎯 Prediction Confidence")
                        confidence_col1, confidence_col2 = st.columns([1, 3])
                        with confidence_col1:
                            st.markdown("**Level:**")
                        with confidence_col2:
                            if confidence_score >= 0.4:
                                st.markdown("🟢 **HIGH CONFIDENCE**")
                                st.markdown(
                                    f"Strong model agreement and clear probability distribution (Score: {confidence_score:.2f})")
                            elif confidence_score >= 0.25:
                                st.markdown("🟡 **MEDIUM CONFIDENCE**")
                                st.markdown(
                                    f"Moderate model agreement and probability distribution (Score: {confidence_score:.2f})")
                            else:
                                st.markdown("🔴 **LOW CONFIDENCE**")
                                st.markdown(
                                    f"Low model agreement or unclear probability distribution (Score: {confidence_score:.2f})")

                        # Create visualization
                        fig = create_visualization(
                            home_team, away_team, home_stats, away_stats,
                            [away_prob, home_prob, draw_prob], h2h_stats
                        )
                        st.pyplot(fig)

                    with tab2:
                        # Show team stats with enhanced metrics
                        col8, col9 = st.columns(2)
                        with col8:
                            st.markdown(f"### {home_team} Statistics")
                            home_stats_df = pd.DataFrame({
                                'Metric': [
                                    'Recent Goals For',
                                    'Recent Goals Against',
                                    'Expected Goals',
                                    'Expected Goals Against',
                                    'Recent Form',
                                    'Market Implied Probability'
                                ],
                                'Value': [
                                    f"{float(home_stats['recent_goals_for']):.2f}",
                                    f"{float(home_stats['recent_goals_against']):.2f}",
                                    f"{float(home_stats['recent_xgoals_for']):.2f}",
                                    f"{float(home_stats['recent_xgoals_against']):.2f}",
                                    f"{betting_info['all_edges']['Home'] * 100:+.1f}%",
                                    f"{(1 / home_odds / betting_info['market_efficiency']):.1%}"
                                ]
                            })
                            st.table(home_stats_df)

                        with col9:
                            st.markdown(f"### {away_team} Statistics")
                            away_stats_df = pd.DataFrame({
                                'Metric': [
                                    'Recent Goals For',
                                    'Recent Goals Against',
                                    'Expected Goals',
                                    'Expected Goals Against',
                                    'Recent Form',
                                    'Market Implied Probability'
                                ],
                                'Value': [
                                    f"{float(away_stats['recent_goals_for']):.2f}",
                                    f"{float(away_stats['recent_goals_against']):.2f}",
                                    f"{float(away_stats['recent_xgoals_for']):.2f}",
                                    f"{float(away_stats['recent_xgoals_against']):.2f}",
                                    f"{betting_info['all_edges']['Away'] * 100:+.1f}%",
                                    f"{(1 / away_odds / betting_info['market_efficiency']):.1%}"
                                ]
                            })
                            st.table(away_stats_df)

                    with tab3:
                        # Enhanced head to head stats
                        st.markdown("### Head to Head Statistics")
                        if h2h_stats['games_played'] > 0:
                            # Overall H2H metrics
                            col10, col11, col12 = st.columns(3)
                            with col10:
                                st.metric("Total Games", int(h2h_stats['games_played']))
                                st.metric(f"{home_team} Wins",
                                          int(h2h_stats['home_team_wins']))
                            with col11:
                                st.metric("Average Total Goals",
                                          f"{h2h_stats['avg_total_goals']:.1f}")
                                st.metric("Goal STD Dev",
                                          f"{h2h_stats['std_total_goals']:.1f}")
                            with col12:
                                st.metric(f"{away_team} Wins",
                                          int(h2h_stats['away_team_wins']))
                                draws = (h2h_stats['games_played'] -
                                         h2h_stats['home_team_wins'] -
                                         h2h_stats['away_team_wins'])
                                st.metric("Draws", int(draws))

                            # Recent H2H Meetings
                            st.markdown("#### Recent Head-to-Head Meetings")
                            h2h_form = get_form_guide(home_team, away_team)

                            if not h2h_form.empty:
                                # Create a styled dataframe
                                def highlight_winner(row):
                                    """Highlight the winner in the Score column"""
                                    if row['Winner'] == home_team:
                                        return ['background-color: #e6ffe6'] * len(row)
                                    elif row['Winner'] == away_team:
                                        return ['background-color: #ffe6e6'] * len(row)
                                    return ['background-color: #f0f0f0'] * len(row)

                                # Apply styling to the dataframe
                                styled_df = h2h_form.style.apply(highlight_winner, axis=1)
                                st.dataframe(styled_df)

                                # Add summary statistics
                                st.markdown("#### Recent Form Analysis")
                                recent_stats = {
                                    f"{home_team} Wins": len(h2h_form[h2h_form['Winner'] == home_team]),
                                    f"{away_team} Wins": len(h2h_form[h2h_form['Winner'] == away_team]),
                                    "Draws": len(h2h_form[h2h_form['Winner'] == 'Draw'])
                                }

                                # Display recent form stats
                                recent_cols = st.columns(3)
                                for i, (label, value) in enumerate(recent_stats.items()):
                                    with recent_cols[i]:
                                        st.metric(
                                            f"Recent {label}",
                                            value,
                                            f"{(value / len(h2h_form)) * 100:.1f}%"
                                        )

                                # Add form insights
                                recent_form_insights = []

                                # Check for dominance
                                if recent_stats[f"{home_team} Wins"] >= 3:
                                    recent_form_insights.append(
                                        f"✅ {home_team} has won {recent_stats[f'{home_team} Wins']} "
                                        f"of the last {len(h2h_form)} meetings"
                                    )
                                elif recent_stats[f"{away_team} Wins"] >= 3:
                                    recent_form_insights.append(
                                        f"✅ {away_team} has won {recent_stats[f'{away_team} Wins']} "
                                        f"of the last {len(h2h_form)} meetings"
                                    )

                                # Display insights
                                if recent_form_insights:
                                    st.markdown("#### Form Insights")
                                    for insight in recent_form_insights:
                                        st.markdown(insight)

                        else:
                            st.info("No recent head-to-head matches found")

                            # Show individual team recent form instead
                            st.markdown("#### Recent Form (Individual Teams)")

                            col13, col14 = st.columns(2)
                            with col13:
                                st.markdown(f"**{home_team} Recent Form**")
                                home_form = get_team_form(home_team)
                                if not home_form.empty:
                                    st.dataframe(home_form)
                                else:
                                    st.info("No recent matches found")

                            with col14:
                                st.markdown(f"**{away_team} Recent Form**")
                                away_form = get_team_form(away_team)
                                if not away_form.empty:
                                    st.dataframe(away_form)
                                else:
                                    st.info("No recent matches found")

                    with tab4:
                        # Enhanced betting analysis
                        st.markdown("### Betting Analysis")

                        # Market efficiency metrics
                        st.markdown("#### Market Analysis")
                        market_cols = st.columns(3)
                        with market_cols[0]:
                            st.metric("Market Efficiency",
                                      f"{betting_info['market_efficiency']:.3f}")
                        with market_cols[1]:
                            overround = (betting_info['market_efficiency'] - 1) * 100
                            st.metric("Overround", f"{overround:.1f}%")
                        with market_cols[2]:
                            st.metric("Best Edge Found",
                                      f"{betting_info['best_edge'] * 100:+.1f}%")

                        # Edge analysis for all outcomes
                        st.markdown("#### Edge Analysis")
                        edge_cols = st.columns(3)
                        for i, (outcome, edge) in enumerate(betting_info['all_edges'].items()):
                            with edge_cols[i]:
                                implied_prob = 1 / locals()[f"{outcome.lower()}_odds"] / betting_info[
                                    'market_efficiency']
                                model_prob = locals()[f"{outcome.lower()}_prob"]
                                st.metric(
                                    f"{outcome} Edge",
                                    f"{edge * 100:+.1f}%",
                                    f"Model: {model_prob:.1%} vs Market: {implied_prob:.1%}"
                                )

                        # Stake recommendations
                        if betting_info['best_bet']:
                            st.markdown("#### Stake Recommendation")
                            stake_cols = st.columns(2)
                            with stake_cols[0]:
                                st.metric("Recommended Bet", betting_info['best_bet'])
                            with stake_cols[1]:
                                st.metric("Recommended Stake",
                                          f"£{betting_info['recommended_stake']:.2f}")
                        else:
                            st.warning("No bet recommended based on current edges and confidence")

                    with tab5:
                        # Enhanced risk assessment
                        st.markdown("### Risk Assessment")

                        # Calculate risk metrics
                        prob_spread = max(home_prob, away_prob, draw_prob) - min(home_prob, away_prob, draw_prob)
                        form_difference = abs(
                            float(home_stats['recent_goals_for']) - float(away_stats['recent_goals_for']))
                        h2h_sample_size = int(h2h_stats['games_played'])

                        # Display risk metrics
                        risk_cols = st.columns(3)
                        with risk_cols[0]:
                            st.metric("Probability Spread", f"{prob_spread:.1%}")
                        with risk_cols[1]:
                            st.metric("Form Difference", f"{form_difference:+.1f}")
                        with risk_cols[2]:
                            st.metric("H2H Sample Size", h2h_sample_size)

                        # Risk factors
                        st.markdown("#### Risk Factors")
                        risk_factors = []

                        if prob_spread < 0.15:
                            risk_factors.append("Close probability spread indicates uncertain outcome")
                        if form_difference < 0.5:
                            risk_factors.append("Teams showing similar recent form")
                        if h2h_sample_size < 3:
                            risk_factors.append("Limited head-to-head history")
                        if confidence_score < 0.25:
                            risk_factors.append("Low model confidence")
                        if betting_info['market_efficiency'] > 1.1:
                            risk_factors.append("High market overround")

                        if risk_factors:
                            for factor in risk_factors:
                                st.warning(f"⚠️ {factor}")
                        else:
                            st.success("✅ No significant risk factors identified")

                        # Add key insights
                        st.markdown("#### Key Insights")
                        insights = []

                        # Form comparison
                        if float(home_stats['recent_goals_for']) > float(away_stats['recent_goals_for']) + 0.5:
                            insights.append(f"✅ {home_team} showing stronger recent scoring form")
                        elif float(away_stats['recent_goals_for']) > float(home_stats['recent_goals_for']) + 0.5:
                            insights.append(f"✅ {away_team} showing stronger recent scoring form")

                        # Defense comparison
                        if float(home_stats['recent_goals_against']) < float(away_stats['recent_goals_against']) - 0.5:
                            insights.append(f"✅ {home_team} showing stronger defensive form")
                        elif float(away_stats['recent_goals_against']) < float(
                                home_stats['recent_goals_against']) - 0.5:
                            insights.append(f"✅ {away_team} showing stronger defensive form")

                        # Edge insights
                        best_edge = max(betting_info['all_edges'].values())
                        if best_edge > 0.15:
                            insights.append(f"✅ Strong edge found ({best_edge * 100:.1f}%)")
                        elif best_edge > 0.10:
                            insights.append(f"✅ Moderate edge found ({best_edge * 100:.1f}%)")

                        if insights:
                            for insight in insights:
                                st.markdown(insight)
                        else:
                            st.info("No clear advantages identified between teams")

                    # Add save prediction button in a new container
                    st.markdown("---")
                    save_container = st.container()
                    with save_container:
                        col_save1, col_save2 = st.columns([2, 1])

                        with col_save1:
                            game_date = st.date_input(
                                "Game Date",
                                value=datetime.now().date(),
                                min_value=datetime.now().date()
                            )
                            notes = st.text_area(
                                "Notes (optional)",
                                placeholder="Add any additional notes about this prediction..."
                            )

                        with col_save2:
                            st.markdown("### Save Prediction")
                            if st.button("📝 Save Prediction to Database", type="secondary"):
                                # Update prediction data with game date and notes
                                prediction_data['game_date'] = game_date
                                prediction_data['notes'] = notes if notes else None

                                # Save to database
                                success, message = save_prediction_to_db(
                                    init_connection(),
                                    prediction_data
                                )

                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)

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