import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
from datetime import datetime

# Database connection
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

# Team logo mappings
# Team logo mappings
team_logos = {
    "Anaheim Ducks": "https://loodibee.com/wp-content/uploads/nhl-anaheim-ducks-logo.png",
    "Arizona Coyotes": "https://loodibee.com/wp-content/uploads/nhl-arizona-coyotes-logo.png",
    "Utah NHL Team": "https://loodibee.com/wp-content/uploads/NHL-Utah-Hockey-Club-Logo.png",  # Added this
    "Utah Hockey Club": "https://loodibee.com/wp-content/uploads/NHL-Utah-Hockey-Club-Logo.png",  # And this
    "Boston Bruins": "https://loodibee.com/wp-content/uploads/nhl-boston-bruins-logo.png",
    "Buffalo Sabres": "https://loodibee.com/wp-content/uploads/nhl-buffalo-sabres-logo.png",
    "Calgary Flames": "https://loodibee.com/wp-content/uploads/nhl-calgary-flames-logo.png",
    "Carolina Hurricanes": "https://loodibee.com/wp-content/uploads/nhl-carolina-hurricanes-logo.png",
    "Chicago Blackhawks": "https://loodibee.com/wp-content/uploads/nhl-chicago-blackhawks-logo.png",
    "Colorado Avalanche": "https://loodibee.com/wp-content/uploads/nhl-colorado-avalanche-logo.png",
    "Columbus Blue Jackets": "https://loodibee.com/wp-content/uploads/nhl-columbus-blue-jackets-logo.png",
    "Dallas Stars": "https://loodibee.com/wp-content/uploads/nhl-dallas-stars-logo.png",
    "Detroit Red Wings": "https://loodibee.com/wp-content/uploads/nhl-detroit-red-wings-logo.png",
    "Edmonton Oilers": "https://loodibee.com/wp-content/uploads/nhl-edmonton-oilers-logo.png",
    "Florida Panthers": "https://loodibee.com/wp-content/uploads/nhl-florida-panthers-logo.png",
    "Los Angeles Kings": "https://loodibee.com/wp-content/uploads/nhl-los-angeles-kings-logo.png",
    "Minnesota Wild": "https://loodibee.com/wp-content/uploads/nhl-minnesota-wild-logo.png",
    "Montreal Canadiens": "https://loodibee.com/wp-content/uploads/nhl-montreal-canadiens-logo.png",
    "Nashville Predators": "https://loodibee.com/wp-content/uploads/nhl-nashville-predators-logo.png",
    "New Jersey Devils": "https://loodibee.com/wp-content/uploads/nhl-new-jersey-devils-logo.png",
    "New York Islanders": "https://loodibee.com/wp-content/uploads/nhl-new-york-islanders-logo.png",
    "New York Rangers": "https://loodibee.com/wp-content/uploads/nhl-new-york-rangers-logo.png",
    "Ottawa Senators": "https://loodibee.com/wp-content/uploads/nhl-ottawa-senators-logo.png",
    "Philadelphia Flyers": "https://loodibee.com/wp-content/uploads/nhl-philadelphia-flyers-logo.png",
    "Pittsburgh Penguins": "https://loodibee.com/wp-content/uploads/nhl-pittsburgh-penguins-logo.png",
    "San Jose Sharks": "https://loodibee.com/wp-content/uploads/nhl-san-jose-sharks-logo.png",
    "Seattle Kraken": "https://loodibee.com/wp-content/uploads/nhl-seattle-kraken-logo.png",
    "St. Louis Blues": "https://loodibee.com/wp-content/uploads/nhl-st-louis-blues-logo.png",
    "Tampa Bay Lightning": "https://loodibee.com/wp-content/uploads/nhl-tampa-bay-lightning-logo.png",
    "Toronto Maple Leafs": "https://loodibee.com/wp-content/uploads/nhl-toronto-maple-leafs-logo.png",
    "Vancouver Canucks": "https://loodibee.com/wp-content/uploads/nhl-vancouver-canucks-logo.png",
    "Vegas Golden Knights": "https://loodibee.com/wp-content/uploads/nhl-vegas-golden-knights-logo.png",
    "Washington Capitals": "https://loodibee.com/wp-content/uploads/nhl-washington-capitals-logo.png",
    "Winnipeg Jets": "https://loodibee.com/wp-content/uploads/nhl-winnipeg-jets-logo.png"
}

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

def get_team_stats(team):
    """Get comprehensive team statistics with proper column references"""
    query = """
    WITH recent_games AS (
        SELECT 
            team,
            "goalsFor",
            "goalsAgainst",
            "xGoalsPercentage",
            "corsiPercentage",
            "fenwickPercentage",
            "shotsOnGoalFor",
            "shotsOnGoalAgainst",
            "highDangerShotsFor",
            "highDangerGoalsFor",
            "xGoalsFor",
            "xGoalsAgainst",
            "highDangerGoalsAgainst",
            "mediumDangerGoalsFor",
            "mediumDangerGoalsAgainst",
            "penaltiesFor",
            "penaltiesAgainst",
            "takeawaysFor",
            "giveawaysFor"
        FROM nhl24_matchups_with_situations
        WHERE team = %(team)s
        ORDER BY games_played DESC
        LIMIT 10
    ),
    goalie_stats AS (
        SELECT 
            team,
            MAX(save_percentage) as best_save_percentage,
            MAX(games_played) as most_games_played
        FROM nhl24_goalie_stats
        WHERE team = %(team)s
        GROUP BY team
    ),
    skater_stats AS (
        SELECT 
            team,
            MAX(goals) as top_scorer_goals,
            COUNT(*) as num_players
        FROM nhl24_skater_stats
        WHERE team = %(team)s AND position != 'G'
        GROUP BY team
    )
    SELECT 
        AVG(rg."goalsFor") as goalsFor,
        AVG(rg."goalsAgainst") as goalsAgainst,
        AVG(rg."xGoalsPercentage") as xGoalsPercentage,
        AVG(rg."corsiPercentage") as corsiPercentage,
        AVG(rg."fenwickPercentage") as fenwickPercentage,
        AVG(rg."shotsOnGoalFor") as recent_shots_for,
        AVG(rg."shotsOnGoalAgainst") as recent_shots_against,
        AVG(rg."highDangerShotsFor") as recent_hd_shots_for,
        AVG(rg."highDangerGoalsFor") as recent_hd_goals_for,
        AVG(rg."xGoalsFor") as recent_xgoals_for,
        AVG(rg."xGoalsAgainst") as recent_xgoals_against,
        COUNT(CASE WHEN rg."goalsFor" > rg."goalsAgainst" THEN 1 END)::float / 
            NULLIF(COUNT(*), 0) as win_percentage,
        gs.best_save_percentage as goalie_save_percentage,
        gs.most_games_played as goalie_games,
        ss.top_scorer_goals,
        AVG(rg."takeawaysFor" - rg."giveawaysFor") as possession_diff,
        AVG(rg."highDangerGoalsFor" + rg."mediumDangerGoalsFor") as danger_goals_for,
        AVG(rg."highDangerGoalsAgainst" + rg."mediumDangerGoalsAgainst") as danger_goals_against,
        AVG(rg."penaltiesFor"::float / NULLIF(rg."penaltiesAgainst", 0)) as special_teams_ratio
    FROM recent_games rg
    LEFT JOIN goalie_stats gs ON rg.team = gs.team
    LEFT JOIN skater_stats ss ON rg.team = ss.team
    GROUP BY gs.best_save_percentage, gs.most_games_played, ss.top_scorer_goals
    """
    engine = init_connection()
    return pd.read_sql(query, engine, params={'team': team}).iloc[0]

def get_head_to_head_stats(home_team, away_team):
    """Get head-to-head statistics with proper column references"""
    query = """
    WITH h2h_games AS (
        SELECT 
            home_team,
            away_team,
            CAST(home_team_score AS INTEGER) as home_score,
            CAST(away_team_score AS INTEGER) as away_score,
            CAST(home_team_score AS INTEGER) + CAST(away_team_score AS INTEGER) as total_goals,
            status
        FROM nhl24_results
        WHERE (home_team = %(home)s AND away_team = %(away)s)
           OR (home_team = %(away)s AND away_team = %(home)s)
        ORDER BY game_date DESC
        LIMIT 10
    )
    SELECT 
        COALESCE(AVG(total_goals), 0) as avg_total_goals,
        COUNT(*) as games_played,
        SUM(CASE 
            WHEN home_team = %(home)s AND home_score > away_score THEN 1
            WHEN away_team = %(home)s AND away_score > home_score THEN 1
            ELSE 0 
        END) as home_team_wins,
        AVG(CASE WHEN home_team = %(home)s THEN home_score ELSE away_score END) as home_team_avg_goals,
        AVG(CASE WHEN home_team = %(home)s THEN away_score ELSE home_score END) as away_team_avg_goals,
        COUNT(CASE WHEN status NOT IN ('Final', 'Regulation') THEN 1 END) as overtime_games
    FROM h2h_games
    """
    engine = init_connection()
    # Pass parameters as a dictionary
    params = {
        'home': home_team,
        'away': away_team
    }
    return pd.read_sql(query, engine, params=params).iloc[0]

def get_recent_form(team, games_limit=5):
    """Get team's recent form with proper column references"""
    query = """
    WITH recent_results AS (
        SELECT 
            CASE 
                WHEN home_team = %(team)s THEN 
                    CASE WHEN CAST(home_team_score AS INTEGER) > CAST(away_team_score AS INTEGER) 
                    THEN 1 ELSE 0 END
                ELSE 
                    CASE WHEN CAST(away_team_score AS INTEGER) > CAST(home_team_score AS INTEGER) 
                    THEN 1 ELSE 0 END
            END as is_win,
            CASE 
                WHEN home_team = %(team)s THEN CAST(home_team_score AS INTEGER)
                ELSE CAST(away_team_score AS INTEGER)
            END as goals_scored,
            CASE 
                WHEN home_team = %(team)s THEN CAST(away_team_score AS INTEGER)
                ELSE CAST(home_team_score AS INTEGER)
            END as goals_conceded,
            status
        FROM nhl24_results
        WHERE home_team = %(team)s OR away_team = %(team)s
        ORDER BY game_date DESC
        LIMIT %(limit)s
    )
    SELECT 
        AVG(is_win::float) as recent_win_rate,
        AVG(goals_scored) as avg_goals_scored,
        AVG(goals_conceded) as avg_goals_conceded,
        COUNT(CASE WHEN status NOT IN ('Final', 'Regulation') THEN 1 END) as overtime_games
    FROM recent_results
    """
    engine = init_connection()
    return pd.read_sql(query, engine, params={'team': team, 'limit': games_limit}).iloc[0]

def safe_get(stats, key, default=0.0):
    try:
        val = stats.get(key, default)
        if val is None or pd.isna(val):
            return float(default)
        return float(val)  # Always convert to float
    except (TypeError, ValueError):
        return float(default)  # Always return float


def prepare_features(home_team, away_team, home_odds, away_odds, draw_odds):
    try:
        # Get all required stats
        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)
        h2h_stats = get_head_to_head_stats(home_team, away_team)
        home_form = get_recent_form(home_team)
        away_form = get_recent_form(away_team)

        # Calculate market probabilities
        home_implied_prob = 1 / float(home_odds) if float(home_odds) != 0 else 0.33
        away_implied_prob = 1 / float(away_odds) if float(away_odds) != 0 else 0.33
        draw_implied_prob = 1 / float(draw_odds) if float(draw_odds) != 0 else 0.33
        market_efficiency = home_implied_prob + away_implied_prob + draw_implied_prob

        # Create relative ratios for features
        features = pd.DataFrame([{
            # H2H features remain unchanged
            'h2h_home_win_pct': safe_get(h2h_stats, 'home_team_wins', 0) / max(h2h_stats['games_played'], 1),
            'h2h_games_played': safe_get(h2h_stats, 'games_played', 0),
            'h2h_avg_total_goals': safe_get(h2h_stats, 'avg_total_goals', 5.0),
            'draw_implied_prob_normalized': draw_implied_prob / market_efficiency,

            # Create relative ratios for all paired metrics
            'relative_recent_wins_ratio': safe_get(home_form, 'recent_win_rate', 0.5) /
                                          max(safe_get(away_form, 'recent_win_rate', 0.5), 0.001),

            'relative_recent_goals_avg_ratio': safe_get(home_stats, 'goalsFor', 2.5) /
                                               max(safe_get(away_stats, 'goalsFor', 2.5), 0.001),

            'relative_recent_goals_allowed_ratio': safe_get(home_stats, 'goalsAgainst', 2.5) /
                                                   max(safe_get(away_stats, 'goalsAgainst', 2.5), 0.001),

            'relative_goalie_save_pct_ratio': safe_get(home_stats, 'goalie_save_percentage', 0.9) /
                                              max(safe_get(away_stats, 'goalie_save_percentage', 0.9), 0.001),

            'relative_goalie_games_ratio': safe_get(home_stats, 'goalie_games', 1) /
                                           max(safe_get(away_stats, 'goalie_games', 1), 0.001),

            'relative_team_goals_per_game_ratio': safe_get(home_stats, 'goalsFor', 2.5) /
                                                  max(safe_get(away_stats, 'goalsFor', 2.5), 0.001),

            'relative_team_top_scorer_goals_ratio': safe_get(home_stats, 'top_scorer_goals', 0) /
                                                    max(safe_get(away_stats, 'top_scorer_goals', 0), 0.001),

            'relative_implied_prob_normalized_ratio': (home_implied_prob / market_efficiency) /
                                                      max((away_implied_prob / market_efficiency), 0.001),

            'relative_xGoalsPercentage_ratio': safe_get(home_stats, 'xGoalsPercentage', 50.0) /
                                               max(safe_get(away_stats, 'xGoalsPercentage', 50.0), 0.001),

            'relative_corsiPercentage_ratio': safe_get(home_stats, 'corsiPercentage', 50.0) /
                                              max(safe_get(away_stats, 'corsiPercentage', 50.0), 0.001),

            'relative_fenwickPercentage_ratio': safe_get(home_stats, 'fenwickPercentage', 50.0) /
                                                max(safe_get(away_stats, 'fenwickPercentage', 50.0), 0.001)
        }])

        return features, home_stats, away_stats, h2h_stats

    except Exception as e:
        st.error(f"Failed to prepare features: {str(e)}")
        raise

# Page config
st.set_page_config(
    page_title="NHL Game Predictor",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with animations
st.markdown("""
    <style>
    /* Base styles */
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }

    /* Animations */
    @keyframes slideIn {
        0% {
            transform: translateX(-100%);
            opacity: 0;
        }
        100% {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    @keyframes rotation {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Animated components */
    .title-animation {
        animation: slideIn 1s ease-out;
    }

    .prediction-box {
        animation: fadeIn 1s ease-out;
        transition: transform 0.3s ease;
    }

    .prediction-box:hover {
        transform: translateY(-5px);
    }

    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }

    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .recommendation-box {
        animation: pulse 2s infinite;
        transition: all 0.3s ease;
    }

    .recommendation-box:hover {
        transform: scale(1.02);
    }

    /* Loading animation */
    .loader {
        width: 48px;
        height: 48px;
        border: 5px solid #FFF;
        border-bottom-color: #FF3D00;
        border-radius: 50%;
        display: inline-block;
        box-sizing: border-box;
        animation: rotation 1s linear infinite;
    }

    /* Stats box */
    .stats-box {
        border: 1px solid #ddd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }

    .stats-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Progress bar */
    .progress-bar {
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }

    .progress-bar-fill {
        height: 100%;
        background-color: #4CAF50;
        animation: fillProgress 1s ease-out;
        transition: width 1s ease-out;
    }

    @keyframes fillProgress {
        from { width: 0; }
        to { width: var(--width); }
    }

    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }

    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: black;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px 0;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }

    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Add this to your existing CSS */
    @keyframes shine {
        0% {
            background-position: -100% 50%;
        }
        100% {
            background-position: 200% 50%;
        }
    }
    
    @keyframes glow {
        0% {
            filter: drop-shadow(0 0 5px rgba(255, 215, 0, 0.7));
        }
        50% {
            filter: drop-shadow(0 0 20px rgba(255, 215, 0, 0.9));
        }
        100% {
            filter: drop-shadow(0 0 5px rgba(255, 215, 0, 0.7));
        }
    }
    
    .team-logo {
        transform: perspective(1000px) rotateY(10deg);
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .team-logo::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 0%,
            transparent 35%,
            rgba(255, 255, 255, 0.4) 45%,
            rgba(255, 255, 255, 0.7) 50%,
            rgba(255, 255, 255, 0.4) 55%,
            transparent 65%,
            transparent 100%
        );
        animation: shine 3s infinite;
        pointer-events: none;
    }
    
    .team-logo:hover {
        transform: perspective(1000px) rotateY(0deg);
    }
    
    .recommended-team {
        transform: scale(1.2) perspective(1000px) rotateY(10deg);
        animation: glow 2s infinite;
        z-index: 1;
    }
    
    .recommended-team:hover {
        transform: scale(1.2) perspective(1000px) rotateY(0deg);
    }
    
    /* Add more depth to the VS text */
    .vs-text {
        font-weight: bold;
        color: #333;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-size: 2em;
        background: linear-gradient(45deg, #1a1a1a, #4a4a4a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(2px 2px 2px rgba(0,0,0,0.3));
    }
    
    .progress-bar {
    height: 20px;
    background-color: #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    margin: 10px 0;
    transition: width 1s ease-out;  /* Smooth width transition */
    }
    </style>
    """, unsafe_allow_html=True)

# Animation helper functions
def animated_loading():
    loading_placeholder = st.empty()
    with loading_placeholder:
        st.markdown("""
            <div style="text-align: center;">
                <span class="loader"></span>
                <p>Analyzing matchup data...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    return loading_placeholder


def add_animated_probability_bars(home_prob, away_prob, draw_prob):
    st.markdown("""
        <style>
        .probability-container { margin: 20px 0; }
        .probability-label { margin-bottom: 5px; font-weight: bold; }
        .progress-bar-container {
            width: 100%;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

    for label, prob, color in [
        ("Home Win", home_prob, "#66b3ff"),
        ("Away Win", away_prob, "#ff9999"),
        ("Draw", draw_prob, "#99ff99")
    ]:
        # Calculate the width based on the probability
        width = prob * 100

        st.markdown(f"""
            <div class="probability-container">
                <div class="probability-label">{label}: {prob:.1%}</div>
                <div class="progress-bar" style="width: {width}%;">  <!-- Adjust width here -->
                    <div class="progress-bar-fill" 
                         style="--width: 100%; background-color: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def add_animated_stats_box(title, stats_dict):
    stats_html = "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in stats_dict.items())
    st.markdown(f"""
        <div class="stats-box">
            <h3>{title}</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                {stats_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)


# Add the new display_team_matchup function here
def display_team_matchup(home_team, away_team, recommended_team=None):
    recommendation_text = f"Recommended Bet: {recommended_team}" if recommended_team else ""
    st.markdown(f"""
        <div style="
            background: linear-gradient(to bottom, #1a1f2c, #161922);
            border-radius: 15px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
        <div style="
            text-align: center; 
            margin-bottom: 20px; 
            color: #4CAF50; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 500; 
            font-size: 1.2em;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        ">
            {recommendation_text}
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 0.2, 1])

    with col1:
        home_class = "team-logo recommended-team" if home_team == recommended_team else "team-logo"
        highlight_style = f"""
            background: {'radial-gradient(circle at center, rgba(255,215,0,0.08) 30%, rgba(26,31,44,0) 70%)' if home_team == recommended_team else 'radial-gradient(circle at center, rgba(255,255,255,0.03) 30%, rgba(26,31,44,0) 70%)'};
            padding: 20px;
            border-radius: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        """

        st.markdown(f"""
            <div style="{highlight_style}">
                <img src="{team_logos.get(home_team, '')}" 
                     class="{home_class}"
                     style="max-width: 150px; height: auto;">
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="display: flex; justify-content: center; align-items: center; min-height: 200px;">
                <h2 class="vs-text" style="margin: 0; padding: 0;">VS</h2>
            </div>
            """, unsafe_allow_html=True)

    with col3:
        away_class = "team-logo recommended-team" if away_team == recommended_team else "team-logo"
        highlight_style = f"""
            background: {'radial-gradient(circle at center, rgba(255,215,0,0.08) 30%, rgba(26,31,44,0) 70%)' if away_team == recommended_team else 'radial-gradient(circle at center, rgba(255,255,255,0.03) 30%, rgba(26,31,44,0) 70%)'};
            padding: 20px;
            border-radius: 15px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 200px;
        """

        st.markdown(f"""
            <div style="{highlight_style}">
                <img src="{team_logos.get(away_team, '')}" 
                     class="{away_class}"
                     style="max-width: 150px; height: auto;">
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def add_confidence_indicators(st, home_prob, away_prob, draw_prob):
    """Add clear visual confidence indicators with animation"""
    st.markdown("""
        <div class="prediction-box">
            <h3 style='margin-bottom: 1rem;'>🎯 Confidence Analysis</h3>
        </div>
        """, unsafe_allow_html=True)

    # Calculate overall confidence level
    max_prob = max(home_prob, away_prob, draw_prob)

    # Create confidence meter
    confidence_html = """<div class="metric-container">"""
    if max_prob >= 0.90:
        confidence_html += """
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">🟢</span>
                <div>
                    <div style="font-weight: bold; font-size: 18px;">VERY HIGH</div>
                    <div style="color: #666;">90%+ Confidence</div>
                </div>
            </div>
        """
    elif max_prob >= 0.80:
        confidence_html += """
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">🟡</span>
                <div>
                    <div style="font-weight: bold; font-size: 18px;">HIGH</div>
                    <div style="color: #666;">80-90% Confidence</div>
                </div>
            </div>
        """
    elif max_prob >= 0.70:
        confidence_html += """
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">🟠</span>
                <div>
                    <div style="font-weight: bold; font-size: 18px;">MEDIUM</div>
                    <div style="color: #666;">70-80% Confidence</div>
                </div>
            </div>
        """
    else:
        confidence_html += """
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">🔴</span>
                <div>
                    <div style="font-weight: bold; font-size: 18px;">LOW</div>
                    <div style="color: #666;"><70% Confidence</div>
                </div>
            </div>
        """
    confidence_html += "</div>"
    st.markdown(confidence_html, unsafe_allow_html=True)


# Keep your existing helper functions but update the betting recommendations
def add_betting_recommendations(st, home_team, away_team, home_prob, away_prob, draw_prob,
                                home_odds, away_odds, draw_odds, stake,
                                home_stats, away_stats, h2h_stats):
    """Add betting recommendations based on multiple factors"""
    st.markdown("""
        <div class="prediction-box">
            <h3 style='margin-bottom: 1rem;'>💰 Betting Analysis & Recommendations</h3>
        </div>
        """, unsafe_allow_html=True)

    # Calculate market-implied probabilities
    home_implied = 1 / home_odds
    away_implied = 1 / away_odds
    draw_implied = 1 / draw_odds
    total_implied = home_implied + away_implied + draw_implied

    # Normalize implied probabilities
    home_implied_norm = home_implied / total_implied
    away_implied_norm = away_implied / total_implied
    draw_implied_norm = draw_implied / total_implied

    # Calculate edge
    home_edge = home_prob - home_implied_norm
    away_edge = away_prob - away_implied_norm
    draw_edge = draw_prob - draw_implied_norm

    # Calculate confidence scores
    def calculate_confidence_score(prob, implied_prob, stats, is_home=True):
        score = 0

        # Edge factor (30%)
        edge = prob - implied_prob
        score += (edge * 100) * 0.3

        # Recent form factor (25%)
        recent_goals = float(safe_get(stats, 'recent_goals_for', 2.5))
        recent_goals_against = float(safe_get(stats, 'recent_goals_against', 2.5))
        form_score = ((recent_goals - recent_goals_against) / max(recent_goals_against, 1)) * 25
        score += min(max(form_score, -25), 25)

        # H2H factor (20%)
        if h2h_stats['games_played'] > 0:
            h2h_wins = h2h_stats['home_team_wins'] if is_home else (
                    h2h_stats['games_played'] - h2h_stats['home_team_wins'])
            h2h_score = (h2h_wins / h2h_stats['games_played']) * 20
            score += h2h_score

        # Odds value factor (15%)
        odds = home_odds if is_home else away_odds
        odds_score = (1 / odds) * 15
        score += odds_score

        # Statistical dominance (10%)
        xg_percentage = float(safe_get(stats, 'recent_xgoals_pct', 50.0))
        score += ((xg_percentage - 50) / 50) * 10

        return score

    # Calculate confidence scores
    home_confidence = calculate_confidence_score(home_prob, home_implied_norm, home_stats, True)
    away_confidence = calculate_confidence_score(away_prob, away_implied_norm, away_stats, False)
    draw_confidence = 0  # Conservative for draws

    # Determine best betting opportunities
    bets = [
        {
            'type': 'Home',
            'team': home_team,
            'odds': home_odds,
            'prob': home_prob,
            'edge': home_edge,
            'confidence': home_confidence
        },
        {
            'type': 'Away',
            'team': away_team,
            'odds': away_odds,
            'prob': away_prob,
            'edge': away_edge,
            'confidence': away_confidence
        },
        {
            'type': 'Draw',
            'team': 'Draw',
            'odds': draw_odds,
            'prob': draw_prob,
            'edge': draw_edge,
            'confidence': draw_confidence
        }
    ]

    # Sort bets by confidence score
    bets.sort(key=lambda x: x['confidence'], reverse=True)
    best_bet = bets[0]

    # Only recommend if confidence score is positive and edge exists
    if best_bet['confidence'] > 0 and best_bet['edge'] > 0:
        st.markdown(
            f"""
            <div class="recommendation-box" style='background-color: #1a472a; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; margin-top: 0;'>🎯 RECOMMENDED BET</h4>
                <p style='color: white; font-size: 18px;'><strong>{best_bet['team']}</strong> ({best_bet['type']})</p>
                <ul style='color: white;'>
                    <li>Confidence Score: {best_bet['confidence']:.1f}</li>
                    <li>Edge: {best_bet['edge']:.1%}</li>
                    <li>Model Probability: {best_bet['prob']:.1%}</li>
                    <li>Market Odds: {best_bet['odds']:.2f}</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Add reasoning
        st.markdown("#### Key Factors Supporting This Bet")
        reasons = []
        if best_bet['edge'] > 0.05:
            reasons.append(f"✅ Strong edge against market odds ({best_bet['edge']:.1%})")
        if best_bet['confidence'] > 15:
            reasons.append("✅ High confidence based on recent form and historical data")
        if best_bet['type'] == 'Home' and float(safe_get(home_stats, 'recent_goals_for', 0)) > float(
                safe_get(away_stats, 'recent_goals_for', 0)):
            reasons.append("✅ Home team showing superior recent scoring form")
        elif best_bet['type'] == 'Away' and float(safe_get(away_stats, 'recent_goals_for', 0)) > float(
                safe_get(home_stats, 'recent_goals_for', 0)):
            reasons.append("✅ Away team showing superior recent scoring form")
        if h2h_stats['games_played'] > 2:
            if best_bet['type'] == 'Home' and h2h_stats['home_team_wins'] / h2h_stats['games_played'] > 0.5:
                reasons.append(
                    f"✅ Strong head-to-head record ({h2h_stats['home_team_wins']} wins in {h2h_stats['games_played']} games)")
            elif best_bet['type'] == 'Away' and (h2h_stats['games_played'] - h2h_stats['home_team_wins']) / h2h_stats[
                'games_played'] > 0.5:
                reasons.append(
                    f"✅ Strong head-to-head record ({h2h_stats['games_played'] - h2h_stats['home_team_wins']} wins in {h2h_stats['games_played']} games)")

        for reason in reasons:
            st.markdown(reason)

    else:
        st.warning("⚠️ No high-confidence bets recommended for this game.")

    # Add detailed value breakdown
    st.markdown("#### Value Comparison")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            f"{home_team} (Home)",
            f"Edge: {home_edge:.1%}",
            f"Confidence: {home_confidence:.1f}",
            delta_color="normal" if home_edge > 0 else "off"
        )

    with col2:
        st.metric(
            "Draw",
            f"Edge: {draw_edge:.1%}",
            f"Confidence: {draw_confidence:.1f}",
            delta_color="normal" if draw_edge > 0 else "off"
        )

    with col3:
        st.metric(
            f"{away_team} (Away)",
            f"Edge: {away_edge:.1%}",
            f"Confidence: {away_confidence:.1f}",
            delta_color="normal" if away_edge > 0 else "off"
        )


def create_visualization(home_team, away_team, home_stats, away_stats, probabilities, h2h_stats):
    """Create enhanced visualization charts"""
    try:
        fig = plt.Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Enhanced Prediction Probability Pie Chart
        ax1 = fig.add_subplot(gs[0, 0])
        labels = [
            f'{away_team}\n{probabilities[0]:.1%}',
            f'{home_team}\n{probabilities[1]:.1%}',
            f'Draw\n{probabilities[2]:.1%}'
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        explode = (0.1, 0.1, 0.1)
        wedges, texts, autotexts = ax1.pie(probabilities, explode=explode,
                                           labels=labels, colors=colors,
                                           autopct='%1.1f%%', shadow=True)
        ax1.set_title('Prediction Probabilities', pad=20)

        # 2. Advanced Metrics Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        metrics = ['xG%', 'Corsi%', 'Fenwick%']
        home_values = [
            safe_get(home_stats, 'xGoalsPercentage', 50),
            safe_get(home_stats, 'corsiPercentage', 50),
            safe_get(home_stats, 'fenwickPercentage', 50)
        ]
        away_values = [
            safe_get(away_stats, 'xGoalsPercentage', 50),
            safe_get(away_stats, 'corsiPercentage', 50),
            safe_get(away_stats, 'fenwickPercentage', 50)
        ]

        x = np.arange(len(metrics))
        width = 0.35
        ax2.bar(x - width / 2, home_values, width, label=home_team, color='#66b3ff', alpha=0.8)
        ax2.bar(x + width / 2, away_values, width, label=away_team, color='#ff9999', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.set_title('Advanced Metrics Comparison')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # 3. Enhanced Offensive Output Radar Chart
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        categories = [
            'Goals/Game', 'xG/Game',
            'High Danger\nChances', 'Shot\nAttempts',
            'Special Teams\nRatio'
        ]

        # Scale factors for radar chart
        goals_scale = 100 / 5  # Assuming 5 goals is max
        xg_scale = 100 / 4  # Assuming 4 xG is max
        hd_scale = 100 / 15  # Assuming 15 high danger chances is max
        shots_scale = 100 / 40  # Assuming 40 shots is max
        st_scale = 100 / 2  # Assuming 2.0 ratio is max

        # Get scaled values
        home_values = [
            safe_get(home_stats, 'goalsFor', 2.5) * goals_scale,
            safe_get(home_stats, 'xGoalsFor', 2) * xg_scale,
            safe_get(home_stats, 'highDangerShotsFor', 8) * hd_scale,
            safe_get(home_stats, 'shotAttemptsFor', 30) * shots_scale,
            safe_get(home_stats, 'penaltiesFor', 1) * st_scale
        ]

        away_values = [
            safe_get(away_stats, 'goalsFor', 2.5) * goals_scale,
            safe_get(away_stats, 'xGoalsFor', 2) * xg_scale,
            safe_get(away_stats, 'highDangerShotsFor', 8) * hd_scale,
            safe_get(away_stats, 'shotAttemptsFor', 30) * shots_scale,
            safe_get(away_stats, 'penaltiesFor', 1) * st_scale
        ]

        # Normalize values
        home_values = np.clip(home_values, 0, 100)
        away_values = np.clip(away_values, 0, 100)

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        home_values = np.concatenate((home_values, [home_values[0]]))
        away_values = np.concatenate((away_values, [away_values[0]]))

        ax3.plot(angles, home_values, 'o-', label=home_team, color='#66b3ff', linewidth=2)
        ax3.fill(angles, home_values, alpha=0.25, color='#66b3ff')
        ax3.plot(angles, away_values, 'o-', label=away_team, color='#ff9999', linewidth=2)
        ax3.fill(angles, away_values, alpha=0.25, color='#ff9999')

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        # 4. Head-to-Head Summary with Enhanced Stats
        ax4 = fig.add_subplot(gs[1, 1])
        if h2h_stats['games_played'] > 0:
            h2h_data = [
                float(h2h_stats['home_team_wins']),
                float(h2h_stats['games_played'] - h2h_stats['home_team_wins']),
                float(h2h_stats['overtime_games'])  # Use the raw overtime_games count
            ]
            h2h_labels = [f'{home_team}\nWins', f'{away_team}\nWins', 'OT/SO\nGames']
            colors = ['#66b3ff', '#ff9999', '#99ff99']

            bars = ax4.bar(h2h_labels, h2h_data, color=colors)
            ax4.set_title('Head-to-Head Results')

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}',
                         ha='center', va='bottom')

            # Add average goals annotation
            ax4.text(0.5, -0.1,
                     f'Avg Total Goals: {h2h_stats["avg_total_goals"]:.1f}',
                     ha='center', transform=ax4.transAxes)

        else:
            ax4.text(0.5, 0.5, 'No H2H Data Available',
                     horizontalalignment='center',
                     verticalalignment='center')
            ax4.set_xticks([])
            ax4.set_yticks([])

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Failed to create visualization: {str(e)}")
        raise


def add_performance_metrics(st, home_stats, away_stats):
    """Add detailed performance metrics comparison"""
    st.markdown("""
        <div class="metric-container">
            <h4 style='margin-top: 0;'>🎯 Performance Metrics Comparison</h4>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        metrics = {
            "Expected Goals (xG)": ('xGoalsFor', 2),
            "Shot Attempts": ('shotAttemptsFor', 30),
            "High Danger Chances": ('highDangerShotsFor', 8),
            "Save Percentage": ('best_save_percentage', 0.9)
        }

        for label, (key, default) in metrics.items():
            home_val = safe_get(home_stats, key, default)
            away_val = safe_get(away_stats, key, default)

            # Determine which team is better for this metric
            if key == 'best_save_percentage':
                home_val = home_val * 100  # Convert to percentage
                away_val = away_val * 100

            better_team = '🏠' if home_val > away_val else '✈️' if away_val > home_val else '='

            st.metric(
                label=f"{label} {better_team}",
                value=f"{home_val:.1f}" if key != 'best_save_percentage' else f"{home_val:.1f}%",
                delta=f"{home_val - away_val:+.1f}",
                delta_color="normal" if home_val >= away_val else "inverse"
            )

    with col2:
        metrics = {
            "Corsi %": ('corsiPercentage', 50),
            "Fenwick %": ('fenwickPercentage', 50),
            "Goals Per Game": ('goalsFor', 2.5),
            "Special Teams Ratio": ('penaltiesFor', 1)
        }

        for label, (key, default) in metrics.items():
            home_val = safe_get(home_stats, key, default)
            away_val = safe_get(away_stats, key, default)
            better_team = '🏠' if home_val > away_val else '✈️' if away_val > home_val else '='

            st.metric(
                label=f"{label} {better_team}",
                value=f"{home_val:.1f}" + ("%" if "%" in label else ""),
                delta=f"{home_val - away_val:+.1f}",
                delta_color="normal" if home_val >= away_val else "inverse"
            )


def add_form_guide(st, home_team, away_team):
    """Add recent form guide with last 5 games"""
    home_form = get_recent_form(home_team)
    away_form = get_recent_form(away_team)

    st.markdown("""
        <div class="metric-container">
            <h4 style='margin-top: 0;'>📈 Recent Form Guide (Last 5 Games)</h4>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{home_team}**")
        metrics = {
            "Win Rate": ('recent_win_rate', 0.5),
            "Goals Scored": ('avg_goals_scored', 2.5),
            "Goals Conceded": ('avg_goals_conceded', 2.5),
            "xG Performance": ('avg_xg_for', 2.0)
        }

        for label, (key, default) in metrics.items():
            value = safe_get(home_form, key, default)
            if label == "Win Rate":
                st.metric(label, f"{value:.1%}")
            else:
                st.metric(label, f"{value:.2f}")

    with col2:
        st.markdown(f"**{away_team}**")
        for label, (key, default) in metrics.items():
            value = safe_get(away_form, key, default)
            if label == "Win Rate":
                st.metric(label, f"{value:.1%}")
            else:
                st.metric(label, f"{value:.2f}")


def add_betting_insights(st, home_prob, away_prob, draw_prob, home_odds, away_odds, draw_odds):
    """Add betting insights and value analysis"""
    st.markdown("""
        <div class="metric-container">
            <h4 style='margin-top: 0;'>💰 Betting Value Analysis</h4>
        </div>
        """, unsafe_allow_html=True)

    # Calculate implied probabilities from odds
    home_implied = 1 / home_odds
    away_implied = 1 / away_odds
    draw_implied = 1 / draw_odds
    total_implied = home_implied + away_implied + draw_implied

    # Normalize implied probabilities
    home_implied_norm = home_implied / total_implied
    away_implied_norm = away_implied / total_implied
    draw_implied_norm = draw_implied / total_implied

    # Calculate edges
    home_edge = home_prob - home_implied_norm
    away_edge = away_prob - away_implied_norm
    draw_edge = draw_prob - draw_implied_norm

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Home Edge",
            f"{home_edge:+.1%}",
            f"Fair Odds: {1 / home_prob:.2f}",
            delta_color="normal" if home_edge > 0 else "inverse"
        )

    with col2:
        st.metric(
            "Draw Edge",
            f"{draw_edge:+.1%}",
            f"Fair Odds: {1 / draw_prob:.2f}",
            delta_color="normal" if draw_edge > 0 else "inverse"
        )

    with col3:
        st.metric(
            "Away Edge",
            f"{away_edge:+.1%}",
            f"Fair Odds: {1 / away_prob:.2f}",
            delta_color="normal" if away_edge > 0 else "inverse"
        )

    # Add betting recommendations based on edge
    threshold = 0.05  # 5% edge threshold
    recommendations = []

    if home_edge > threshold:
        recommendations.append(f"✅ Home win shows value at odds of {home_odds:.2f}")
    if draw_edge > threshold:
        recommendations.append(f"✅ Draw shows value at odds of {draw_odds:.2f}")
    if away_edge > threshold:
        recommendations.append(f"✅ Away win shows value at odds of {away_odds:.2f}")

    if recommendations:
        st.markdown("### Recommended Bets")
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.warning("⚠️ No significant betting value found at current odds")

def main():
    # Animated title
    st.markdown('<h1 class="title-animation">NHL Game Predictor 🏒</h1>', unsafe_allow_html=True)

    try:
        # Load model and get teams
        with st.spinner("Loading model and data..."):
            model = load_model()
            teams = get_teams()

        # Create main columns
        col1, col2 = st.columns(2)

        # Team Selection
        with col1:
            st.markdown("""
                <div class="tooltip">
                    <h3>Home Team</h3>
                    <span class="tooltiptext">Select the home team</span>
                </div>
                """, unsafe_allow_html=True)
            home_team = st.selectbox("", teams, key="home")

            st.markdown("""
                <div class="tooltip">
                    <p>Home Team Odds</p>
                    <span class="tooltiptext">Enter current bookmaker odds</span>
                </div>
                """, unsafe_allow_html=True)
            home_odds = st.number_input("", min_value=1.0, value=2.0, step=0.1, key="home_odds")

        with col2:
            st.markdown("""
                <div class="tooltip">
                    <h3>Away Team</h3>
                    <span class="tooltiptext">Select the away team</span>
                </div>
                """, unsafe_allow_html=True)
            away_team = st.selectbox("", teams, key="away")

            st.markdown("""
                <div class="tooltip">
                    <p>Away Team Odds</p>
                    <span class="tooltiptext">Enter current bookmaker odds</span>
                </div>
                """, unsafe_allow_html=True)
            away_odds = st.number_input("", min_value=1.0, value=2.0, step=0.1, key="away_odds")

        # Additional inputs
        st.markdown('<h3 class="fade-in">Additional Information</h3>', unsafe_allow_html=True)
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("""
                <div class="tooltip">
                    <p>Draw Odds</p>
                    <span class="tooltiptext">Enter current draw odds</span>
                </div>
                """, unsafe_allow_html=True)
            draw_odds = st.number_input("", min_value=1.0, value=3.5, step=0.1, key="draw_odds")

        with col4:
            st.markdown("""
                <div class="tooltip">
                    <p>Stake (£)</p>
                    <span class="tooltiptext">Enter your betting stake</span>
                </div>
                """, unsafe_allow_html=True)
            stake = st.number_input("", min_value=0.0, value=10.0, step=1.0, key="stake")

        # Make prediction button
        if st.button("Make Prediction", type="primary"):
            if home_team == away_team:
                st.error("Please select different teams")
            else:
                # Show loading animation
                loading_placeholder = animated_loading()

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

                # Clear the loading animation
                loading_placeholder.empty()

                # Create tabs for different views
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Prediction Results",
                    "Team Stats",
                    "Head to Head",
                    "Betting Analysis",
                    "Risk Assessment"
                ])

                with tab1:
                    st.markdown('<h3 class="fade-in">Prediction Results</h3>', unsafe_allow_html=True)

                    # Determine recommended team
                    recommended_team = None
                    if home_prob > away_prob and home_prob > draw_prob:
                        recommended_team = home_team
                    elif away_prob > home_prob and away_prob > draw_prob:
                        recommended_team = away_team

                    # Display team matchup with recommended team highlighting
                    display_team_matchup(home_team, away_team, recommended_team)

                    # Add a separator
                    st.markdown("<hr>", unsafe_allow_html=True)

                    add_animated_probability_bars(home_prob, away_prob, draw_prob)

                    # Add visualization
                    fig = create_visualization(
                        home_team, away_team, home_stats, away_stats,
                        [away_prob, home_prob, draw_prob], h2h_stats
                    )
                    st.pyplot(fig)

                    # Show animated stats boxes
                    col5, col6 = st.columns(2)
                    with col5:
                        add_animated_stats_box(f"{home_team} Form", {
                            "Recent Goals": f"{safe_get(home_stats, 'goalsFor'):.2f}",
                            "Goals Against": f"{safe_get(home_stats, 'goalsAgainst'):.2f}",
                            "xG Percentage": f"{safe_get(home_stats, 'xGoalsPercentage'):.1f}%",
                            "Corsi": f"{safe_get(home_stats, 'corsiPercentage'):.1f}%",
                            "Fenwick": f"{safe_get(home_stats, 'fenwickPercentage'):.1f}%"
                        })

                    with col6:
                        add_animated_stats_box(f"{away_team} Form", {
                            "Recent Goals": f"{safe_get(away_stats, 'goalsFor'):.2f}",
                            "Goals Against": f"{safe_get(away_stats, 'goalsAgainst'):.2f}",
                            "xG Percentage": f"{safe_get(away_stats, 'xGoalsPercentage'):.1f}%",
                            "Corsi": f"{safe_get(away_stats, 'corsiPercentage'):.1f}%",
                            "Fenwick": f"{safe_get(away_stats, 'fenwickPercentage'):.1f}%"
                        })

                with tab2:
                    add_confidence_indicators(st, home_prob, away_prob, draw_prob)

                    # Show detailed team stats
                    col7, col8 = st.columns(2)
                    with col7:
                        add_animated_stats_box(f"{home_team} Detailed Stats", {
                            "Goals/Game": f"{safe_get(home_stats, 'goalsFor'):.2f}",
                            "Shots/Game": f"{safe_get(home_stats, 'shotsOnGoalFor'):.1f}",
                            "Save %": f"{safe_get(home_stats, 'goalie_save_percentage', 0.9) * 100:.1f}%",
                            "High Danger Shots": f"{safe_get(home_stats, 'highDangerShotsFor'):.1f}"
                        })

                    with col8:
                        add_animated_stats_box(f"{away_team} Detailed Stats", {
                            "Goals/Game": f"{safe_get(away_stats, 'goalsFor'):.2f}",
                            "Shots/Game": f"{safe_get(away_stats, 'shotsOnGoalFor'):.1f}",
                            "Save %": f"{safe_get(away_stats, 'goalie_save_percentage', 0.9) * 100:.1f}%",
                            "High Danger Shots": f"{safe_get(away_stats, 'highDangerShotsFor'):.1f}"
                        })

                with tab3:
                    st.markdown('<h3 class="fade-in">Head to Head Analysis</h3>', unsafe_allow_html=True)
                    if h2h_stats['games_played'] > 0:
                        add_animated_stats_box("Head to Head Stats", {
                            "Total Games": int(h2h_stats['games_played']),
                            f"{home_team} Wins": int(h2h_stats['home_team_wins']),
                            f"{away_team} Wins": int(h2h_stats['games_played'] - h2h_stats['home_team_wins']),
                            "Avg Total Goals": f"{safe_get(h2h_stats, 'avg_total_goals'):.1f}"
                        })
                    else:
                        st.info("No recent head-to-head matches found")

                with tab4:
                    add_betting_recommendations(
                        st, home_team, away_team,
                        home_prob, away_prob, draw_prob,
                        home_odds, away_odds, draw_odds,
                        stake, home_stats, away_stats, h2h_stats
                    )

                with tab5:
                    st.markdown('<h3 class="fade-in">Risk Assessment</h3>', unsafe_allow_html=True)

                    # Calculate risk factors
                    risk_factors = []
                    if abs(home_prob - away_prob) < 0.1:
                        risk_factors.append("Very close match prediction")
                    if h2h_stats['games_played'] < 3:
                        risk_factors.append("Limited head-to-head history")
                    if abs(safe_get(home_stats, 'goalsFor') - safe_get(away_stats, 'goalsFor')) < 0.3:
                        risk_factors.append("Teams showing similar recent form")

                    if risk_factors:
                        for factor in risk_factors:
                            st.warning(f"⚠️ {factor}")
                    else:
                        st.success("✅ No significant risk factors identified")

        # Add footer with additional information
        st.markdown("---")
        st.markdown(
            """
            <div class="fade-in">
            ℹ️ <b>How to use this predictor:</b>
            <ol>
                <li>Select home and away teams</li>
                <li>Enter the current odds from your bookmaker</li>
                <li>Set your stake amount</li>
                <li>Click "Make Prediction" for detailed analysis</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if st.button("Refresh App"):
            st.experimental_rerun()


if __name__ == "__main__":
    main()