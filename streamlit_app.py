import streamlit as st


# Page configuration
def setup_page():
    st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒", layout="wide")


# Team name mappings
TEAM_MAPPINGS = {
    # Eastern Conference
    'Boston': 'Boston Bruins',
    'Buffalo': 'Buffalo Sabres',
    'Carolina': 'Carolina Hurricanes',
    'Columbus': 'Columbus Blue Jackets',
    'Detroit': 'Detroit Red Wings',
    'Florida': 'Florida Panthers',
    'Montreal': 'Montreal Canadiens',
    'New Jersey': 'New Jersey Devils',
    'NY Islanders': 'New York Islanders',
    'NY Rangers': 'New York Rangers',
    'Ottawa': 'Ottawa Senators',
    'Philadelphia': 'Philadelphia Flyers',
    'Pittsburgh': 'Pittsburgh Penguins',
    'Tampa Bay': 'Tampa Bay Lightning',
    'Toronto': 'Toronto Maple Leafs',
    'Washington': 'Washington Capitals',

    # Western Conference
    'Anaheim': 'Anaheim Ducks',
    'Arizona': 'Arizona Coyotes',
    'Calgary': 'Calgary Flames',
    'Chicago': 'Chicago Blackhawks',
    'Colorado': 'Colorado Avalanche',
    'Dallas': 'Dallas Stars',
    'Edmonton': 'Edmonton Oilers',
    'Los Angeles': 'Los Angeles Kings',
    'Minnesota': 'Minnesota Wild',
    'Nashville': 'Nashville Predators',
    'San Jose': 'San Jose Sharks',
    'Seattle': 'Seattle Kraken',
    'St. Louis': 'St. Louis Blues',
    'Utah': 'Utah Hockey Club',
    'Vancouver': 'Vancouver Canucks',
    'Vegas': 'Vegas Golden Knights',
    'Winnipeg': 'Winnipeg Jets',

    # Alternative names
    'NY': 'New York Rangers',
    'TB': 'Tampa Bay Lightning',
    'NJ': 'New Jersey Devils',
    'LA': 'Los Angeles Kings',
    'SJ': 'San Jose Sharks',
    'VGK': 'Vegas Golden Knights',
    'NYR': 'New York Rangers',
    'NYI': 'New York Islanders',
    'TBL': 'Tampa Bay Lightning',
    'STL': 'St. Louis Blues',
    'CBJ': 'Columbus Blue Jackets',
    'Utah HC': 'Utah Hockey Club'
}

import requests
from bs4 import BeautifulSoup
import streamlit as st
import re
from config import TEAM_MAPPINGS


def clean_team_name(team_name):
    """Clean and standardize team names to match database format"""
    cleaned_name = team_name.strip()
    return TEAM_MAPPINGS.get(cleaned_name, cleaned_name)


def extract_decimal_odds(odds_text):
    """Extract and convert odds to decimal format"""
    try:
        odds_text = re.sub(r'[^0-9.-]', '', odds_text)
        return float(odds_text) if odds_text else 3.5
    except:
        return 2.0


@st.cache_data(ttl=300)  # Cache for 5 minutes
def scrape_nhl_odds():
    """Scrape NHL odds from checkbestodds.com"""
    try:
        url = "https://checkbestodds.com/hockey-odds/nhl"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        games = []

        game_rows = soup.find_all('tr', class_=lambda x: x and 'odds-row' in x)

        for row in game_rows:
            try:
                teams = row.find_all('span', class_='team-name')
                if len(teams) < 2:
                    continue

                away_team = clean_team_name(teams[0].text.strip())
                home_team = clean_team_name(teams[1].text.strip())

                odds_cells = row.find_all('td', class_='odds-cell')
                if len(odds_cells) < 3:
                    continue

                away_odds = extract_decimal_odds(odds_cells[0].text)
                draw_odds = extract_decimal_odds(odds_cells[1].text)
                home_odds = extract_decimal_odds(odds_cells[2].text)

                time_cell = row.find('td', class_='time-cell')
                game_time = time_cell.text.strip() if time_cell else "TBD"

                if away_team and home_team:
                    games.append({
                        'away_team': away_team,
                        'home_team': home_team,
                        'away_odds': away_odds,
                        'draw_odds': draw_odds,
                        'home_odds': home_odds,
                        'game_time': game_time
                    })

            except Exception as e:
                st.warning(f"Error processing game row: {str(e)}")
                continue

        return games

    except Exception as e:
        st.error(f"Error scraping odds: {str(e)}")
        return []

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import streamlit as st
import io
import joblib

@st.cache_resource
def load_model_from_drive():
    """Load the prediction model from Google Drive"""
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


import streamlit as st
import pandas as pd
from sqlalchemy import create_engine


@st.cache_resource
def init_connection():
    return create_engine(st.secrets["db_connection"])


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
    """Get team statistics including all required metrics"""
    query = """
    WITH recent_games AS (
        SELECT 
            team,
            COALESCE("goalsFor", 0) as "goalsFor",
            COALESCE("goalsAgainst", 0) as "goalsAgainst",
            COALESCE("xGoalsPercentage", 50) as "xGoalsPercentage",
            COALESCE("corsiPercentage", 50) as "corsiPercentage",
            COALESCE("fenwickPercentage", 50) as "fenwickPercentage"
        FROM nhl24_matchups_with_situations
        WHERE team = %(team)s
        ORDER BY games_played DESC
        LIMIT 10
    ),
    goalie_stats AS (
        SELECT 
            team,
            COALESCE(MAX(save_percentage), 0.9) as goalie_save_percentage,
            COALESCE(MAX(games_played), 1) as goalie_games
        FROM nhl24_goalie_stats
        WHERE team = %(team)s
        GROUP BY team
    ),
    skater_stats AS (
        SELECT 
            team,
            COALESCE(MAX(goals), 0) as top_scorer_goals
        FROM nhl24_skater_stats
        WHERE team = %(team)s AND position != 'G'
        GROUP BY team
    )
    SELECT 
        COALESCE(AVG(rg."goalsFor"), 2.5) as "goalsFor",
        COALESCE(AVG(rg."goalsAgainst"), 2.5) as "goalsAgainst",
        COALESCE(AVG(rg."xGoalsPercentage"), 50) as "xGoalsPercentage",
        COALESCE(AVG(rg."corsiPercentage"), 50) as "corsiPercentage",
        COALESCE(AVG(rg."fenwickPercentage"), 50) as "fenwickPercentage",
        COALESCE(gs.goalie_save_percentage, 0.9) as goalie_save_percentage,
        COALESCE(gs.goalie_games, 1) as goalie_games,
        COALESCE(ss.top_scorer_goals, 0) as top_scorer_goals,
        COALESCE(COUNT(CASE WHEN rg."goalsFor" > rg."goalsAgainst" THEN 1 END)::float / 
            NULLIF(COUNT(*), 0), 0.5) as recent_win_rate
    FROM recent_games rg
    LEFT JOIN goalie_stats gs ON rg.team = gs.team
    LEFT JOIN skater_stats ss ON rg.team = ss.team
    GROUP BY gs.goalie_save_percentage, gs.goalie_games, ss.top_scorer_goals
    """
    engine = init_connection()
    result = pd.read_sql(query, engine, params={'team': team})

    default_values = {
        'goalsFor': 2.5,
        'goalsAgainst': 2.5,
        'xGoalsPercentage': 50.0,
        'corsiPercentage': 50.0,
        'fenwickPercentage': 50.0,
        'goalie_save_percentage': 0.9,
        'goalie_games': 1,
        'top_scorer_goals': 0,
        'recent_win_rate': 0.5
    }

    if result.empty:
        return pd.Series(default_values)

    for col in result.columns:
        if col in default_values:
            result[col] = result[col].fillna(default_values[col])

    return result.iloc[0]


def get_head_to_head_stats(home_team, away_team):
    """Get head-to-head statistics"""
    query = """
    WITH game_results AS (
        SELECT 
            game_date,
            home_team,
            away_team,
            CAST(home_team_score AS INTEGER) as home_score,
            CAST(away_team_score AS INTEGER) as away_score
        FROM nhl24_results
        WHERE (home_team = %(home)s AND away_team = %(away)s)
           OR (home_team = %(away)s AND away_team = %(home)s)
        ORDER BY game_date DESC
        LIMIT 5
    )
    SELECT 
        COUNT(*) as games_played,
        SUM(CASE 
            WHEN home_team = %(home)s AND home_score > away_score THEN 1
            WHEN away_team = %(home)s AND away_score > home_score THEN 1
            ELSE 0 
        END) as home_team_wins,
        AVG(home_score + away_score) as avg_total_goals
    FROM game_results
    """
    engine = init_connection()
    result = pd.read_sql(query, engine, params={'home': home_team, 'away': away_team})
    return result.iloc[0] if not result.empty else pd.Series({
        'games_played': 0,
        'home_team_wins': 0,
        'avg_total_goals': 5.0
    })


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import setup_page
from scraper import scrape_nhl_odds
from model_utils import load_model_from_drive
from database import get_teams, get_team_stats, get_head_to_head_stats
from feature_engineering import prepare_basic_features, display_prediction_results


def main():
    setup_page()
    st.title("NHL Game Predictor 🏒")

    # Initialize session state for game index if not exists
    if 'game_index' not in st.session_state:
        st.session_state.game_index = 0
    if 'games' not in st.session_state:
        st.session_state.games = []
    if 'predictions_cache' not in st.session_state:
        st.session_state.predictions_cache = {}

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model_from_drive()

    if model is None:
        st.error("Failed to load model. Please check the connection.")
        return

    # Add refresh button for odds
    if st.button("🔄 Refresh Odds"):
        st.cache_data.clear()
        st.session_state.predictions_cache = {}
        st.session_state.games = []

    # Scrape current odds
    with st.spinner("Fetching latest odds..."):
        games = scrape_nhl_odds()
        if games:
            st.session_state.games = games

    if not st.session_state.games:
        st.error("Failed to fetch odds. Showing manual input mode.")
        # Fall back to manual input mode
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", get_teams())
            home_odds = st.number_input("Home Odds", min_value=1.01, value=2.0, step=0.05)
        with col2:
            away_team = st.selectbox("Away Team", get_teams())
            away_odds = st.number_input("Away Odds", min_value=1.01, value=2.0, step=0.05)
        draw_odds = st.number_input("Draw Odds", min_value=1.01, value=3.5, step=0.05)

        if st.button("Get Prediction", type="primary"):
            make_prediction(model, home_team, away_team, home_odds, away_odds, draw_odds)
    else:
        # Navigation controls
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous Game") and st.session_state.game_index > 0:
                st.session_state.game_index -= 1
        with col2:
            st.write(f"Game {st.session_state.game_index + 1} of {len(st.session_state.games)}")
        with col3:
            if st.button("Next Game ➡️") and st.session_state.game_index < len(st.session_state.games) - 1:
                st.session_state.game_index += 1

        # Display current game
        current_game = st.session_state.games[st.session_state.game_index]

        # Create columns for team names and odds
        col1, col2, col3 = st.columns([2, 1, 2])

        with col1:
            st.subheader(f"🏃 {current_game['away_team']}")
            st.metric("Away Odds", f"{current_game['away_odds']:.2f}")

        with col2:
            st.subheader("vs")
            st.metric("Draw Odds", f"{current_game['draw_odds']:.2f}")

        with col3:
            st.subheader(f"🏠 {current_game['home_team']}")
            st.metric("Home Odds", f"{current_game['home_odds']:.2f}")

        st.write(f"Game Time: {current_game['game_time']}")

        # Check if we have a cached prediction
        game_key = f"{current_game['home_team']}_{current_game['away_team']}"

        if game_key not in st.session_state.predictions_cache:
            # Make new prediction
            with st.spinner("Analyzing matchup..."):
                features = prepare_basic_features(
                    current_game['home_team'],
                    current_game['away_team'],
                    current_game['home_odds'],
                    current_game['away_odds'],
                    current_game['draw_odds']
                )

                if features is not None:
                    probabilities = model.predict_proba(features)[0]
                    st.session_state.predictions_cache[game_key] = {
                        'probabilities': probabilities,
                        'features': features
                    }

        # Display prediction if we have it
        if game_key in st.session_state.predictions_cache:
            prediction_data = st.session_state.predictions_cache[game_key]

            # Show debug info in expander
            with st.expander("Debug Information"):
                st.write("Model Input Features:")
                st.dataframe(prediction_data['features'])
                st.write("Raw Model Probabilities:")
                st.write({
                    "Away Win": f"{prediction_data['probabilities'][0]:.4f}",
                    "Draw": f"{prediction_data['probabilities'][1]:.4f}",
                    "Home Win": f"{prediction_data['probabilities'][2]:.4f}"
                })

            # Display results
            display_prediction_results(
                prediction_data['probabilities'],
                current_game['home_team'],
                current_game['away_team'],
                current_game['home_odds'],
                current_game['away_odds'],
                current_game['draw_odds']
            )

            # Show team stats
            st.header("Team Statistics")
            stats_col1, stats_col2 = st.columns(2)

            try:
                home_stats = get_team_stats(current_game['home_team'])
                away_stats = get_team_stats(current_game['away_team'])
                h2h_stats = get_head_to_head_stats(current_game['home_team'], current_game['away_team'])

                with stats_col1:
                    st.subheader(f"{current_game['home_team']} Stats")
                    st.metric("Goals For", f"{home_stats['goalsFor']:.2f}")
                    st.metric("xG%", f"{home_stats['xGoalsPercentage']:.1f}%")
                    st.metric("Corsi%", f"{home_stats['corsiPercentage']:.1f}%")
                    st.metric("Recent Win Rate", f"{home_stats['recent_win_rate'] * 100:.1f}%")

                with stats_col2:
                    st.subheader(f"{current_game['away_team']} Stats")
                    st.metric("Goals For", f"{away_stats['goalsFor']:.2f}")
                    st.metric("xG%", f"{away_stats['xGoalsPercentage']:.1f}%")
                    st.metric("Corsi%", f"{away_stats['corsiPercentage']:.1f}%")
                    st.metric("Recent Win Rate", f"{away_stats['recent_win_rate'] * 100:.1f}%")

                # Head to Head Stats
                if h2h_stats['games_played'] > 0:
                    st.header("Head to Head History")
                    h2h_col1, h2h_col2 = st.columns(2)
                    with h2h_col1:
                        st.write(f"Previous Meetings: {int(h2h_stats['games_played'])}")
                        st.write(f"Average Total Goals: {h2h_stats['avg_total_goals']:.1f}")
                    with h2h_col2:
                        home_wins = int(h2h_stats['home_team_wins'])
                        away_wins = int(h2h_stats['games_played'] - h2h_stats['home_team_wins'])
                        st.write(f"{current_game['home_team']} Wins: {home_wins}")
                        st.write(f"{current_game['away_team']} Wins: {away_wins}")
                else:
                    st.info("No recent head-to-head matches found")

            except Exception as e:
                st.error(f"Error displaying team stats: {str(e)}")

    # Add information about how to use
    with st.expander("How to use"):
        st.write("""
        1. The app automatically fetches today's NHL games and odds
        2. Use the Previous/Next buttons to navigate through games
        3. Each game shows:
           - Team matchup and current odds
           - Win/Draw probabilities
           - Team statistics comparison
           - Head-to-head history (if available)

        Note: Click 'Refresh Odds' to get the latest odds data.
        """)

    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>NHL Game Predictor v4.2 | Using updated model trained on 2023-24 season data</p>
            <p>Odds data provided by checkbestodds.com</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()