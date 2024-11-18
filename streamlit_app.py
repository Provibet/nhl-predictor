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


# Model loading from Google Drive
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


# Helper functions
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

    # Return default values if no data found
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

    # Replace any None or NaN values with defaults
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
        'avg_total_goals': 5.0  # Default value
    })


def safe_get(stats, key, default=0.0):
    """Safely get a value from stats with a default"""
    try:
        val = stats.get(key, default)
        if val is None or pd.isna(val):
            return float(default)
        # Convert to float and handle division by zero
        val = float(val)
        return val if val != 0 else float(default)
    except (TypeError, ValueError):
        return float(default)


def prepare_basic_features(home_team, away_team, home_odds, away_odds, draw_odds):
    """Prepare features that match the model's expectations exactly"""
    try:
        # Get team stats
        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)
        h2h_stats = get_head_to_head_stats(home_team, away_team)

        # Debug info before feature calculation
        with st.expander("Raw Data Check"):
            st.write("Home Stats:", dict(home_stats))
            st.write("Away Stats:", dict(away_stats))
            st.write("H2H Stats:", dict(h2h_stats))

        # Ensure small non-zero value for division
        EPSILON = 0.0001

        # Calculate features with safety checks
        features = {
            'h2h_home_win_pct': float(h2h_stats['home_team_wins']) / max(float(h2h_stats['games_played']), 1),
            'h2h_games_played': float(h2h_stats['games_played']),
            'h2h_avg_total_goals': float(h2h_stats['avg_total_goals']),

            'draw_implied_prob_normalized': ((1 / float(draw_odds)) /
                                             ((1 / float(home_odds)) + (1 / float(away_odds)) + (
                                                         1 / float(draw_odds)))),

            'relative_recent_wins_ratio': (float(home_stats['recent_win_rate']) /
                                           max(float(away_stats['recent_win_rate']), EPSILON)),

            'relative_recent_goals_avg_ratio': (float(home_stats['goalsFor']) /
                                                max(float(away_stats['goalsFor']), EPSILON)),

            'relative_recent_goals_allowed_ratio': (float(home_stats['goalsAgainst']) /
                                                    max(float(away_stats['goalsAgainst']), EPSILON)),

            'relative_goalie_save_pct_ratio': (float(home_stats['goalie_save_percentage']) /
                                               max(float(away_stats['goalie_save_percentage']), EPSILON)),

            'relative_goalie_games_ratio': (float(home_stats['goalie_games']) /
                                            max(float(away_stats['goalie_games']), EPSILON)),

            'relative_team_goals_per_game_ratio': (float(home_stats['goalsFor']) /
                                                   max(float(away_stats['goalsFor']), EPSILON)),

            'relative_team_top_scorer_goals_ratio': ((float(home_stats['top_scorer_goals']) + EPSILON) /
                                                     max(float(away_stats['top_scorer_goals']) + EPSILON, EPSILON)),

            'relative_implied_prob_normalized_ratio': (((1 / float(home_odds)) /
                                                        ((1 / float(home_odds)) + (1 / float(away_odds)) + (
                                                                    1 / float(draw_odds)))) /
                                                       max((1 / float(away_odds)) /
                                                           ((1 / float(home_odds)) + (1 / float(away_odds)) + (
                                                                       1 / float(draw_odds))),
                                                           EPSILON)),

            'relative_xGoalsPercentage_ratio': (float(home_stats['xGoalsPercentage']) /
                                                max(float(away_stats['xGoalsPercentage']), EPSILON)),

            'relative_corsiPercentage_ratio': (float(home_stats['corsiPercentage']) /
                                               max(float(away_stats['corsiPercentage']), EPSILON)),

            'relative_fenwickPercentage_ratio': (float(home_stats['fenwickPercentage']) /
                                                 max(float(away_stats['fenwickPercentage']), EPSILON))
        }

        # Debug the calculated features
        with st.expander("Debug Feature Information"):
            st.write("Calculated Features:", features)
            # Check for any None or infinite values
            for key, value in features.items():
                if value is None or np.isinf(value) or np.isnan(value):
                    st.error(f"Invalid value in {key}: {value}")

        return pd.DataFrame([features])

    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        st.write("Debug information:")
        st.write(f"Home team: {home_team}")
        st.write(f"Away team: {away_team}")
        return None

def display_prediction_results(probs, home_team, away_team, home_odds, away_odds, draw_odds):
    """Display prediction results with formatting"""
    st.header("Prediction Results")

    col1, col2, col3 = st.columns(3)

    # Calculate implied probabilities
    implied_home = 1 / home_odds
    implied_away = 1 / away_odds
    implied_draw = 1 / draw_odds
    total_implied = implied_home + implied_away + implied_draw

    # Normalize implied probabilities
    implied_home_norm = implied_home / total_implied
    implied_away_norm = implied_away / total_implied
    implied_draw_norm = implied_draw / total_implied

    # Calculate edges
    home_edge = probs[2] - implied_home_norm
    away_edge = probs[0] - implied_away_norm
    draw_edge = probs[1] - implied_draw_norm

    with col1:
        st.metric(
            f"{home_team} (Home)",
            f"{probs[2]:.1%}",
            f"Edge: {home_edge:+.1%}",
            delta_color="normal" if home_edge > 0 else "inverse"
        )

    with col2:
        st.metric(
            "Draw",
            f"{probs[1]:.1%}",
            f"Edge: {draw_edge:+.1%}",
            delta_color="normal" if draw_edge > 0 else "inverse"
        )

    with col3:
        st.metric(
            f"{away_team} (Away)",
            f"{probs[0]:.1%}",
            f"Edge: {away_edge:+.1%}",
            delta_color="normal" if away_edge > 0 else "inverse"
        )

    # Display best value bet
    st.subheader("Betting Value Analysis")
    edges = [away_edge, draw_edge, home_edge]
    outcomes = ["Away Win", "Draw", "Home Win"]
    odds = [away_odds, draw_odds, home_odds]

    best_edge_idx = np.argmax(edges)
    if edges[best_edge_idx] > 0.05:  # 5% edge threshold
        st.success(f"Best Value Bet: {outcomes[best_edge_idx]} @ {odds[best_edge_idx]:.2f} "
                   f"(Edge: {edges[best_edge_idx]:+.1%})")
    else:
        st.warning("No significant betting value found")

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f'{away_team}\n{probs[0]:.1%}', f'Draw\n{probs[1]:.1%}', f'{home_team}\n{probs[2]:.1%}']
    model_probs = [probs[0], probs[1], probs[2]]
    market_probs = [implied_away_norm, implied_draw_norm, implied_home_norm]

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, model_probs, width, label='Model Probability', color='lightblue')
    ax.bar(x + width / 2, market_probs, width, label='Market Implied', color='lightgray')

    ax.set_ylabel('Probability')
    ax.set_title('Model vs Market Probabilities')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title("NHL Game Predictor 🏒")

    # Load model
    with st.spinner("Loading prediction model..."):
        model = load_model_from_drive()

    if model is None:
        st.error("Failed to load model. Please check the connection.")
        return

    # Create interface
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

        try:
            with st.spinner("Analyzing matchup..."):
                # Prepare features
                features = prepare_basic_features(
                    home_team, away_team,
                    home_odds, away_odds, draw_odds
                )

                if features is not None:
                    # Make prediction
                    probabilities = model.predict_proba(features)[0]

                    # Show debug info in expander
                    with st.expander("Debug Information"):
                        st.write("Model Input Features:")
                        st.dataframe(features)
                        st.write("Raw Model Probabilities:")
                        st.write({
                            "Away Win": f"{probabilities[0]:.4f}",
                            "Draw": f"{probabilities[1]:.4f}",
                            "Home Win": f"{probabilities[2]:.4f}"
                        })

                    # Display results
                    display_prediction_results(
                        probabilities,
                        home_team,
                        away_team,
                        home_odds,
                        away_odds,
                        draw_odds
                    )

                    # Show additional stats
                    st.header("Team Statistics")
                    col1, col2 = st.columns(2)

                    try:
                        home_stats = get_team_stats(home_team)
                        away_stats = get_team_stats(away_team)
                        h2h_stats = get_head_to_head_stats(home_team, away_team)

                        with col1:
                            st.subheader(f"{home_team} Stats")
                            st.metric("Goals For", f"{safe_get(home_stats, 'goalsFor'):.2f}")
                            st.metric("xG%", f"{safe_get(home_stats, 'xGoalsPercentage'):.1f}%")
                            st.metric("Corsi%", f"{safe_get(home_stats, 'corsiPercentage'):.1f}%")
                            st.metric("Recent Win Rate", f"{safe_get(home_stats, 'recent_win_rate')*100:.1f}%")

                        with col2:
                            st.subheader(f"{away_team} Stats")
                            st.metric("Goals For", f"{safe_get(away_stats, 'goalsFor'):.2f}")
                            st.metric("xG%", f"{safe_get(away_stats, 'xGoalsPercentage'):.1f}%")
                            st.metric("Corsi%", f"{safe_get(away_stats, 'corsiPercentage'):.1f}%")
                            st.metric("Recent Win Rate", f"{safe_get(away_stats, 'recent_win_rate')*100:.1f}%")

                        # Head to Head Stats
                        if h2h_stats['games_played'] > 0:
                            st.header("Head to Head History")
                            st.write(f"Previous Meetings: {int(h2h_stats['games_played'])}")
                            home_wins = int(h2h_stats['home_team_wins'])
                            away_wins = int(h2h_stats['games_played'] - h2h_stats['home_team_wins'])
                            st.write(f"{home_team} Wins: {home_wins}")
                            st.write(f"{away_team} Wins: {away_wins}")
                            st.write(f"Average Total Goals: {h2h_stats['avg_total_goals']:.1f}")
                        else:
                            st.info("No recent head-to-head matches found")

                    except Exception as e:
                        st.error(f"Error displaying team stats: {str(e)}")

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Debug information:")
            st.write(f"Home team: {home_team}")
            st.write(f"Away team: {away_team}")
            st.write(f"Odds: {home_odds}/{away_odds}/{draw_odds}")

    # Add information about how to use
    with st.expander("How to use"):
        st.write("""
        1. Select the home and away teams
        2. Enter the current betting odds (decimal format)
        3. Click 'Get Prediction' for analysis
        4. The model will show:
           - Win/Draw probabilities
           - Betting value analysis
           - Team statistics comparison
           - Head-to-head history (if available)
        """)

    # Add footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>NHL Game Predictor v4.1 | Using updated model trained on 2023-24 season data</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()