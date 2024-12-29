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
    """Safely get a value from stats with robust type checking and conversion"""
    try:
        # Get the value, use default if not found or None
        val = stats.get(key)
        if val is None or pd.isna(val):
            return float(default)

        # Try to convert to float, use default if fails
        try:
            val = float(val)
            # Check for infinite or NaN values
            if np.isinf(val) or np.isnan(val):
                return float(default)
            return val
        except (ValueError, TypeError):
            return float(default)

    except Exception:
        return float(default)


def prepare_basic_features(home_team, away_team, home_odds, away_odds, draw_odds):
    """Prepare features with robust error handling"""
    try:
        # Get team stats with debug output
        st.write("Fetching team stats...")
        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)
        h2h_stats = get_head_to_head_stats(home_team, away_team)

        # Debug raw data
        with st.expander("Raw Data"):
            st.write("Home Stats:", dict(home_stats))
            st.write("Away Stats:", dict(away_stats))
            st.write("H2H Stats:", dict(h2h_stats))

        # Constants for safety
        EPSILON = 0.0001
        DEFAULT_PERCENTAGE = 50.0

        # Safely prepare features with explicit defaults
        features = {
            # H2H features
            'h2h_home_win_pct': safe_get(h2h_stats, 'home_team_wins', 0) /
                                max(safe_get(h2h_stats, 'games_played', 1), 1),
            'h2h_games_played': safe_get(h2h_stats, 'games_played', 0),
            'h2h_avg_total_goals': safe_get(h2h_stats, 'avg_total_goals', 5.0),

            # Implied probabilities
            'draw_implied_prob_normalized': ((1 / float(draw_odds)) /
                                             ((1 / float(home_odds)) +
                                              (1 / float(away_odds)) +
                                              (1 / float(draw_odds)))),

            # Recent form ratios
            'relative_recent_wins_ratio': safe_get(home_stats, 'recent_win_rate', 0.5) /
                                          max(safe_get(away_stats, 'recent_win_rate', 0.5), EPSILON),

            'relative_recent_goals_avg_ratio': safe_get(home_stats, 'goalsFor', 2.5) /
                                               max(safe_get(away_stats, 'goalsFor', 2.5), EPSILON),

            'relative_recent_goals_allowed_ratio': safe_get(home_stats, 'goalsAgainst', 2.5) /
                                                   max(safe_get(away_stats, 'goalsAgainst', 2.5), EPSILON),

            # Goalie stats ratios
            'relative_goalie_save_pct_ratio': safe_get(home_stats, 'goalie_save_percentage', 0.9) /
                                              max(safe_get(away_stats, 'goalie_save_percentage', 0.9), EPSILON),

            'relative_goalie_games_ratio': safe_get(home_stats, 'goalie_games', 1) /
                                           max(safe_get(away_stats, 'goalie_games', 1), EPSILON),

            # Team scoring ratios
            'relative_team_goals_per_game_ratio': safe_get(home_stats, 'goalsFor', 2.5) /
                                                  max(safe_get(away_stats, 'goalsFor', 2.5), EPSILON),

            'relative_team_top_scorer_goals_ratio': (safe_get(home_stats, 'top_scorer_goals', 1) + EPSILON) /
                                                    max(safe_get(away_stats, 'top_scorer_goals', 1) + EPSILON, EPSILON),

            # Market probabilities ratio
            'relative_implied_prob_normalized_ratio': ((1 / float(home_odds)) /
                                                       ((1 / float(home_odds)) + (1 / float(away_odds)) + (
                                                                   1 / float(draw_odds)))) /
                                                      max((1 / float(away_odds)) /
                                                          ((1 / float(home_odds)) + (1 / float(away_odds)) + (
                                                                      1 / float(draw_odds))),
                                                          EPSILON),

            # Advanced stats ratios
            'relative_xGoalsPercentage_ratio': safe_get(home_stats, 'xGoalsPercentage', DEFAULT_PERCENTAGE) /
                                               max(safe_get(away_stats, 'xGoalsPercentage', DEFAULT_PERCENTAGE),
                                                   EPSILON),

            'relative_corsiPercentage_ratio': safe_get(home_stats, 'corsiPercentage', DEFAULT_PERCENTAGE) /
                                              max(safe_get(away_stats, 'corsiPercentage', DEFAULT_PERCENTAGE), EPSILON),

            'relative_fenwickPercentage_ratio': safe_get(home_stats, 'fenwickPercentage', DEFAULT_PERCENTAGE) /
                                                max(safe_get(away_stats, 'fenwickPercentage', DEFAULT_PERCENTAGE),
                                                    EPSILON)
        }

        # Validate features before returning
        for key, value in features.items():
            if value is None or np.isinf(value) or np.isnan(value):
                st.error(f"Invalid value detected for {key}: {value}")
                features[key] = 1.0  # Safe default

        # Create DataFrame and validate
        features_df = pd.DataFrame([features])

        # Debug final features
        with st.expander("Final Features Check"):
            st.write("Feature names:", features_df.columns.tolist())
            st.write("Feature values:", features_df.iloc[0].to_dict())
            st.write("Any null values:", features_df.isnull().sum().to_dict())
            st.write("Any infinite values:", np.isinf(features_df).sum().to_dict())

        return features_df

    except Exception as e:
        st.error(f"Error in feature preparation: {str(e)}")
        st.write("Debug information:")
        st.write(f"Home team: {home_team}")
        st.write(f"Away team: {away_team}")
        st.write(f"Odds: {home_odds}/{away_odds}/{draw_odds}")
        return None


def display_prediction_results(probs, home_team, away_team, home_odds, away_odds, draw_odds):
    """Display prediction results focusing on model confidence"""
    st.header("Prediction Results")

    # Find highest probability outcome
    outcomes = ["Away Win", "Draw", "Home Win"]
    teams = [away_team, "Draw", home_team]
    best_pred_idx = np.argmax(probs)
    confidence = probs[best_pred_idx]

    # Create columns for display
    col1, col2, col3 = st.columns(3)

    # Display probabilities with confidence indicators
    with col1:
        st.metric(
            f"{away_team} (Away)",
            f"{probs[0]:.1%}",
            f"Confidence: {'High' if probs[0] > 0.5 else 'Medium' if probs[0] > 0.33 else 'Low'}",
            delta_color="normal" if probs[0] == max(probs) else "off"
        )

    with col2:
        st.metric(
            "Draw",
            f"{probs[1]:.1%}",
            f"Confidence: {'High' if probs[1] > 0.5 else 'Medium' if probs[1] > 0.33 else 'Low'}",
            delta_color="normal" if probs[1] == max(probs) else "off"
        )

    with col3:
        st.metric(
            f"{home_team} (Home)",
            f"{probs[2]:.1%}",
            f"Confidence: {'High' if probs[2] > 0.5 else 'Medium' if probs[2] > 0.33 else 'Low'}",
            delta_color="normal" if probs[2] == max(probs) else "off"
        )

    # Display model's prediction with confidence
    st.subheader("Model Prediction")
    if confidence > 0.5:
        st.success(f"Strong Prediction: {teams[best_pred_idx]} ({confidence:.1%} confident)")
    elif confidence > 0.4:
        st.info(f"Moderate Prediction: {teams[best_pred_idx]} ({confidence:.1%} confident)")
    else:
        st.warning(f"Low Confidence Prediction: {teams[best_pred_idx]} ({confidence:.1%} confident)")

    # Additional confidence analysis
    st.subheader("Prediction Confidence Analysis")
    confidence_explanation = {
        (0.6, 1.0): "Very High Confidence - Model strongly favors this outcome",
        (0.5, 0.6): "High Confidence - Model shows clear preference",
        (0.4, 0.5): "Moderate Confidence - Model shows slight preference",
        (0.33, 0.4): "Low Confidence - Prediction uncertainty is high",
        (0, 0.33): "Very Low Confidence - Too close to call"
    }

    for (lower, upper), explanation in confidence_explanation.items():
        if confidence >= lower and confidence < upper:
            st.write(explanation)
            break

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Probability distribution
    labels = [f'{away_team}\n{probs[0]:.1%}', f'Draw\n{probs[1]:.1%}', f'{home_team}\n{probs[2]:.1%}']
    colors = ['lightblue' if i == best_pred_idx else 'lightgray' for i in range(3)]

    ax1.bar(range(3), probs, color=colors)
    ax1.set_ylabel('Probability')
    ax1.set_title('Model Prediction Probabilities')
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)

    # Confidence gauge
    gauge_colors = [(1, 0.7, 0.7), (1, 0.9, 0.7), (0.7, 1, 0.7)]  # Red to Yellow to Green
    confidence_level = confidence
    ax2.pie([confidence_level, 1 - confidence_level], colors=[gauge_colors[int(confidence_level * 2)], 'lightgray'],
            labels=[f'Confidence\n{confidence_level:.1%}', ''], startangle=90)
    ax2.set_title('Prediction Confidence Level')

    plt.tight_layout()
    st.pyplot(fig)

    # Detailed stats comparison
    st.subheader("Prediction Factors")
    factors_df = pd.DataFrame({
        'Factor': ['Model Confidence', 'Probability Margin', 'Prediction Strength'],
        'Value': [
            f"{confidence:.1%}",
            f"{(max(probs) - sorted(probs)[-2]):.1%}",
            "Strong" if confidence > 0.5 else "Moderate" if confidence > 0.4 else "Weak"
        ],
        'Interpretation': [
            "How confident the model is in its prediction",
            "Difference between best and second-best prediction",
            "Overall strength of the prediction signal"
        ]
    })
    st.table(factors_df)

    # Warning for low confidence predictions
    if confidence < 0.4:
        st.warning("""
        ⚠️ Low Confidence Prediction Warning:
        - The model shows significant uncertainty in this prediction
        - Consider waiting for more data or skipping this game
        - Multiple outcomes have similar probabilities
        """)

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