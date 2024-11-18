import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Page config
st.set_page_config(page_title="NHL Game Predictor", page_icon="🏒")


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


# Get teams from database
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
    """Get basic team statistics"""
    query = """
    SELECT 
        AVG("xGoalsPercentage") as xg_percentage,
        AVG("corsiPercentage") as corsi_percentage,
        AVG("fenwickPercentage") as fenwick_percentage
    FROM nhl24_matchups_with_situations
    WHERE team = %(team)s
    GROUP BY team
    """
    engine = init_connection()
    return pd.read_sql(query, engine, params={'team': team}).iloc[0]


def prepare_basic_features(home_team, away_team, home_odds, away_odds, draw_odds):
    """Prepare basic features for prediction"""
    try:
        # Get team stats
        home_stats = get_team_stats(home_team)
        away_stats = get_team_stats(away_team)

        # Calculate implied probabilities
        total_odds = (1 / home_odds) + (1 / away_odds) + (1 / draw_odds)
        home_implied = (1 / home_odds) / total_odds
        away_implied = (1 / away_odds) / total_odds
        draw_implied = (1 / draw_odds) / total_odds

        # Create features DataFrame
        features = pd.DataFrame([{
            'home_xGoalsPercentage': home_stats['xg_percentage'],
            'away_xGoalsPercentage': away_stats['xg_percentage'],
            'home_corsiPercentage': home_stats['corsi_percentage'],
            'away_corsiPercentage': away_stats['corsi_percentage'],
            'home_fenwickPercentage': home_stats['fenwick_percentage'],
            'away_fenwickPercentage': away_stats['fenwick_percentage'],
            'home_implied_prob_normalized': home_implied,
            'away_implied_prob_normalized': away_implied,
            'draw_implied_prob_normalized': draw_implied
        }])

        return features

    except Exception as e:
        st.error(f"Error preparing features: {str(e)}")
        return None


def main():
    st.title("NHL Game Predictor 🏒")

    # Load model
    model = load_model_from_drive()

    if model is None:
        st.error("Failed to load model. Please check the connection.")
        return

    # Create interface
    col1, col2 = st.columns(2)

    with col1:
        home_team = st.selectbox("Home Team", get_teams())
        home_odds = st.number_input("Home Odds", min_value=1.0, value=2.0, step=0.05)

    with col2:
        away_team = st.selectbox("Away Team", get_teams())
        away_odds = st.number_input("Away Odds", min_value=1.0, value=2.0, step=0.05)

    draw_odds = st.number_input("Draw Odds", min_value=1.0, value=3.5, step=0.05)

    if st.button("Get Prediction"):
        if home_team == away_team:
            st.error("Please select different teams")
            return

        # Prepare features and make prediction
        features = prepare_basic_features(home_team, away_team, home_odds, away_odds, draw_odds)

        if features is not None:
            probabilities = model.predict_proba(features)[0]

            # Display results
            st.header("Prediction Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Home Win",
                    f"{probabilities[2]:.1%}",
                    f"Odds: {home_odds:.2f}"
                )

            with col2:
                st.metric(
                    "Draw",
                    f"{probabilities[1]:.1%}",
                    f"Odds: {draw_odds:.2f}"
                )

            with col3:
                st.metric(
                    "Away Win",
                    f"{probabilities[0]:.1%}",
                    f"Odds: {away_odds:.2f}"
                )

            # Show best value bet
            implied_probs = [1 / away_odds, 1 / draw_odds, 1 / home_odds]
            total_implied = sum(implied_probs)
            normalized_implied = [p / total_implied for p in implied_probs]

            edges = [prob - imp for prob, imp in zip(probabilities, normalized_implied)]
            best_edge_idx = np.argmax(edges)

            if edges[best_edge_idx] > 0:
                bet_type = ["Away Win", "Draw", "Home Win"][best_edge_idx]
                odds = [away_odds, draw_odds, home_odds][best_edge_idx]
                st.success(f"Best Value: {bet_type} @ {odds:.2f} (Edge: {edges[best_edge_idx]:.1%})")


if __name__ == "__main__":
    main()