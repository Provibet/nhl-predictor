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


def get_team_stats(team):
    query = """
    SELECT 
        AVG("goalsFor") as goals_for,
        AVG("goalsAgainst") as goals_against,
        AVG("xGoalsPercentage") as xg_percentage,
        AVG("corsiPercentage") as corsi_percentage,
        AVG("fenwickPercentage") as fenwick_percentage
    FROM nhl24_matchups_with_situations
    WHERE team = %(team)s
    GROUP BY team
    """
    engine = init_connection()
    return pd.read_sql(query, engine, params={'team': team}).iloc[0]


# Simple UI
st.title("NHL Game Predictor 🏒")

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
            home_stats = get_team_stats(home_team)
            away_stats = get_team_stats(away_team)

            # Display basic stats comparison
            st.subheader("Team Statistics")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**{home_team}**")
                st.write(f"Goals For: {home_stats['goals_for']:.2f}")
                st.write(f"xG%: {home_stats['xg_percentage']:.1f}%")
                st.write(f"Corsi%: {home_stats['corsi_percentage']:.1f}%")

            with col2:
                st.write(f"**{away_team}**")
                st.write(f"Goals For: {away_stats['goals_for']:.2f}")
                st.write(f"xG%: {away_stats['xg_percentage']:.1f}%")
                st.write(f"Corsi%: {away_stats['corsi_percentage']:.1f}%")

            # Display raw data for debugging
            with st.expander("Debug Data"):
                st.write("Home Team Stats:")
                st.write(dict(home_stats))
                st.write("Away Team Stats:")
                st.write(dict(away_stats))

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
    3. Click "Analyze Match" for predictions
""")