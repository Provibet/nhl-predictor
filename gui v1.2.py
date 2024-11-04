import tkinter as tk
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# Load the trained model and scaler
model = joblib.load('nhl_game_predictor_model_v1.2_gb_best.pkl')
scaler = joblib.load('scaler.pkl')


# Function to fetch team stats from the database
def get_team_stats(team):
    query = f"""
        SELECT "xGoalsPercentage", "corsiPercentage", "fenwickPercentage" 
        FROM "nhl23_matchups_with_situations" 
        WHERE "team" = '{team}' LIMIT 1;
    """
    stats = pd.read_sql(query, engine)
    return stats.iloc[0] if not stats.empty else None


# Function to fetch the win percentage for home and away teams
def get_team_win_percentage(team, is_home=True):
    if is_home:
        query = f"""
            SELECT AVG(CASE WHEN "Home Team Score" > "Away Team Score" THEN 1 ELSE 0 END) as "Home_win_percentage"
            FROM nhl23_results
            WHERE "Home Team" = '{team}';
        """
    else:
        query = f"""
            SELECT AVG(CASE WHEN "Away Team Score" > "Home Team Score" THEN 1 ELSE 0 END) as "Away_win_percentage"
            FROM nhl23_results
            WHERE "Away Team" = '{team}';
        """
    win_percentage = pd.read_sql(query, engine)
    return win_percentage.iloc[0, 0] if not win_percentage.empty else 0.5  # Return 50% if no data


# Prediction function
def predict_game(home_team, away_team, home_odds, away_odds):
    # Get stats for home and away teams
    home_team_stats = get_team_stats(home_team)
    away_team_stats = get_team_stats(away_team)

    if home_team_stats is None or away_team_stats is None:
        result_label.config(text="Error fetching team stats")
        return

    # Fetch home and away win percentages
    home_win_percentage = get_team_win_percentage(home_team, is_home=True)
    away_win_percentage = get_team_win_percentage(away_team, is_home=False)

    # Create feature set for prediction
    X_next_game = pd.DataFrame({
        'odds_ratio': [float(home_odds) / float(away_odds)],
        'xGoalsPercentage_diff': [home_team_stats['xGoalsPercentage'] - away_team_stats['xGoalsPercentage']],
        'shotsAttemptedPercentage_diff': [home_team_stats['corsiPercentage'] - away_team_stats['fenwickPercentage']],
        'Home_win_percentage': [home_win_percentage],
        'Away_win_percentage': [away_win_percentage]
    })

    # Scale features
    X_next_game_scaled = scaler.transform(X_next_game)

    # Make prediction
    prediction = model.predict(X_next_game_scaled)[0]
    prediction_probs = model.predict_proba(X_next_game_scaled)[0]  # Probability for both classes

    # Model predicted probabilities
    home_win_prob = prediction_probs[1]  # Assuming home team win is represented by 1
    away_win_prob = prediction_probs[0]  # Assuming away team win is represented by 0

    # Calculate implied probabilities
    implied_home_prob = 1 / float(home_odds)
    implied_away_prob = 1 / float(away_odds)

    # Determine value bets
    home_value_bet = "Yes" if implied_home_prob < home_win_prob else "No"
    away_value_bet = "Yes" if implied_away_prob < away_win_prob else "No"

    # Display the result
    result_label.config(text=f"Prediction: {'Home Team Wins' if prediction == 1 else 'Away Team Wins'}\n"
                             f"Confidence: {max(home_win_prob, away_win_prob) * 100:.2f}%\n"
                             f"Odds Ratio: {float(home_odds) / float(away_odds):.2f}\n\n"
                             f"xGoals Percentage Difference: {X_next_game['xGoalsPercentage_diff'][0]:.2f}\n"
                             f"Shots Attempted Difference: {X_next_game['shotsAttemptedPercentage_diff'][0]:.2f}\n"
                             f"Home Win Percentage: {home_win_percentage * 100:.2f}%\n"
                             f"Away Win Percentage: {away_win_percentage * 100:.2f}%\n\n"
                             f"Implied Home Probability: {implied_home_prob * 100:.2f}%\n"
                             f"Implied Away Probability: {implied_away_prob * 100:.2f}%\n\n"
                             f"Home Team Value Bet: {home_value_bet}\n"
                             f"Away Team Value Bet: {away_value_bet}")


# Create the main application window
root = tk.Tk()
root.title("NHL Game Predictor")

# Create dropdowns and entry boxes
home_team_label = tk.Label(root, text="Select Home Team:")
home_team_label.pack()
home_team_var = tk.StringVar(root)
home_team_dropdown = tk.OptionMenu(root, home_team_var, *sorted(
    pd.read_sql('SELECT DISTINCT "Home Team" FROM nhl23_results', engine)['Home Team']))
home_team_dropdown.pack()

away_team_label = tk.Label(root, text="Select Away Team:")
away_team_label.pack()
away_team_var = tk.StringVar(root)
away_team_dropdown = tk.OptionMenu(root, away_team_var, *sorted(
    pd.read_sql('SELECT DISTINCT "Away Team" FROM nhl23_results', engine)['Away Team']))
away_team_dropdown.pack()

home_odds_label = tk.Label(root, text="Enter Home Team Odds:")
home_odds_label.pack()
home_odds_entry = tk.Entry(root)
home_odds_entry.pack()

away_odds_label = tk.Label(root, text="Enter Away Team Odds:")
away_odds_label.pack()
away_odds_entry = tk.Entry(root)
away_odds_entry.pack()

# Prediction result label
result_label = tk.Label(root, text="", justify="left")
result_label.pack()

# Predict button
predict_button = tk.Button(root, text="Predict Outcome", command=lambda: predict_game(
    home_team_var.get(), away_team_var.get(), home_odds_entry.get(), away_odds_entry.get()))
predict_button.pack()

# Properly close the window
root.protocol("WM_DELETE_WINDOW", root.quit)

# Start the application
root.mainloop()