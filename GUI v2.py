# File: nhl_prediction_gui.py
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sqlalchemy import create_engine
import joblib
from datetime import datetime


class NHLPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NHL Game Predictor")

        # Database connection
        self.engine = create_engine('postgresql://postgres:Provibet2024@localhost:5432/Provibet_NHL')

        # Load the model
        self.model = joblib.load('nhl_game_predictor_model_v3.0_enhanced.pkl')

        # Get list of teams
        self.teams = self.get_teams_list()

        self.create_gui()

    def get_teams_list(self):
        """Get list of NHL teams from database"""
        query = "SELECT DISTINCT team FROM nhl24_matchups_with_situations ORDER BY team"
        teams = pd.read_sql(query, self.engine)
        return teams['team'].tolist()

    def create_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Team Selection
        ttk.Label(main_frame, text="Home Team:").grid(row=0, column=0, padx=5, pady=5)
        self.home_team = ttk.Combobox(main_frame, values=self.teams)
        self.home_team.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(main_frame, text="Away Team:").grid(row=1, column=0, padx=5, pady=5)
        self.away_team = ttk.Combobox(main_frame, values=self.teams)
        self.away_team.grid(row=1, column=1, padx=5, pady=5)

        # Odds Entry
        ttk.Label(main_frame, text="Home Team Odds:").grid(row=2, column=0, padx=5, pady=5)
        self.home_odds = ttk.Entry(main_frame)
        self.home_odds.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(main_frame, text="Away Team Odds:").grid(row=3, column=0, padx=5, pady=5)
        self.away_odds = ttk.Entry(main_frame)
        self.away_odds.grid(row=3, column=1, padx=5, pady=5)

        # Predict Button
        ttk.Button(main_frame, text="Predict", command=self.make_prediction).grid(row=4, column=0, columnspan=2,
                                                                                  pady=20)

        # Results Display
        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.result_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        # Results Labels
        self.predicted_winner_label = ttk.Label(self.result_frame, text="")
        self.predicted_winner_label.grid(row=0, column=0, columnspan=2)

        self.confidence_label = ttk.Label(self.result_frame, text="")
        self.confidence_label.grid(row=1, column=0, columnspan=2)

        self.high_confidence_label = ttk.Label(self.result_frame, text="")
        self.high_confidence_label.grid(row=2, column=0, columnspan=2)

    def get_team_stats(self, team):
        """Get recent team statistics from database"""
        query = f"""
        SELECT 
            AVG(goalsFor) as goals_for,
            AVG(goalsAgainst) as goals_against,
            AVG(shotsOnGoalFor) as shots_on_goal_for,
            AVG(shotsOnGoalAgainst) as shots_on_goal_against,
            AVG(xGoalsPercentage) as xgoals_percentage,
            AVG(corsiPercentage) as corsi_percentage,
            AVG(fenwickPercentage) as fenwick_percentage,
            AVG(scoringEfficiency) as scoring_efficiency,
            AVG(defensiveEfficiency) as defensive_efficiency
        FROM nhl24_matchups_with_situations
        WHERE team = '{team}'
        AND game_date >= CURRENT_DATE - INTERVAL '30 days'
        """
        return pd.read_sql(query, self.engine).iloc[0]

    def prepare_features(self, home_team, away_team, home_odds, away_odds):
        """Prepare features for prediction"""
        # Get team stats
        home_stats = self.get_team_stats(home_team)
        away_stats = self.get_team_stats(away_team)

        # Calculate implied probabilities
        home_implied_prob = 1 / float(home_odds)
        away_implied_prob = 1 / float(away_odds)
        market_efficiency = home_implied_prob + away_implied_prob

        # Create feature dictionary
        features = {
            'goalsFor': home_stats['goals_for'],
            'goalsAgainst': home_stats['goals_against'],
            'shotsOnGoalFor': home_stats['shots_on_goal_for'],
            'shotsOnGoalAgainst': home_stats['shots_on_goal_against'],
            'xGoalsPercentage': home_stats['xgoals_percentage'],
            'corsiPercentage': home_stats['corsi_percentage'],
            'fenwickPercentage': home_stats['fenwick_percentage'],
            'scoring_efficiency': home_stats['scoring_efficiency'],
            'defensive_efficiency': home_stats['defensive_efficiency'],
            'is_home': 1,
            'home_implied_prob': home_implied_prob,
            'away_implied_prob': away_implied_prob,
            'market_efficiency': market_efficiency
        }

        return pd.DataFrame([features])

    def make_prediction(self):
        """Make prediction based on input"""
        try:
            # Get values from GUI
            home_team = self.home_team.get()
            away_team = self.away_team.get()
            home_odds = float(self.home_odds.get())
            away_odds = float(self.away_odds.get())

            # Input validation
            if not all([home_team, away_team, home_odds, away_odds]):
                messagebox.showerror("Error", "Please fill in all fields")
                return

            if home_team == away_team:
                messagebox.showerror("Error", "Home and Away teams must be different")
                return

            # Prepare features
            features = self.prepare_features(home_team, away_team, home_odds, away_odds)

            # Make prediction
            win_probability = self.model.predict_proba(features)[0]
            predicted_winner = home_team if win_probability[1] > win_probability[0] else away_team
            confidence = max(win_probability)

            # Update results display
            self.predicted_winner_label.config(
                text=f"Predicted Winner: {predicted_winner}",
                font=('Arial', 12, 'bold')
            )

            self.confidence_label.config(
                text=f"Confidence: {confidence:.1%}"
            )

            if confidence >= 0.65:
                self.high_confidence_label.config(
                    text="*** HIGH CONFIDENCE PICK ***",
                    foreground='green'
                )
            else:
                self.high_confidence_label.config(
                    text="Standard Confidence Pick",
                    foreground='black'
                )

        except ValueError:
            messagebox.showerror("Error", "Please enter valid odds values")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = NHLPredictionGUI(root)
    root.mainloop()