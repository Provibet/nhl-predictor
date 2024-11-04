import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load data from CSV files (adjust paths if needed)
nhl23_odds = pd.read_csv('D:/CGPT Uploads/nhl23_odds.csv')
nhl23_results = pd.read_csv('D:/CGPT Uploads/nhl23_results.csv')
nhl23_matchups_with_situations = pd.read_csv('D:/CGPT Uploads/nhl23_matchups_with_situations.csv')

# Step 1: Merge odds and matchups (Using home team 'team' from matchups)
merged_data = pd.merge(nhl23_matchups_with_situations, nhl23_odds,
                       left_on='team',
                       right_on='home_team',
                       how='inner')

# Features available BEFORE the game (avoid goal_diff, final score, etc.)
# Feature 1: Odds ratio between home and away teams
merged_data['odds_ratio'] = merged_data['home_odds'] / merged_data['away_odds']

# Feature 2: xGoals percentage difference, shots attempted difference, etc. (using available stats)
# You'll need to create these as difference metrics between home and away team performance
merged_data['xGoalsPercentage_diff'] = merged_data['xGoalsPercentage'] - merged_data['xGoalsPercentage']  # Adjust as needed
merged_data['shotsAttemptedPercentage_diff'] = merged_data['corsiPercentage'] - merged_data['fenwickPercentage']  # Adjust as needed

# Selecting features for modeling
X = merged_data[['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff']]

# Step 2: Merge with results to create the target variable
results_data_filtered = nhl23_results[['Away Team', 'Home Team', 'Away Team Score', 'Home Team Score']]
merged_data_results = pd.merge(merged_data, results_data_filtered,
                               left_on='team', right_on='Home Team', how='inner')

# Create the target variable: 1 if home team won, 0 otherwise
y = (merged_data_results['Home Team Score'] > merged_data_results['Away Team Score']).astype(int)

# Step 3: Drop any rows with missing data
merged_data_results.dropna(subset=['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff', 'Home Team Score', 'Away Team Score'], inplace=True)

# Re-create the features and target variable after dropping missing rows
X = merged_data_results[['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff']]
y = (merged_data_results['Home Team Score'] > merged_data_results['Away Team Score']).astype(int)

# Ensure consistent sizes between X and y
assert len(X) == len(y), "Feature set X and target variable y should have the same number of rows."

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Example: Calculate rolling average of goals scored by home and away teams over last 5 games
nhl23_results['home_team_rolling_avg'] = nhl23_results.groupby('Home Team')['Home Team Score'].transform(lambda x: x.rolling(5, min_periods=1).mean())
nhl23_results['away_team_rolling_avg'] = nhl23_results.groupby('Away Team')['Away Team Score'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Example: Calculate rolling win percentage of home and away teams over the last 5 games
nhl23_results['home_win'] = (nhl23_results['Home Team Score'] > nhl23_results['Away Team Score']).astype(int)
nhl23_results['home_team_rolling_win_pct'] = nhl23_results.groupby('Home Team')['home_win'].transform(lambda x: x.rolling(5, min_periods=1).mean())
nhl23_results['away_team_rolling_win_pct'] = nhl23_results.groupby('Away Team')['home_win'].transform(lambda x: (1 - x).rolling(5, min_periods=1).mean())

# Merge rolling stats into the merged dataset
merged_data_results = pd.merge(merged_data, nhl23_results[['Away Team', 'Home Team', 'home_team_rolling_win_pct', 'away_team_rolling_win_pct']],
                               left_on='team', right_on='Home Team', how='inner')

# Update feature set with rolling stats
X = merged_data_results[['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff', 'home_team_rolling_win_pct', 'away_team_rolling_win_pct']]

# Train a RandomForestClassifier as the final model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the model using joblib
joblib.dump(model, 'nhl_game_predictor_model_v1.pkl')
print("Model saved to 'nhl_game_predictor_model_v1.pkl'")
