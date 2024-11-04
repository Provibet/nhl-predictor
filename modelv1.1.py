import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# Load data from the database
nhl23_odds = pd.read_sql("SELECT * FROM nhl23_odds", engine)
nhl23_results = pd.read_sql("SELECT * FROM nhl23_results", engine)
nhl23_matchups_with_situations = pd.read_sql("SELECT * FROM nhl23_matchups_with_situations", engine)

# Merge odds and matchups (Using home team 'team' from matchups)
merged_data = pd.merge(nhl23_matchups_with_situations, nhl23_odds,
                       left_on='team',
                       right_on='home_team',
                       how='inner')

# Feature 1: Odds ratio between home and away teams
merged_data['odds_ratio'] = merged_data['home_odds'] / merged_data['away_odds']

# Feature 2: xGoals percentage difference, shots attempted difference, etc.
merged_data['xGoalsPercentage_diff'] = merged_data['xGoalsPercentage'] - merged_data['xGoalsPercentage']
merged_data['shotsAttemptedPercentage_diff'] = merged_data['corsiPercentage'] - merged_data['fenwickPercentage']

# Select features for modeling
X = merged_data[['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff']]

# Merge with results to create the target variable
results_data_filtered = nhl23_results[['Away Team', 'Home Team', 'Away Team Score', 'Home Team Score']]
merged_data_results = pd.merge(merged_data, results_data_filtered,
                               left_on='team', right_on='Home Team', how='inner')

# Create the target variable: 1 if home team won, 0 otherwise
y = (merged_data_results['Home Team Score'] > merged_data_results['Away Team Score']).astype(int)

# Drop any rows with missing data
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

# Train a Gradient Boosting Classifier model
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the model and the scaler
joblib.dump(model, 'nhl_game_predictor_model_v1.1_gb_best.pkl')
joblib.dump(scaler, 'scaler.pkl')
