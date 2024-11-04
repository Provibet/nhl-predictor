import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sqlalchemy import create_engine
import joblib

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# Load the data from the database
nhl23_odds = pd.read_sql("SELECT * FROM nhl23_odds", engine)
nhl23_results = pd.read_sql("SELECT * FROM nhl23_results", engine)
nhl23_matchups_with_situations = pd.read_sql("SELECT * FROM nhl23_matchups_with_situations", engine)
print(nhl23_odds.columns)

# Merge odds and matchups data
merged_data = pd.merge(nhl23_matchups_with_situations, nhl23_odds, left_on='team', right_on='Home_Team', how='inner')

# Feature engineering: calculating odds ratio, xGoals percentage difference, and shots attempted difference
merged_data['odds_ratio'] = merged_data['Home_Odds'] / merged_data['Away_Odds']
merged_data['xGoalsPercentage_diff'] = merged_data['xGoalsPercentage'] - merged_data['xGoalsPercentage']
merged_data['shotsAttemptedPercentage_diff'] = merged_data['corsiPercentage'] - merged_data['fenwickPercentage']

# Calculate the home and away win percentages
home_win_data = nhl23_results.groupby('Home Team').agg(
    home_wins=('Home Team Score', lambda x: (x > nhl23_results['Away Team Score']).sum()),
    home_losses=('Home Team Score', lambda x: (x < nhl23_results['Away Team Score']).sum())
)
home_win_data['Home_win_percentage'] = home_win_data['home_wins'] / (home_win_data['home_wins'] + home_win_data['home_losses'])

away_win_data = nhl23_results.groupby('Away Team').agg(
    away_wins=('Away Team Score', lambda x: (x > nhl23_results['Home Team Score']).sum()),
    away_losses=('Away Team Score', lambda x: (x < nhl23_results['Home Team Score']).sum())
)
away_win_data['Away_win_percentage'] = away_win_data['away_wins'] / (away_win_data['away_wins'] + away_win_data['away_losses'])

# Merge the win percentages back into the merged_data DataFrame
merged_data = pd.merge(merged_data, home_win_data[['Home_win_percentage']], left_on='team', right_index=True, how='left')
merged_data = pd.merge(merged_data, away_win_data[['Away_win_percentage']], left_on='team', right_index=True, how='left')

# Drop rows with missing values
merged_data.dropna(subset=['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff', 'Home_win_percentage', 'Away_win_percentage'], inplace=True)

# Feature matrix (X) and target variable (y)
X = merged_data[['odds_ratio', 'xGoalsPercentage_diff', 'shotsAttemptedPercentage_diff', 'Home_win_percentage', 'Away_win_percentage']]
y = (merged_data['Home Team Score'] > merged_data['Away Team Score']).astype(int)  # 1 if home team wins, 0 if away team wins

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the model and the scaler
joblib.dump(model, 'nhl_game_predictor_model_v1.2_gb_best.pkl')
joblib.dump(scaler, 'scaler.pkl')

