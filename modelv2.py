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

# 1. Load data from PostgreSQL using correct column names
# Load nhl23_results from the PostgreSQL database
nhl23_results_query = """
    SELECT "Date", "Home Team", "Home Team Score", "Away Team", "Away Team Score", "Status"
    FROM nhl23_results
"""
nhl23_results = pd.read_sql(nhl23_results_query, engine)

# Load nhl23_odds from the PostgreSQL database
nhl23_odds_query = """
    SELECT "Date", "Home_Team", "Away_Team", "Home_Odds", "Draw_Odds", "Away_Odds"
    FROM nhl23_odds
"""
nhl23_odds = pd.read_sql(nhl23_odds_query, engine)

# Load nhl24_odds from the PostgreSQL database
nhl24_odds_query = """
    SELECT "Date", "Home_Team", "Away_Team", "Home_Odds", "Draw_Odds", "Away_Odds"
    FROM nhl24_odds
"""
nhl24_odds = pd.read_sql(nhl24_odds_query, engine)

# Combine nhl23_odds and nhl24_odds for complete odds data
combined_odds_data = pd.concat([nhl23_odds, nhl24_odds])

# 2. Merge results with odds data
merged_data = pd.merge(nhl23_results, combined_odds_data, how='left',
                       left_on=['Date', 'Home Team', 'Away Team'],
                       right_on=['Date', 'Home_Team', 'Away_Team'])

# 3. Load nhl24_matchups_with_situations from the PostgreSQL database
nhl24_matchups_query = """
    SELECT *
    FROM nhl24_matchups_with_situations
"""
nhl24_matchups = pd.read_sql(nhl24_matchups_query, engine)

# Merge nhl24_matchups data into the main dataset
merged_data = pd.merge(merged_data, nhl24_matchups, how='left', left_on='Home Team', right_on='team')

# 4. Calculate odds ratio
merged_data['odds_ratio'] = merged_data['Home_Odds'] / merged_data['Away_Odds']

# 5. Filter for Powerplay (PP) situations: e.g., 5-on-4
pp_situations = nhl24_matchups[nhl24_matchups['situation'].str.contains('5on4')]

# 6. Filter for Penalty Kill (PK) situations: e.g., 4-on-5
pk_situations = nhl24_matchups[nhl24_matchups['situation'].str.contains('4on5')]

# 7. Aggregate stats for PP and PK situations using xGoalsPercentage (you can also use corsiPercentage, fenwickPercentage)
# Example: Aggregating 'xGoalsPercentage' for Powerplay (PP)
pp_effectiveness = pp_situations.groupby('team')['xGoalsPercentage'].mean().reset_index()
pp_effectiveness.columns = ['team', 'PP_Effectiveness']

# Example: Aggregating 'xGoalsPercentage' for Penalty Kill (PK)
pk_effectiveness = pk_situations.groupby('team')['xGoalsPercentage'].mean().reset_index()
pk_effectiveness.columns = ['team', 'PK_Effectiveness']

# 8. Merge PP and PK effectiveness with the main dataset (merged_data)

# Merge PP effectiveness for the home team
merged_data = pd.merge(merged_data, pp_effectiveness, how='left', left_on='Home Team', right_on='team', suffixes=('_home', '_away'))

# Merge PP effectiveness for the away team
merged_data = pd.merge(merged_data, pp_effectiveness, how='left', left_on='Away Team', right_on='team', suffixes=('_home', '_away'))

# Rename PP columns for clarity
merged_data.rename(columns={
    'PP_Effectiveness_home': 'Home_PP_Effectiveness',
    'PP_Effectiveness_away': 'Away_PP_Effectiveness'
}, inplace=True)

# Now, merge PK effectiveness without using suffixes to avoid conflict
# Merge PK effectiveness for the home team
merged_data = pd.merge(merged_data, pk_effectiveness, how='left', left_on='Home Team', right_on='team')

# Rename the column for home team PK effectiveness
merged_data.rename(columns={'PK_Effectiveness': 'Home_PK_Effectiveness'}, inplace=True)

# Merge PK effectiveness for the away team
merged_data = pd.merge(merged_data, pk_effectiveness, how='left', left_on='Away Team', right_on='team')

# Rename the column for away team PK effectiveness
merged_data.rename(columns={'PK_Effectiveness': 'Away_PK_Effectiveness'}, inplace=True)

# 9. Calculate the difference between home and away team's Powerplay and Penalty Kill effectiveness
merged_data['PP_diff'] = merged_data['Home_PP_Effectiveness'] - merged_data['Away_PP_Effectiveness']
merged_data['PK_diff'] = merged_data['Home_PK_Effectiveness'] - merged_data['Away_PK_Effectiveness']

# 10. Calculate rolling averages for xGoalsPercentage and shotsAttemptedPercentage
# Rolling window size
window_size = 5

# Create a new DataFrame for home and away rolling calculations
home_rolling = merged_data.groupby('Home Team')['xGoalsPercentage'].apply(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()).reset_index(drop=True)

away_rolling = merged_data.groupby('Away Team')['xGoalsPercentage'].apply(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()).reset_index(drop=True)

# Align home and away rolling averages by re-assigning them to merged_data
merged_data['rolling_xGoalsPercentage_diff'] = home_rolling - away_rolling

# Similarly calculate rolling_shotsAttemptedPercentage_diff
home_shots_rolling = merged_data.groupby('Home Team')['corsiPercentage'].apply(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()).reset_index(drop=True)

away_shots_rolling = merged_data.groupby('Away Team')['fenwickPercentage'].apply(
    lambda x: x.rolling(window=window_size, min_periods=1).mean()).reset_index(drop=True)

merged_data['rolling_shotsAttemptedPercentage_diff'] = home_shots_rolling - away_shots_rolling

# 11. Drop any rows with missing values after calculating features
merged_data.dropna(inplace=True)

# 12. Define features (X) and target (y)
X = merged_data[['odds_ratio', 'PP_diff', 'PK_diff', 'rolling_xGoalsPercentage_diff', 'rolling_shotsAttemptedPercentage_diff']]
y = (merged_data['Home Team Score'] > merged_data['Away Team Score']).astype(int)

# 13. Train the Model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Gradient Boosting Classifier model
model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the updated model and the scaler
joblib.dump(model, 'nhl_game_predictor_model_with_situations.pkl')
joblib.dump(scaler, 'scaler_with_situations.pkl')
