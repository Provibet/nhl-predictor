import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.impute import SimpleImputer
import numpy as np

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# Load matchup and odds data from the database
nhl23_odds = pd.read_sql("SELECT * FROM nhl23_odds", engine)
nhl23_results = pd.read_sql("SELECT * FROM nhl23_results", engine)
nhl23_matchups_with_situations = pd.read_sql("SELECT * FROM nhl23_matchups_with_situations", engine)

nhl24_odds = pd.read_sql("SELECT * FROM nhl24_odds", engine)
nhl24_results = pd.read_sql("SELECT * FROM nhl24_results", engine)
nhl24_matchups_with_situations = pd.read_sql("SELECT * FROM nhl24_matchups_with_situations", engine)

# Load player stats from the database
nhl23_goalie_stats = pd.read_sql("SELECT * FROM nhl23_goalie_stats", engine)
nhl24_goalie_stats = pd.read_sql("SELECT * FROM nhl24_goalie_stats", engine)
nhl23_skater_stats = pd.read_sql("SELECT * FROM nhl23_skater_stats", engine)
nhl24_skater_stats = pd.read_sql("SELECT * FROM nhl24_skater_stats", engine)

# Define a mapping between team abbreviations and full names
team_abbreviation_mapping = {
    'ANA': 'Anaheim Ducks',
    'ARI': 'Arizona Coyotes',
    'BOS': 'Boston Bruins',
    'BUF': 'Buffalo Sabres',
    'CAR': 'Carolina Hurricanes',
    'CBJ': 'Columbus Blue Jackets',
    'CGY': 'Calgary Flames',
    'CHI': 'Chicago Blackhawks',
    'COL': 'Colorado Avalanche',
    'DAL': 'Dallas Stars',
    'DET': 'Detroit Red Wings',
    'EDM': 'Edmonton Oilers',
    'FLA': 'Florida Panthers',
    'LAK': 'Los Angeles Kings',
    'MIN': 'Minnesota Wild',
    'MTL': 'Montreal Canadiens',
    'NJD': 'New Jersey Devils',
    'NSH': 'Nashville Predators',
    'NYI': 'New York Islanders',
    'NYR': 'New York Rangers',
    'OTT': 'Ottawa Senators',
    'PHI': 'Philadelphia Flyers',
    'PIT': 'Pittsburgh Penguins',
    'SEA': 'Seattle Kraken',
    'SJS': 'San Jose Sharks',
    'STL': 'St. Louis Blues',
    'TBL': 'Tampa Bay Lightning',
    'TOR': 'Toronto Maple Leafs',
    'UTA': 'Utah Hockey Club',
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets',
    'WSH': 'Washington Capitals'
}

# Apply the team abbreviation mapping to all player stat tables
nhl23_goalie_stats['team'] = nhl23_goalie_stats['team'].map(team_abbreviation_mapping)
nhl24_goalie_stats['team'] = nhl24_goalie_stats['team'].map(team_abbreviation_mapping)
nhl23_skater_stats['team'] = nhl23_skater_stats['team'].map(team_abbreviation_mapping)
nhl24_skater_stats['team'] = nhl24_skater_stats['team'].map(team_abbreviation_mapping)

# Handle the team change between seasons (Arizona Coyotes -> Utah Hockey Club)
nhl23_goalie_stats = nhl23_goalie_stats[nhl23_goalie_stats['team'] != 'Utah Hockey Club']
nhl24_goalie_stats = nhl24_goalie_stats[nhl24_goalie_stats['team'] != 'Arizona Coyotes']

nhl23_skater_stats = nhl23_skater_stats[nhl23_skater_stats['team'] != 'Utah Hockey Club']
nhl24_skater_stats = nhl24_skater_stats[nhl24_skater_stats['team'] != 'Arizona Coyotes']

# Merge results for both 2023 and 2024 seasons
# For 2023
nhl23_results = nhl23_results.rename(
    columns={'Home Team Score': 'Home_Team_Score', 'Away Team Score': 'Away_Team_Score'})
merged_results_23 = pd.merge(nhl23_matchups_with_situations, nhl23_results, left_on='team', right_on='Home Team',
                             how='inner')

# For 2024
nhl24_results = nhl24_results.rename(
    columns={'home_team_score': 'Home_Team_Score', 'away_team_score': 'Away_Team_Score'})
merged_results_24 = pd.merge(nhl24_matchups_with_situations, nhl24_results, left_on='team', right_on='home_team',
                             how='inner')


# Aggregate skater and goalie stats for each team
def aggregate_skater_stats(skater_stats):
    available_columns = skater_stats.columns
    aggregation = {}

    if 'goals' in available_columns:
        aggregation['goals'] = 'sum'
    if 'assists' in available_columns:
        aggregation['assists'] = 'sum'
    if 'points' in available_columns:
        aggregation['points'] = 'sum'
    if 'shots' in available_columns:
        aggregation['shots'] = 'sum'
    if 'blocks' in available_columns:
        aggregation['blocks'] = 'sum'
    if 'hits' in available_columns:
        aggregation['hits'] = 'sum'
    if 'giveaways' in available_columns:
        aggregation['giveaways'] = 'sum'
    if 'takeaways' in available_columns:
        aggregation['takeaways'] = 'sum'

    team_stats = skater_stats.groupby('team').agg(aggregation).reset_index()
    return team_stats


def aggregate_goalie_stats(goalie_stats):
    team_goalie_stats = goalie_stats.groupby('team').agg({
        'wins': 'sum',
        'losses': 'sum',
        'save_percentage': 'mean',
        'gaa': 'mean',
        'goals_saved_above_average': 'mean'
    }).reset_index()
    return team_goalie_stats


# Aggregate skater and goalie stats for both seasons
team_skater_stats_23 = aggregate_skater_stats(nhl23_skater_stats)
team_skater_stats_24 = aggregate_skater_stats(nhl24_skater_stats)

team_goalie_stats_23 = aggregate_goalie_stats(nhl23_goalie_stats)
team_goalie_stats_24 = aggregate_goalie_stats(nhl24_goalie_stats)

# Now we need to join these aggregated team-level stats with the matchups for both seasons
# Merge team stats with matchups data for both seasons
merged_results_23 = pd.merge(merged_results_23, team_skater_stats_23, left_on='team', right_on='team', how='left')
merged_results_23 = pd.merge(merged_results_23, team_goalie_stats_23, left_on='team', right_on='team', how='left')

merged_results_24 = pd.merge(merged_results_24, team_skater_stats_24, left_on='team', right_on='team', how='left')
merged_results_24 = pd.merge(merged_results_24, team_goalie_stats_24, left_on='team', right_on='team', how='left')

# Add odds columns and process them
merged_results_23 = pd.merge(merged_results_23, nhl23_odds, left_on='team', right_on='Home_Team', how='inner')
merged_results_24 = pd.merge(merged_results_24, nhl24_odds, left_on='team', right_on='Home_Team', how='inner')

# Ensure the odds columns are numeric types
merged_results_23['Home_Odds'] = pd.to_numeric(merged_results_23['Home_Odds'], errors='coerce')
merged_results_23['Away_Odds'] = pd.to_numeric(merged_results_23['Away_Odds'], errors='coerce')
merged_results_24['Home_Odds'] = pd.to_numeric(merged_results_24['Home_Odds'], errors='coerce')
merged_results_24['Away_Odds'] = pd.to_numeric(merged_results_24['Away_Odds'], errors='coerce')

# Feature Engineering: Odds ratio, team performance metrics
merged_results_23['odds_ratio'] = merged_results_23['Home_Odds'] / merged_results_23['Away_Odds']
merged_results_24['odds_ratio'] = merged_results_24['Home_Odds'] / merged_results_24['Away_Odds']

# Merge all features for 2023 and 2024 into the main dataset for the model
merged_data_all = pd.concat([merged_results_23, merged_results_24])

# Drop rows with missing data from both features and target variable
merged_data_all.dropna(subset=['odds_ratio', 'Home_Team_Score', 'Away_Team_Score'], inplace=True)

# Re-create the feature set and target variable
X = merged_data_all[
    ['odds_ratio', 'goals', 'assists', 'points', 'shots', 'blocks', 'hits', 'giveaways', 'takeaways', 'save_percentage',
     'gaa', 'goals_saved_above_average']]
y = (merged_data_all['Home_Team_Score'] > merged_data_all['Away_Team_Score']).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values by replacing them with the mean value of each column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Use BalancedRandomForestClassifier for class imbalance handling
model = BalancedRandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, sampling_strategy='all',
                                       replacement=True, bootstrap=False)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Save the best model and the scaler
joblib.dump(model, 'nhl_game_predictor_model_v1.3_rf_best.pkl')
joblib.dump(scaler, 'scaler_v1.3.pkl')
