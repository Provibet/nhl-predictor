import requests
import pandas as pd
from io import BytesIO
from sqlalchemy import create_engine

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Step 1: Define the URL to download the CSV file for the 2023 season
csv_url = 'https://moneypuck.com/moneypuck/playerData/seasonSummary/2023/regular/teams.csv'

# Step 2: Download the CSV file
response = requests.get(csv_url)
if response.status_code == 200:
    # Load the CSV file into a pandas DataFrame
    csv_data = pd.read_csv(BytesIO(response.content))
    print("CSV file downloaded and loaded successfully.")
else:
    print(f"Failed to download CSV file: {response.status_code}")
    exit()

# Step 3: Clean and process the data
# Rename columns for clarity (based on the column names in the CSV)
csv_data = csv_data.rename(columns={
    'Corsi%': 'shots_attempted_percentage',
    'Fenwick%': 'unblocked_shot_attempts_percentage',
    'xGoals%': 'xGoals_percentage'
})

# Example: Team name mapping (including ARI = Arizona Coyotes for the 2023 season)
team_name_mapping = {
    'ANA': 'Anaheim Ducks',
    'ARI': 'Arizona Coyotes',  # Adding ARI to the mapping
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
    'VAN': 'Vancouver Canucks',
    'VGK': 'Vegas Golden Knights',
    'WPG': 'Winnipeg Jets',
    'WSH': 'Washington Capitals'
}

# Apply team name mapping
if 'team' in csv_data.columns:
    csv_data['team'] = csv_data['team'].map(team_name_mapping)

# Step 4: Insert additional columns such as differences in stats
# Calculate differences between relevant stats (based on your earlier cleaning steps)
if 'xGoals_percentage' in csv_data.columns:
    csv_data['xGoalsPercentage_diff'] = csv_data['xGoalsFor'] - csv_data['xGoalsAgainst']
    csv_data['shotsAttemptedPercentage_diff'] = csv_data['shots_attempted_percentage'] - csv_data['shots_attempted_percentage']
    csv_data['unblockedShotAttemptsPercentage_diff'] = csv_data['unblocked_shot_attempts_percentage'] - csv_data['unblocked_shot_attempts_percentage']

# Step 5: Set up PostgreSQL connection
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(connection_string)

# Step 6: Insert the new data into the PostgreSQL table named 'nhl23_matchups_with_situations'
table_name = 'nhl23_matchups_with_situations'

# Insert new data into the PostgreSQL table (replace if needed)
csv_data.to_sql(table_name, engine, if_exists='replace', index=False)

print(f"Data for the 2023 season successfully inserted into {table_name}.")
