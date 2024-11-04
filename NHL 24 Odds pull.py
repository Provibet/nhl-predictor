import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, text

# PostgreSQL connection details
host = "localhost"  # Change to your PostgreSQL host if needed
port = "5432"
dbname = "Provibet_NHL"  # Your PostgreSQL database name
user = "postgres"  # Your PostgreSQL username
password = "Provibet2024"  # Your PostgreSQL password

# Create a connection to the PostgreSQL database
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# URL of the page to scrape
url = 'https://checkbestodds.com/hockey-odds/archive-nhl/'

# Send an HTTP request to the page
response = requests.get(url)

# Parse the page content with BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Create lists to store the extracted data
dates = []
matches = []
odds_1 = []
odds_x = []
odds_2 = []

# Variable to hold the current date for match rows
current_date = None

# Find all rows in the table containing the odds data
for row in soup.find_all('tr'):
    # Check if the row is a date row (tr with class 'higher')
    if 'higher' in row.get('class', []):
        # Extract the date from this row
        date_cell = row.find('td')
        if date_cell:
            current_date = date_cell.text.strip()  # Store the date for following matches

    # Now check for match and odds in rows
    match_cell = row.find('td', class_='match')  # Match info (team names)
    time_cell = row.find('span', class_='time')  # Match time

    # Find all <td> with class "r" for the odds
    odds_cells = row.find_all('td', class_='r')

    # Ensure we have enough odds columns (3 odds: for home win, draw, away win)
    if len(odds_cells) == 3 and match_cell and time_cell:
        odds_1_val = odds_cells[0]  # First odds (home win)
        odds_x_val = odds_cells[1]  # Second odds (draw)
        odds_2_val = odds_cells[2]  # Third odds (away win)

        # Remove the time (in the format "18:00") from the match string
        match_text = match_cell.text.strip()
        match_text = match_text.replace(time_cell.text.strip(), '').strip()

        # Append the data to lists
        dates.append(current_date)  # Use the stored date
        matches.append(match_text)  # Extract cleaned match details (team names)
        odds_1.append(odds_1_val.text.strip())  # Extract odds for home win
        odds_x.append(odds_x_val.text.strip())  # Extract odds for draw
        odds_2.append(odds_2_val.text.strip())  # Extract odds for away win

# Create a DataFrame with the collected data
data = pd.DataFrame({
    'Date': dates,
    'Match': matches,
    'Odds 1': odds_1,
    'Odds X': odds_x,
    'Odds 2': odds_2
})

# Split the "Match" column into two separate columns: "Home Team" and "Away Team"
data[['Home_Team', 'Away_Team']] = data['Match'].str.split(' - ', expand=True)

# Drop the original "Match" column as it's now redundant
data.drop(columns=['Match'], inplace=True)

# Rename columns and reorder as requested
data = data.rename(columns={
    'Home_Team': 'Home_Team',
    'Away_Team': 'Away_Team',
    'Odds 1': 'Home_Odds',
    'Odds X': 'Draw_Odds',
    'Odds 2': 'Away_Odds'
})

# Reorder columns
data = data[['Date', 'Home_Team', 'Away_Team', 'Home_Odds', 'Draw_Odds', 'Away_Odds']]

# Remove duplicate rows from DataFrame based on key columns
data = data.drop_duplicates(subset=['Date', 'Home_Team', 'Away_Team'])

# Drop the existing table and create a fresh one
with engine.connect() as connection:
    connection.execute(text("DROP TABLE IF EXISTS nhl24_odds;"))
    connection.execute(text('''
        CREATE TABLE nhl24_odds (
            Date DATE,
            Home_Team VARCHAR(100),
            Away_Team VARCHAR(100),
            Home_Odds FLOAT,
            Draw_Odds FLOAT,
            Away_Odds FLOAT
        );
    '''))

# Insert the deduplicated data into the fresh table
data.to_sql('nhl24_odds', engine, if_exists='replace', index=False)
print("Fresh table 'nhl24_odds' created and deduplicated data inserted successfully.")