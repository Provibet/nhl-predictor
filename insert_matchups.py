import pandas as pd
from sqlalchemy import create_engine

# Load the CSV file from the local path
csv_file_path = r"D:\Downloads\SQL Data Import\matchups_with_situations.csv"  # Use the raw string format to avoid path issues
matchups_with_situations = pd.read_csv(csv_file_path)

# PostgreSQL connection details
host = "localhost"
port = "5432"
dbname = "Provibet_NHL"
user = "postgres"
password = "Provibet2024"

# Create connection string for SQLAlchemy
connection_string = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"

# Create a SQLAlchemy engine
engine = create_engine(connection_string)

# Define table name
table_name = "nhl_matchups_with_situations"

# Insert data into PostgreSQL
matchups_with_situations.to_sql(table_name, engine, if_exists='replace', index=False)

print(f"Data has been successfully inserted into the '{table_name}' table.")
