
import pandas as pd

# Function to standardize column names based on the provided mapping
def standardize_columns(df, table_name, table_column_mapping):
    """
    Renames the columns of a DataFrame based on the table's name and the column mapping dictionary.

    Parameters:
        df (pd.DataFrame): The DataFrame to rename columns for.
        table_name (str): The name of the table (used as a key in the column mapping dictionary).
        table_column_mapping (dict): A dictionary containing the mapping of table names to column lists.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns if applicable.
    """
    if table_name in table_column_mapping:
        mapping = table_column_mapping[table_name]
        # Create a dictionary mapping existing columns to their standardized versions (if they exist in the mapping)
        rename_dict = {col: col for col in df.columns if col in mapping}
        return df.rename(columns=rename_dict)
    else:
        return df

# Function to apply column standardization to all relevant tables
def standardize_all_tables(nhl23_goalie_stats, nhl24_goalie_stats, nhl23_skater_stats, nhl24_skater_stats, column_mapping_csv):
    """
    Applies column standardization to all relevant tables based on the provided mappings.

    Parameters:
        nhl23_goalie_stats (pd.DataFrame): DataFrame for 2023 goalie stats.
        nhl24_goalie_stats (pd.DataFrame): DataFrame for 2024 goalie stats.
        nhl23_skater_stats (pd.DataFrame): DataFrame for 2023 skater stats.
        nhl24_skater_stats (pd.DataFrame): DataFrame for 2024 skater stats.
        column_mapping_csv (str): Path to the CSV file containing column mappings.

    Returns:
        dict: A dictionary of standardized DataFrames.
    """
    # Load the CSV file containing the column standardization mapping
    column_mapping_df = pd.read_csv(column_mapping_csv)
    
    # Extract unique tables and their corresponding columns from the CSV for easier lookups
    table_column_mapping = column_mapping_df.groupby('table_name')['column_name'].apply(list).to_dict()

    # Standardize each table
    standardized_tables = {
        'nhl23_goalie_stats': standardize_columns(nhl23_goalie_stats, 'nhl23_goalie_stats', table_column_mapping),
        'nhl24_goalie_stats': standardize_columns(nhl24_goalie_stats, 'nhl24_goalie_stats', table_column_mapping),
        'nhl23_skater_stats': standardize_columns(nhl23_skater_stats, 'nhl23_skater_stats', table_column_mapping),
        'nhl24_skater_stats': standardize_columns(nhl24_skater_stats, 'nhl24_skater_stats', table_column_mapping)
    }
    
    return standardized_tables

