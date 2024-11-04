import psycopg2
from psycopg2.extras import execute_values
import sys
import time

# Database configurations
LOCAL_DB = {
    "host": "localhost",
    "port": "5432",
    "dbname": "Provibet_NHL",
    "user": "postgres",
    "password": "Provibet2024"
}

NEON_DB = {
    "host": "ep-raspy-leaf-a8qjqbzo.eastus2.azure.neon.tech",
    "dbname": "Provibet_NHL",
    "user": "neondb_owner",
    "password": "Whuy7GKo3UJO"
}


def get_sequence_info(cursor, table_name):
    """Get sequence information for a table if it exists"""
    cursor.execute("""
        SELECT pg_get_serial_sequence(%s, column_name) as sequence_name,
               column_name,
               column_default
        FROM information_schema.columns
        WHERE table_name = %s
        AND column_default LIKE 'nextval%%';
    """, (table_name, table_name))
    return cursor.fetchall()


def create_sequence(cursor, sequence_name):
    """Create a sequence if it doesn't exist"""
    cursor.execute(f"CREATE SEQUENCE IF NOT EXISTS {sequence_name};")


def get_table_schema(cursor, table_name):
    """Get the complete CREATE TABLE statement for a table"""
    # First get sequence information
    sequences = get_sequence_info(cursor, table_name)
    sequence_definitions = []

    for seq_name, col_name, _ in sequences:
        if seq_name:
            sequence_definitions.append(seq_name)

    # Get column definitions
    cursor.execute(r"""
        SELECT 
            column_name,
            data_type,
            character_maximum_length,
            numeric_precision,
            numeric_scale,
            is_nullable,
            column_default,
            CASE 
                WHEN column_default LIKE 'nextval%%' 
                THEN regexp_replace(column_default, 'nextval\(''(.*)''\:\:regclass\)', '\1')
                ELSE NULL 
            END as sequence_name
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
    """, (table_name,))

    columns = cursor.fetchall()
    column_definitions = []

    for col in columns:
        col_name, data_type, max_length, num_precision, num_scale, nullable, default, seq_name = col
        definition = f'"{col_name}" {data_type}'

        if data_type.lower() in ('character varying', 'varchar', 'char'):
            if max_length:
                definition += f'({max_length})'
        elif data_type.lower() in ('numeric', 'decimal'):
            if num_precision is not None:
                if num_scale is not None:
                    definition += f'({num_precision},{num_scale})'
                else:
                    definition += f'({num_precision})'

        # Special handling for the nhl24_schedule table
        if table_name == 'nhl24_schedule' and col_name == 'game_id':
            definition = '"game_id" SERIAL PRIMARY KEY'
        else:
            if default and not default.startswith('nextval'):
                definition += f" DEFAULT {default}"

            if nullable == 'NO':
                definition += " NOT NULL"

        column_definitions.append(definition)

    create_statement = f"CREATE TABLE {table_name} (\n    "
    create_statement += ",\n    ".join(column_definitions)
    create_statement += "\n);"

    return create_statement, sequence_definitions


def get_all_tables(cursor):
    """Get all user tables from the database"""
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
    """)
    return [table[0] for table in cursor.fetchall()]


def table_exists(cursor, table_name):
    """Check if a table exists in the database"""
    cursor.execute("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = %s
        );
    """, (table_name,))
    return cursor.fetchone()[0]


def sync_table(table_name, local_conn, neon_conn):
    print(f"\nSyncing table: {table_name}")

    try:
        local_cur = local_conn.cursor()
        neon_cur = neon_conn.cursor()

        # Check if table exists in Neon
        if not table_exists(neon_cur, table_name):
            print(f"Table {table_name} doesn't exist in Neon. Creating it...")

            # Get schema and sequences
            create_statement, sequences = get_table_schema(local_cur, table_name)

            # Create sequences first
            for sequence in sequences:
                print(f"Creating sequence: {sequence}")
                create_sequence(neon_cur, sequence)

            print(f"Creating table with schema:\n{create_statement}")
            neon_cur.execute(create_statement)
            neon_conn.commit()
            print(f"✓ Table {table_name} created successfully")

        # Get column names
        local_cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s
            ORDER BY ordinal_position;
        """, (table_name,))
        columns = [col[0] for col in local_cur.fetchall()]
        quoted_columns = [f'"{col}"' for col in columns]
        columns_str = ', '.join(quoted_columns)

        # Read data from local database
        select_query = f"SELECT {columns_str} FROM {table_name}"
        print(f"Reading data from local database...")
        local_cur.execute(select_query)
        data = local_cur.fetchall()

        if not data:
            print(f"No data found in local {table_name}")
            return

        # Clear existing data in Neon
        print(f"Clearing existing data from {table_name} in Neon...")
        neon_cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")

        # Insert data into Neon in batches
        total_rows = len(data)
        print(f"Inserting {total_rows} rows into {table_name}...")

        insert_query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES %s
        """

        # Use a smaller page size for large datasets
        page_size = 500
        for i in range(0, total_rows, page_size):
            batch = data[i:i + page_size]
            execute_values(neon_cur, insert_query, batch)
            print(f"Progress: {min(i + page_size, total_rows)}/{total_rows} rows")
            neon_conn.commit()

        print(f"✓ Successfully synced {table_name}")

    except Exception as e:
        print(f"Error syncing {table_name}: {str(e)}")
        if 'select_query' in locals():
            print(f"Query that caused error: {select_query}")
        neon_conn.rollback()
        raise
    finally:
        if 'local_cur' in locals():
            local_cur.close()
        if 'neon_cur' in locals():
            neon_cur.close()


def main():
    print("Starting database sync process...")
    start_time = time.time()

    try:
        # Connect to both databases
        print("Connecting to local database...")
        local_conn = psycopg2.connect(**LOCAL_DB)

        print("Connecting to Neon database...")
        neon_conn = psycopg2.connect(**NEON_DB)

        # Get all tables from local database
        local_cur = local_conn.cursor()
        tables = get_all_tables(local_cur)
        local_cur.close()

        print(f"\nFound {len(tables)} tables to sync:")
        for table in tables:
            print(f"- {table}")
        print("")

        # Sync each table
        successful_tables = []
        failed_tables = []

        for table in tables:
            try:
                sync_table(table, local_conn, neon_conn)
                successful_tables.append(table)
            except Exception as e:
                failed_tables.append((table, str(e)))
                print(f"Failed to sync {table}. Moving to next table...")
                continue

        # Print summary
        print("\nSync Summary:")
        print(f"✓ Successfully synced {len(successful_tables)} tables:")
        for table in successful_tables:
            print(f"  - {table}")

        if failed_tables:
            print(f"\n✗ Failed to sync {len(failed_tables)} tables:")
            for table, error in failed_tables:
                print(f"  - {table}: {error}")

    except Exception as e:
        print(f"\n✗ Sync failed: {str(e)}")
        sys.exit(1)

    finally:
        # Close connections
        if 'local_conn' in locals():
            local_conn.close()
        if 'neon_conn' in locals():
            neon_conn.close()

        elapsed_time = time.time() - start_time
        print(f"\nSync completed in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()