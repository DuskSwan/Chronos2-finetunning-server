import sqlite3

for db_file in ['data/finetune.db', 'data/finetune_jobs.db']:
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"{db_file}: Tables = {tables}")
        # Get row count for each table
        if tables:
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"  {table_name}: {count} rows")
        conn.close()
    except Exception as e:
        print(f"{db_file}: Error - {e}")
