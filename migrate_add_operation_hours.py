"""
Migration: Add operation_start_time and operation_end_time columns to stores table.

These columns control when violations/alerts are triggered for each store.
If not set (NULL), the store is considered always active.

Usage:
    python migrate_add_operation_hours.py
"""
import os
import sys
from urllib.parse import quote_plus

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def get_db_connection():
    """Get database connection using project config"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'sakshiai')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')

    if not db_password:
        try:
            import json
            config_path = os.path.join(os.path.dirname(__file__), 'config', 'default.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            db_config = config.get('database', {})
            db_host = db_config.get('host', db_host)
            db_port = str(db_config.get('port', db_port))
            db_name = db_config.get('name', db_name)
            db_user = db_config.get('username', db_user)
            db_password = db_config.get('password', '')
        except Exception:
            pass

    if db_password:
        import psycopg2
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        return conn, 'postgresql'
    else:
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'sakshi.db')
        conn = sqlite3.connect(db_path)
        return conn, 'sqlite'


def migrate():
    conn, db_type = get_db_connection()
    cursor = conn.cursor()

    print(f"Connected to {db_type} database")

    # Check if columns already exist
    if db_type == 'postgresql':
        cursor.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'stores' AND column_name IN ('operation_start_time', 'operation_end_time')
        """)
        existing = [row[0] for row in cursor.fetchall()]
    else:
        cursor.execute("PRAGMA table_info(stores)")
        columns = [row[1] for row in cursor.fetchall()]
        existing = [c for c in columns if c in ('operation_start_time', 'operation_end_time')]

    if 'operation_start_time' not in existing:
        print("Adding column: operation_start_time")
        cursor.execute("ALTER TABLE stores ADD COLUMN operation_start_time VARCHAR(5)")
    else:
        print("Column operation_start_time already exists, skipping")

    if 'operation_end_time' not in existing:
        print("Adding column: operation_end_time")
        cursor.execute("ALTER TABLE stores ADD COLUMN operation_end_time VARCHAR(5)")
    else:
        print("Column operation_end_time already exists, skipping")

    conn.commit()
    cursor.close()
    conn.close()
    print("Migration complete!")


if __name__ == '__main__':
    migrate()
