"""
Create PostgreSQL database and prepare for migration
"""
import os
import re
import psycopg2

# Load .env file manually
def load_env_file():
    """Load environment variables from .env file"""
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file
load_env_file()

# Get connection details
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'sakshiai')
db_user = os.getenv('DB_USER', 'postgres')
db_password = os.getenv('DB_PASSWORD', '')

print("=" * 60)
print("PostgreSQL Database Setup")
print("=" * 60)
print(f"\nConnecting to PostgreSQL...")
print(f"  Host: {db_host}:{db_port}")
print(f"  User: {db_user}")
print(f"  Target Database: {db_name}")
print()

if not db_password:
    print("❌ Error: DB_PASSWORD is not set in .env file")
    exit(1)

try:
    # Connect to default 'postgres' database
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        database='postgres',
        user=db_user,
        password=db_password
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (db_name,)
    )
    db_exists = cursor.fetchone()
    
    if db_exists:
        print(f"✅ Database '{db_name}' already exists")
    else:
        print(f"📦 Creating database '{db_name}'...")
        cursor.execute(f'CREATE DATABASE "{db_name}"')
        print(f"✅ Database '{db_name}' created successfully")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Database setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Create tables: python app.py (then stop it)")
    print("  2. Migrate data: python migrate_to_postgresql.py")
    print("  3. Start app: python app.py")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)



