#!/usr/bin/env python3
"""
Migration script to add order_wait_time and service_wait_time columns to table_service_violations table.

Run this script to add the missing columns to your database.
Usage: python migrate_add_wait_time_columns.py
"""

import os
import sys
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError

# Try to load from Flask app config
try:
    from app import app
    with app.app_context():
        database_url = app.config.get('SQLALCHEMY_DATABASE_URI')
        if database_url:
            print(f"Using database URL from Flask app config")
except:
    # Fallback to environment variable or default
    database_url = os.getenv('DATABASE_URL') or os.getenv('SQLALCHEMY_DATABASE_URI')
    if not database_url:
        print("❌ Error: Could not find database URL.")
        print("   Set DATABASE_URL environment variable or run from Flask app context")
        sys.exit(1)

def migrate():
    """Add order_wait_time and service_wait_time columns to table_service_violations"""
    print(f"Connecting to database: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    engine = create_engine(database_url)
    
    # Check database type
    is_postgresql = 'postgresql' in database_url.lower()
    is_sqlite = 'sqlite' in database_url.lower()
    
    with engine.connect() as conn:
        try:
            # Check if columns already exist
            inspector = inspect(engine)
            columns = [col['name'] for col in inspector.get_columns('table_service_violations')]
            
            existing_columns = []
            if 'order_wait_time' in columns:
                existing_columns.append('order_wait_time')
            if 'service_wait_time' in columns:
                existing_columns.append('service_wait_time')
            
            if 'order_wait_time' in existing_columns and 'service_wait_time' in existing_columns:
                print("✅ Columns order_wait_time and service_wait_time already exist. No migration needed.")
                return
            
            # Start transaction
            trans = conn.begin()
            
            try:
                # Add order_wait_time column if it doesn't exist
                if 'order_wait_time' not in existing_columns:
                    print("Adding order_wait_time column...")
                    if is_postgresql:
                        conn.execute(text("""
                            ALTER TABLE table_service_violations 
                            ADD COLUMN IF NOT EXISTS order_wait_time FLOAT
                        """))
                    elif is_sqlite:
                        # SQLite doesn't support IF NOT EXISTS in ALTER TABLE
                        # We'll try to add it and catch the error if it exists
                        try:
                            conn.execute(text("""
                                ALTER TABLE table_service_violations 
                                ADD COLUMN order_wait_time REAL
                            """))
                        except Exception as e:
                            if 'duplicate column' in str(e).lower() or 'already exists' in str(e).lower():
                                print("   Column already exists (skipping)")
                            else:
                                raise
                    else:
                        conn.execute(text("""
                            ALTER TABLE table_service_violations 
                            ADD COLUMN order_wait_time FLOAT
                        """))
                    print("✅ Added order_wait_time column")
                
                # Add service_wait_time column if it doesn't exist
                if 'service_wait_time' not in existing_columns:
                    print("Adding service_wait_time column...")
                    if is_postgresql:
                        conn.execute(text("""
                            ALTER TABLE table_service_violations 
                            ADD COLUMN IF NOT EXISTS service_wait_time FLOAT
                        """))
                    elif is_sqlite:
                        try:
                            conn.execute(text("""
                                ALTER TABLE table_service_violations 
                                ADD COLUMN service_wait_time REAL
                            """))
                        except Exception as e:
                            if 'duplicate column' in str(e).lower() or 'already exists' in str(e).lower():
                                print("   Column already exists (skipping)")
                            else:
                                raise
                    else:
                        conn.execute(text("""
                            ALTER TABLE table_service_violations 
                            ADD COLUMN service_wait_time FLOAT
                        """))
                    print("✅ Added service_wait_time column")
                
                # Commit transaction
                trans.commit()
                print("\n✅ Migration completed successfully!")
                print("   The dashboard should now display service discipline violations correctly.")
                
            except Exception as e:
                trans.rollback()
                print(f"❌ Error during migration: {e}")
                raise
                
        except OperationalError as e:
            print(f"❌ Database connection error: {e}")
            print(f"   Make sure the database is accessible")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == '__main__':
    print("=" * 60)
    print("Migration: Add order_wait_time and service_wait_time columns")
    print("=" * 60)
    print()
    migrate()
