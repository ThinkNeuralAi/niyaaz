#!/usr/bin/env python3
"""
Migration Script: Add Operation Timing to Stores
================================================

This script adds operation timing (opening and closing hours) to stores in the database.
Operation timing controls when violations/alerts are triggered - they will only be recorded
during the store's operating hours.

Usage:
    python migrate_operation_timing.py

Configuration:
    Edit the STORE_OPERATION_TIMES dictionary below to set the operation hours for each store.
    Format: "HH:MM" (24-hour format)
    
Example:
    STORE_OPERATION_TIMES = {
        "store_1": {"start": "07:00", "end": "23:59"},  # 7 AM to 11:59 PM
        "store_2": {"start": "11:00", "end": "23:59"},  # 11 AM to 11:59 PM
        "store_3": {"start": "11:00", "end": "23:00"},  # 11 AM to 11 PM
    }
"""

import os
import sys
import logging
from datetime import datetime
from urllib.parse import quote_plus

from sqlalchemy import create_engine, inspect, text

# Add current directory to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define operation times for each store (edit these as needed)
STORE_OPERATION_TIMES = {
    "store_1": {"start": "07:00", "end": "23:59"},  # 7 AM to 11:59 PM
    "store_2": {"start": "11:00", "end": "23:59"},  # 11 AM to 11:59 PM
    "store_3": {"start": "11:00", "end": "23:00"},  # 11 AM to 11 PM (12 hours)
}


def get_database_url():
    """Build the database URL without importing the full Flask app."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    base_dir = os.path.dirname(os.path.abspath(__file__))

    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'sakshiai')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', '')

    if not db_password and os.getenv('USE_POSTGRESQL', '').lower() == 'true':
        try:
            import json
            with open(os.path.join(base_dir, 'config', 'default.json'), 'r', encoding='utf-8') as handle:
                config = json.load(handle)
            db_config = config.get('database', {})
            db_host = db_config.get('host', db_host)
            db_port = str(db_config.get('port', db_port))
            db_name = db_config.get('name', db_name)
            db_user = db_config.get('username', db_user)
            db_password = db_config.get('password', db_password)
        except Exception as exc:
            logger.warning(f"Could not load database config from file: {exc}")

    if db_password or os.getenv('USE_POSTGRESQL', '').lower() == 'true':
        if db_password:
            encoded_password = quote_plus(db_password)
            return f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

    db_path = os.path.join(base_dir, 'data', 'sakshi.db')
    return f"sqlite:///{db_path}"


def ensure_operation_timing_columns(database_url):
    """Add operation timing columns to the stores table if they do not exist."""
    logger.info("Checking database schema for operation timing columns...")

    engine = create_engine(database_url)
    inspector = inspect(engine)

    if 'stores' not in inspector.get_table_names():
        logger.error("❌ stores table not found in the configured database")
        return False

    columns = {column['name'] for column in inspector.get_columns('stores')}
    missing_columns = []
    if 'operation_start_time' not in columns:
        missing_columns.append('operation_start_time')
    if 'operation_end_time' not in columns:
        missing_columns.append('operation_end_time')

    if not missing_columns:
        logger.info("✅ Operation timing columns already exist")
        return True

    logger.info(f"Adding missing columns to stores table: {', '.join(missing_columns)}")
    is_postgresql = 'postgresql' in database_url.lower()

    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            if 'operation_start_time' in missing_columns:
                if is_postgresql:
                    connection.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS operation_start_time VARCHAR(5)"))
                else:
                    connection.execute(text("ALTER TABLE stores ADD COLUMN operation_start_time VARCHAR(5)"))

            if 'operation_end_time' in missing_columns:
                if is_postgresql:
                    connection.execute(text("ALTER TABLE stores ADD COLUMN IF NOT EXISTS operation_end_time VARCHAR(5)"))
                else:
                    connection.execute(text("ALTER TABLE stores ADD COLUMN operation_end_time VARCHAR(5)"))

            transaction.commit()
            logger.info("✅ Database schema updated successfully")
            return True
        except Exception as exc:
            transaction.rollback()
            logger.error(f"❌ Failed to update stores schema: {exc}")
            return False


def validate_time_format(time_str):
    """Validate time format (HH:MM)"""
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False


def migrate():
    """Run the migration to add operation timing to stores"""
    database_url = get_database_url()
    logger.info(f"Using database connection: {database_url.split('@')[-1] if '@' in database_url else database_url}")

    if not ensure_operation_timing_columns(database_url):
        return False

    # Import app and database manager only after the schema is ready
    from app import app, db_manager
    
    logger.info("=" * 70)
    logger.info("Starting Operation Timing Migration")
    logger.info("=" * 70)
    
    # Validate all times before making changes
    logger.info("Validating operation times...")
    for store_id, times in STORE_OPERATION_TIMES.items():
        start = times.get("start")
        end = times.get("end")
        
        if not start or not validate_time_format(start):
            logger.error(f"❌ Invalid start time for {store_id}: {start}")
            return False
        
        if not end or not validate_time_format(end):
            logger.error(f"❌ Invalid end time for {store_id}: {end}")
            return False
        
        logger.info(f"✅ {store_id}: {start} - {end}")
    
    # Apply changes
    logger.info("\nApplying operation timing to stores...")
    
    with app.app_context():
        success_count = 0
        error_count = 0
        
        for store_id, times in STORE_OPERATION_TIMES.items():
            try:
                result = db_manager.update_store(
                    store_id,
                    operation_start_time=times["start"],
                    operation_end_time=times["end"]
                )
                
                if result:
                    logger.info(f"✅ Updated {store_id}: {times['start']} - {times['end']}")
                    success_count += 1
                else:
                    logger.error(f"❌ Failed to update {store_id}")
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error updating {store_id}: {e}")
                error_count += 1
        
        # Summary
        logger.info("\n" + "=" * 70)
        logger.info(f"Migration Complete: {success_count} succeeded, {error_count} failed")
        logger.info("=" * 70)
        
        # Display configured stores
        logger.info("\nConfigured Stores with Operation Timing:")
        all_stores = db_manager.get_all_stores()
        for store in all_stores:
            store_name = store.get('name', store.get('store_id'))
            start_time = store.get('operation_start_time', 'Not set')
            end_time = store.get('operation_end_time', 'Not set')
            
            logger.info(f"  • {store['store_id']} ({store_name}): {start_time} - {end_time}")
        
        return error_count == 0


if __name__ == '__main__':
    logger.info("Operation Timing Migration Script")
    logger.info("-" * 70)
    
    # Validate configuration
    if not STORE_OPERATION_TIMES:
        logger.error("❌ No store operation times configured!")
        sys.exit(1)
    
    # Run migration
    if migrate():
        logger.info("✅ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed")
        sys.exit(1)
