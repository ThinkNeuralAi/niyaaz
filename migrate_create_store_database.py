#!/usr/bin/env python3
"""
Migration script to create store database tables and populate them from JSON configuration files
"""

import json
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def migrate_stores_to_database(db_manager):
    """Migrate store data from stores.json to database"""
    try:
        stores_data = load_json_file('config/stores.json')
        if not stores_data:
            logger.warning("No stores.json found or invalid JSON")
            return 0
        
        # Handle wrapper object
        if isinstance(stores_data, dict) and 'stores' in stores_data:
            stores_data = stores_data['stores']
        
        if not isinstance(stores_data, list):
            logger.warning("stores.json should contain a list of stores")
            return 0
        
        count = 0
        for store in stores_data:
            store_id = store.get('store_id')
            if not store_id:
                logger.warning("Store entry missing store_id")
                continue
            
            # Check if store already exists
            existing = db_manager.get_store(store_id)
            if existing:
                logger.info(f"Store {store_id} already exists in database, skipping")
                continue
            
            # Add store to database
            result = db_manager.add_store(
                store_id=store_id,
                name=store.get('store_name') or store.get('name', store_id),
                location=store.get('location', ''),
                description=store.get('description'),
                is_active=store.get('enabled', True),
                is_default=store.get('default', False),
                excluded_modules=store.get('excluded_modules', [])
            )
            
            if result:
                logger.info(f"✓ Migrated store: {store_id} - {store.get('store_name') or store.get('name')}")
                count += 1
            else:
                logger.error(f"✗ Failed to migrate store: {store_id}")
        
        return count
    
    except Exception as e:
        logger.error(f"Error migrating stores: {e}")
        return 0

def migrate_channels_to_database(db_manager):
    """Migrate RTSP channel data from channels.json to database"""
    try:
        channels_data = load_json_file('config/channels.json')
        if not channels_data:
            logger.warning("No channels.json found or invalid JSON")
            return 0, 0
        
        # Handle wrapper object
        if isinstance(channels_data, dict) and 'channels' in channels_data:
            channels_data = channels_data['channels']
        
        if not isinstance(channels_data, list):
            logger.warning("channels.json should contain a list of channels")
            return 0, 0
        
        rtsp_count = 0
        module_count = 0
        
        for channel in channels_data:
            channel_id = channel.get('camera_name') or channel.get('channel_id')
            if not channel_id:
                logger.warning("Channel entry missing channel_id or camera_name")
                continue
            
            store_id = channel.get('store_id', 'store_1')
            
            # Check if RTSP link already exists
            existing_link = db_manager.get_rtsp_link(channel_id)
            if existing_link:
                logger.info(f"RTSP link {channel_id} already exists in database, skipping")
            else:
                # Add RTSP link
                rtsp_result = db_manager.add_rtsp_link(
                    channel_id=channel_id,
                    store_id=store_id,
                    channel_name=channel.get('channel_name', channel_id),
                    rtsp_url=channel.get('rtsp_url', ''),
                    description=channel.get('description'),
                    is_active=channel.get('enabled', True),
                    resolution=channel.get('resolution'),
                    fps=channel.get('fps'),
                    codec=channel.get('codec')
                )
                
                if rtsp_result:
                    logger.info(f"✓ Migrated RTSP link: {channel_id} ({store_id})")
                    rtsp_count += 1
                else:
                    logger.error(f"✗ Failed to migrate RTSP link: {channel_id}")
            
            # Add module configurations
            modules = channel.get('modules', [])
            if modules:
                for module in modules:
                    module_name = module.get('name') or module.get('type')
                    if not module_name:
                        logger.warning(f"Module in channel {channel_id} missing name")
                        continue
                    
                    # Check if module config already exists
                    existing_module = db_manager.get_channel_module(channel_id, module_name)
                    if existing_module:
                        logger.info(f"Module {module_name} already configured for {channel_id}, skipping")
                        continue
                    
                    # Add module configuration
                    module_result = db_manager.add_channel_module(
                        channel_id=channel_id,
                        store_id=store_id,
                        module_name=module_name,
                        module_type=module.get('type', module_name),
                        enabled=module.get('enabled', True),
                        config_data=module.get('config'),  # config_data inherits entire config object
                        is_default=module.get('is_default', False)
                    )
                    
                    if module_result:
                        logger.info(f"  ✓ Configured module: {module_name}")
                        module_count += 1
                    else:
                        logger.warning(f"  ✗ Failed to configure module: {module_name}")
        
        return rtsp_count, module_count
    
    except Exception as e:
        logger.error(f"Error migrating channels: {e}")
        return 0, 0

def main():
    """Main migration function"""
    try:
        # Import Flask and database manager
        from app import app, db, db_manager
        
        logger.info("=" * 60)
        logger.info("Starting migration: JSON config to database")
        logger.info("=" * 60)
        
        with app.app_context():
            
            # Create all tables
            logger.info("Creating database tables...")
            db.create_all()
            logger.info("✓ Database tables created")
            
            # Migrate stores
            logger.info("\nMigrating stores data...")
            stores_migrated = migrate_stores_to_database(db_manager)
            logger.info(f"✓ Migrated {stores_migrated} store(s)")
            
            # Migrate channels
            logger.info("\nMigrating RTSP channels and modules...")
            rtsp_migrated, modules_migrated = migrate_channels_to_database(db_manager)
            logger.info(f"✓ Migrated {rtsp_migrated} RTSP link(s) and {modules_migrated} module configuration(s)")
            
            logger.info("\n" + "=" * 60)
            logger.info("Migration completed successfully!")
            logger.info(f"Summary:")
            logger.info(f"  - Stores migrated: {stores_migrated}")
            logger.info(f"  - RTSP links migrated: {rtsp_migrated}")
            logger.info(f"  - Module configs migrated: {modules_migrated}")
            logger.info("=" * 60)
    
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Make sure you're running this from the project root directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
