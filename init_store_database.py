#!/usr/bin/env python3
"""
Initialize database by dropping and recreating all tables, then populating with config data
"""

import json
import logging
from app import app, db, db_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=" * 70)
        logger.info("DATABASE INITIALIZATION - Dropping and Recreating All Tables")
        logger.info("=" * 70)
        
        with app.app_context():
            # Drop all tables
            logger.info("\n⚠️  Dropping all existing tables...")
            db.drop_all()
            logger.info("✓ All tables dropped")
            
            # Create all tables
            logger.info("\n📦 Creating all new tables...")
            db.create_all()
            logger.info("✓ All tables created")
            
            # Migrate stores
            logger.info("\n🏪 Migrating stores...")
            try:
                with open('config/stores.json', 'r') as f:
                    stores_config = json.load(f)
                
                stores_list = stores_config.get('stores', [])
                for store in stores_list:
                    result = db_manager.add_store(
                        store_id=store.get('store_id'),
                        name=store.get('store_name') or store.get('name', store.get('store_id')),
                        location=store.get('location', ''),
                        description=store.get('description'),
                        is_active=store.get('enabled', True),
                        is_default=store.get('default', False),
                        excluded_modules=store.get('excluded_modules', [])
                    )
                    if result:
                        logger.info(f"  ✓ {store.get('store_id')} - {store.get('store_name')}")
                logger.info(f"✓ Migrated {len(stores_list)} store(s)")
            except Exception as e:
                logger.error(f"Error migrating stores: {e}")
            
            # Migrate channels and modules
            logger.info("\n🎥 Migrating RTSP channels and modules...")
            try:
                with open('config/channels.json', 'r') as f:
                    channels_config = json.load(f)
                
                channels_list = channels_config.get('channels', [])
                rtsp_count = 0
                module_count = 0
                
                for channel in channels_list:
                    channel_id = channel.get('channel_id')
                    store_id = channel.get('store_id', 'store_1')
                    
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
                        rtsp_count += 1
                    
                    # Add modules
                    modules = channel.get('modules', [])
                    for module in modules:
                        module_name = module.get('name') or module.get('type')
                        if module_name:
                            module_result = db_manager.add_channel_module(
                                channel_id=channel_id,
                                store_id=store_id,
                                module_name=module_name,
                                module_type=module.get('type', module_name),
                                enabled=module.get('enabled', True),
                                config_data=module.get('config'),
                                is_default=module.get('is_default', False)
                            )
                            if module_result:
                                module_count += 1
                
                logger.info(f"✓ Migrated {rtsp_count} RTSP channel(s) and {module_count} module(s)")
            
            except Exception as e:
                logger.error(f"Error migrating channels: {e}")
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ DATABASE INITIALIZATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info(f"Summary:")
            logger.info(f"  • All tables dropped and recreated")
            logger.info(f"  • Stores configured")
            logger.info(f"  • RTSP channels and module configurations migrated")
            logger.info("=" * 70)
    
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
