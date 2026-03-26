#!/usr/bin/env python3
"""
Script to update store database with the provided data
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up Flask app context
from app import app, db_manager

def update_stores():
    """Update stores with the provided data"""
    with app.app_context():
        try:
            stores_to_update = [
                {
                    'store_id': 'store_1',
                    'name': 'Niyaaz_Restaurant_Belagavi',
                    'location': 'Belagavi, Karnataka',
                    'description': 'Primary location with all monitoring modules',
                    'is_active': True,
                    'is_default': True,
                    'excluded_modules': ['IdleTimeMonitor']
                },
                {
                    'store_id': 'store_2',
                    'name': 'Niyaaz_Restaurant_Goa',
                    'location': 'Bambolim, Goa',
                    'description': 'Secondary location with different modules from main store',
                    'is_active': True,
                    'is_default': False,
                    'excluded_modules': ['CashDetection', 'PersonSmokingDetection', 'MaterialTheftMonitor', 'CrowdDetection']
                },
                {
                    'store_id': 'store_3',
                    'name': 'Niyaaz_Restaurant_Tilakwadi',
                    'location': 'Tilakwadi, Karnataka',
                    'description': 'Niyaaz restaurant in Tilakwadi',
                    'is_active': True,
                    'is_default': False,
                    'excluded_modules': ['CashDetection', 'PersonSmokingDetection', 'MaterialTheftMonitor', 'CrowdDetection', 'IdleTimeMonitor']
                }
            ]
            
            logger.info("=" * 80)
            logger.info("UPDATING STORES DATABASE")
            logger.info("=" * 80)
            
            for store_data in stores_to_update:
                store_id = store_data['store_id']
                logger.info(f"\n📝 Updating {store_id}...")
                
                # Check if store exists
                existing_store = db_manager.get_store(store_id)
                if not existing_store:
                    logger.warning(f"  ⚠️  Store {store_id} not found in database")
                    # Try to add it instead
                    logger.info(f"  ℹ️  Adding new store {store_id}...")
                    result = db_manager.add_store(
                        store_id=store_id,
                        name=store_data['name'],
                        location=store_data['location'],
                        description=store_data['description'],
                        is_active=store_data['is_active'],
                        is_default=store_data['is_default'],
                        excluded_modules=store_data['excluded_modules']
                    )
                    if result:
                        logger.info(f"  ✅ Store {store_id} added successfully")
                    else:
                        logger.error(f"  ❌ Failed to add store {store_id}")
                else:
                    # Update existing store
                    result = db_manager.update_store(
                        store_id=store_id,
                        name=store_data['name'],
                        location=store_data['location'],
                        description=store_data['description'],
                        is_active=store_data['is_active'],
                        is_default=store_data['is_default'],
                        excluded_modules=store_data['excluded_modules']
                    )
                    if result:
                        logger.info(f"  ✅ Store updated successfully")
                        logger.info(f"     Name: {store_data['name']}")
                        logger.info(f"     Location: {store_data['location']}")
                        logger.info(f"     Active: {store_data['is_active']}")
                        logger.info(f"     Default: {store_data['is_default']}")
                        logger.info(f"     Excluded modules: {store_data['excluded_modules']}")
                    else:
                        logger.error(f"  ❌ Failed to update store {store_id}")
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ Store database update completed!")
            logger.info("=" * 80)
            
            # Display all stores
            logger.info("\n📊 Current Stores Database:")
            all_stores = db_manager.get_all_stores()
            for store in all_stores:
                logger.info(f"\n  Store ID: {store['store_id']}")
                logger.info(f"  Name: {store['name']}")
                logger.info(f"  Location: {store['location']}")
                logger.info(f"  Active: {store['is_active']}")
                logger.info(f"  Default: {store['is_default']}")
                logger.info(f"  Excluded Modules: {store['excluded_modules']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating stores: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = update_stores()
    sys.exit(0 if success else 1)
