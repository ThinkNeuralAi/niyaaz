#!/usr/bin/env python3
"""
Script to add IdleTimeMonitor to store_1's excluded_modules list
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up Flask app context
from app import app, db_manager

def update_store_excluded_modules():
    """Add IdleTimeMonitor to store_1's excluded modules"""
    with app.app_context():
        try:
            # Get current store_1 configuration
            stores = db_manager.get_all_stores()
            store_1 = None
            
            for store in stores:
                if store['store_id'] == 'store_1':
                    store_1 = store
                    break
            
            if not store_1:
                print("❌ Store_1 not found in database!")
                return False
            
            print(f"📦 Current store_1 configuration:")
            print(f"   Name: {store_1['name']}")
            print(f"   Current excluded_modules: {store_1['excluded_modules']}")
            
            # Get current excluded modules
            excluded_modules = store_1.get('excluded_modules', [])
            if not isinstance(excluded_modules, list):
                excluded_modules = []
            
            # Add IdleTimeMonitor if not already excluded
            if 'IdleTimeMonitor' not in excluded_modules:
                excluded_modules.append('IdleTimeMonitor')
                print(f"\n✅ Adding 'IdleTimeMonitor' to excluded modules")
                
                # Update the store
                success = db_manager.update_store(
                    'store_1',
                    excluded_modules=json.dumps(excluded_modules)
                )
                
                if success:
                    print(f"✅ Successfully updated store_1")
                    print(f"   New excluded_modules: {excluded_modules}")
                    return True
                else:
                    print("❌ Failed to update store_1")
                    return False
            else:
                print(f"\n✅ 'IdleTimeMonitor' is already in excluded modules for store_1")
                return True
                
        except Exception as e:
            print(f"❌ Error updating store: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    print("=" * 60)
    print("Updating Store 1 Configuration")
    print("=" * 60)
    success = update_store_excluded_modules()
    print("=" * 60)
    if success:
        print("✅ Update completed successfully!")
        print("\nℹ️  Please restart the application for changes to take effect.")
    else:
        print("❌ Update failed!")
    print("=" * 60)
