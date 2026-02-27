#!/usr/bin/env python3
"""
Script to enable camera_17 in the database
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up Flask app context
from app import app, db_manager

def enable_camera_17():
    """Enable camera_17 in the database"""
    with app.app_context():
        try:
            print("\n" + "=" * 80)
            print("ENABLING CAMERA_17")
            print("=" * 80)
            
            # Get current configuration
            all_links = db_manager.get_all_rtsp_links()
            camera_17 = None
            
            for link in all_links:
                if link['channel_id'] == 'camera_17':
                    camera_17 = link
                    break
            
            if not camera_17:
                print("❌ Camera_17 not found in database!")
                return False
            
            print(f"\n📋 Current camera_17 configuration:")
            print(f"   Channel Name: {camera_17.get('channel_name', 'N/A')}")
            print(f"   Store ID: {camera_17.get('store_id', 'N/A')}")
            print(f"   Is Active: {camera_17.get('is_active', False)}")
            print(f"   Enabled: {camera_17.get('enabled', False)}")
            
            # Update to enabled=True
            if not camera_17.get('enabled', False):
                print(f"\n✅ Enabling camera_17...")
                
                # Update the channel
                success = db_manager.update_rtsp_link(
                    'camera_17',
                    enabled=True,
                    is_active=True
                )
                
                if success:
                    print(f"✅ Successfully enabled camera_17")
                    print(f"\nℹ️  Please restart the application for changes to take effect.")
                    print(f"   The camera will start processing and appear in the dashboard.")
                    return True
                else:
                    print("❌ Failed to enable camera_17")
                    return False
            else:
                print(f"\n✅ Camera_17 is already enabled")
                return True
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = enable_camera_17()
    print("=" * 80)
    if success:
        print("✅ Operation completed successfully!")
    else:
        print("❌ Operation failed!")
    print("=" * 80 + "\n")
