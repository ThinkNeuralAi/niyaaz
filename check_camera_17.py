#!/usr/bin/env python3
"""
Diagnostic script to check camera_17 configuration and status
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up Flask app context
from app import app, db_manager, shared_video_processors, channel_modules

def check_camera_17():
    """Check camera_17 configuration and status"""
    with app.app_context():
        try:
            print("\n" + "=" * 80)
            print("CAMERA_17 DIAGNOSTIC CHECK")
            print("=" * 80)
            
            # 1. Check database configuration
            print("\n📋 DATABASE CONFIGURATION:")
            print("-" * 80)
            all_links = db_manager.get_all_rtsp_links()
            camera_17_db = None
            for link in all_links:
                if link['channel_id'] == 'camera_17':
                    camera_17_db = link
                    break
            
            if camera_17_db:
                print(f"✅ Found in database:")
                print(f"   Channel ID: {camera_17_db['channel_id']}")
                print(f"   Channel Name: {camera_17_db.get('channel_name', 'N/A')}")
                print(f"   Store ID: {camera_17_db.get('store_id', 'N/A')}")
                print(f"   RTSP URL: {camera_17_db.get('rtsp_url', 'N/A')}")
                print(f"   Is Active: {camera_17_db.get('is_active', False)}")
                print(f"   Enabled: {camera_17_db.get('enabled', False)}")
            else:
                print("❌ NOT found in database")
            
            # 2. Check if processor is running
            print("\n🎥 PROCESSOR STATUS:")
            print("-" * 80)
            if 'camera_17' in shared_video_processors:
                processor = shared_video_processors['camera_17']
                print(f"✅ Processor exists")
                print(f"   Is Running: {processor.is_running}")
                print(f"   Active Modules: {processor.get_active_modules()}")
            else:
                print("❌ No processor found for camera_17")
            
            # 3. Check channel modules
            print("\n📦 CHANNEL MODULES:")
            print("-" * 80)
            if 'camera_17' in channel_modules:
                modules = channel_modules['camera_17']
                print(f"✅ Modules configured: {list(modules.keys())}")
                
                if 'QueueMonitor' in modules:
                    queue_module = modules['QueueMonitor']
                    print(f"\n   QueueMonitor details:")
                    if hasattr(queue_module, 'get_status'):
                        status = queue_module.get_status()
                        print(f"      Status: {status}")
                    else:
                        print(f"      Module type: {type(queue_module)}")
            else:
                print("❌ No modules configured for camera_17")
            
            # 4. Check store mapping
            print("\n🏪 STORE MAPPING:")
            print("-" * 80)
            from app import get_channel_to_store_mapping
            channel_store_map = get_channel_to_store_mapping()
            if 'camera_17' in channel_store_map:
                print(f"✅ Mapped to: {channel_store_map['camera_17']}")
            else:
                print("❌ Not in store mapping")
            
            # 5. Check all store_2 channels with QueueMonitor
            print("\n🔍 ALL STORE_2 CHANNELS WITH QUEUEMONITOR:")
            print("-" * 80)
            store_2_queue_channels = []
            for link in all_links:
                if link.get('store_id') == 'store_2' and link.get('is_active'):
                    # Check if has QueueMonitor in config
                    modules = link.get('modules', [])
                    if modules:
                        import json
                        if isinstance(modules, str):
                            try:
                                modules = json.loads(modules)
                            except:
                                modules = []
                        
                        has_queue = any(m.get('type') == 'QueueMonitor' for m in modules if isinstance(m, dict))
                        if has_queue:
                            store_2_queue_channels.append(link['channel_id'])
                            status_icon = "✅" if link['channel_id'] in shared_video_processors else "❌"
                            print(f"   {status_icon} {link['channel_id']} ({link.get('channel_name', 'N/A')})")
            
            if not store_2_queue_channels:
                print("   ⚠️ No store_2 channels with QueueMonitor found")
            
            print("\n" + "=" * 80)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    check_camera_17()
