#!/usr/bin/env python
"""Update camera RTSP URLs in database from channels.json"""
import json
import sys
sys.path.insert(0, '.')

from app import app, db_manager
from pathlib import Path

# Load channels from JSON
config_path = Path('config/channels.json')
with open(config_path, 'r') as f:
    config = json.load(f)

channels = config.get('channels', [])
print(f"[UPDATE] Loaded {len(channels)} channels from config/channels.json")

# Use app context for database operations
with app.app_context():
    try:
        # Update each channel in database
        updated_count = 0
        for channel in channels:
            if not channel.get('enabled'):
                continue
            
            channel_id = channel.get('channel_id')
            channel_name = channel.get('channel_name')
            video_source = channel.get('rtsp_url') or channel.get('video_file')
            store_id = channel.get('store_id', 'store_1')
            description = channel.get('description', '')
            
            # Save to database (will update if exists)
            db_manager.save_rtsp_channel(
                channel_id=channel_id,
                name=channel_name,
                rtsp_url=video_source,
                description=description
            )
            updated_count += 1
            print(f'✅ [{updated_count}] Updated {channel_id}: {channel_name}')
        
        print(f"\n[SUCCESS] Updated {updated_count} cameras in database")
        
        # Show cameras 16-27 (store_2)
        store2_cameras = [ch for ch in channels if ch.get('store_id') == 'store_2' and ch.get('enabled')]
        print(f"\n[STORE_2] Found {len(store2_cameras)} cameras:")
        for cam in store2_cameras:
            print(f'   ✓ {cam["channel_id"]}: {cam["channel_name"]}')
            print(f'     RTSP: {cam.get("rtsp_url", "N/A")[:50]}...')

    except Exception as e:
        print(f"❌ Error updating database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

