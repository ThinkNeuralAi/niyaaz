"""
Update RTSP links in database from channels.json
"""
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app, db_manager

def update_rtsp_links_from_config():
    """Update RTSP links table with channels from channels.json"""
    
    # Load channels configuration
    config_path = 'config/channels.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        channels = config.get('channels', [])
        print(f"✓ Loaded {len(channels)} channels from {config_path}")
        
        # Filter store2 channels
        store2_channels = [ch for ch in channels if ch.get('store_id') == 'store_2']
        print(f"✓ Found {len(store2_channels)} store_2 channels")
        
        # Update or add RTSP links for store2 channels
        updated_count = 0
        added_count = 0
        
        with app.app_context():
            for channel in store2_channels:
                channel_id = channel.get('channel_id')
                channel_name = channel.get('channel_name')
                store_id = channel.get('store_id')
                rtsp_url = channel.get('rtsp_url')
                enabled = channel.get('enabled', True)
                
                if not channel_id or not rtsp_url:
                    print(f"⚠ Skipping channel with missing ID or RTSP URL: {channel}")
                    continue
                
                # Check if channel already exists
                existing = db_manager.RTSPLink.query.filter_by(channel_id=channel_id).first()
                
                if existing:
                    # Update existing entry
                    existing.rtsp_url = rtsp_url
                    existing.channel_name = channel_name
                    existing.is_active = enabled
                    db_manager.db.session.commit()
                    updated_count += 1
                    print(f"✓ Updated: {channel_id} ({channel_name}) - {rtsp_url}")
                else:
                    # Add new entry
                    db_manager.add_rtsp_link(
                        channel_id=channel_id,
                        store_id=store_id,
                        channel_name=channel_name,
                        rtsp_url=rtsp_url,
                        description=f"Channel for {channel_name}",
                        is_active=enabled
                    )
                    added_count += 1
                    print(f"✓ Added: {channel_id} ({channel_name}) - {rtsp_url}")
        
        print(f"\n📊 Summary:")
        print(f"   Total store_2 channels: {len(store2_channels)}")
        print(f"   Updated: {updated_count}")
        print(f"   Added: {added_count}")
        print(f"   ✓ RTSP links table successfully updated!")
        
    except FileNotFoundError:
        print(f"❌ Error: {config_path} not found")
        return False
    except json.JSONDecodeError:
        print(f"❌ Error: {config_path} is not valid JSON")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = update_rtsp_links_from_config()
    sys.exit(0 if success else 1)
