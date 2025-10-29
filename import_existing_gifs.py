"""
Import existing alert GIF files into database
"""
import os
import glob
from datetime import datetime
from app import app, db_manager

def import_existing_gifs():
    """Import all existing alert GIF files into the database"""
    alerts_dir = "static/alerts"
    
    if not os.path.exists(alerts_dir):
        print(f"Alerts directory '{alerts_dir}' does not exist")
        return
    
    gif_files = glob.glob(os.path.join(alerts_dir, "alert_*.gif"))
    
    if not gif_files:
        print("No alert GIF files found")
        return
    
    print(f"Found {len(gif_files)} alert GIF files")
    
    with app.app_context():
        imported_count = 0
        
        for gif_path in gif_files:
            try:
                gif_filename = os.path.basename(gif_path)
                file_size = os.path.getsize(gif_path)
                file_ctime = os.path.getctime(gif_path)
                
                # Extract channel_id from filename if possible
                # Format: alert_20251028_095101.gif or queue_alert_MyG_Queue_monitor_20251028_095101.gif
                channel_id = "unknown"
                if "queue_alert_" in gif_filename:
                    parts = gif_filename.replace("queue_alert_", "").split("_")
                    if len(parts) > 2:
                        # Get everything except the last 2 parts (date and time)
                        channel_id = "_".join(parts[:-2])
                
                gif_info = {
                    'gif_path': gif_path,
                    'gif_filename': gif_filename,
                    'frame_count': 0,
                    'duration': 4.0
                }
                
                alert_data = {
                    'queue_count': 0,
                    'counter_count': 0,
                    'alert_time': datetime.fromtimestamp(file_ctime).isoformat(),
                    'channel_id': channel_id,
                    'imported': True
                }
                
                # Check if already in database
                existing = db_manager.get_alert_gifs(channel_id=None, alert_type='queue_alert', limit=1000)
                if any(g['gif_filename'] == gif_filename for g in existing):
                    print(f"  Skipping {gif_filename} - already in database")
                    continue
                
                gif_id = db_manager.save_alert_gif(
                    channel_id=channel_id,
                    alert_type='queue_alert',
                    gif_info=gif_info,
                    alert_message=f"Imported alert from {datetime.fromtimestamp(file_ctime).strftime('%Y-%m-%d %H:%M:%S')}",
                    alert_data=alert_data
                )
                
                print(f"  ✓ Imported {gif_filename} (ID: {gif_id}, Size: {file_size/1024/1024:.1f}MB)")
                imported_count += 1
                
            except Exception as e:
                print(f"  ✗ Error importing {gif_filename}: {e}")
        
        print(f"\nImported {imported_count} alert GIFs into database")

if __name__ == "__main__":
    import_existing_gifs()
