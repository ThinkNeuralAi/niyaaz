"""
Add missing GIFs from filesystem to alert_gifs table
"""
import sqlite3
import os
from datetime import datetime
import re

db_path = "data/sakshi.db"
alerts_dir = "static/alerts"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("ADDING MISSING GIFS FROM FILESYSTEM TO DATABASE")
print("=" * 80)

# Get all GIFs in filesystem
if os.path.exists(alerts_dir):
    fs_files = set([f for f in os.listdir(alerts_dir) if f.endswith('.gif')])
    print(f"\nFound {len(fs_files)} GIF files in filesystem")
    
    # Get all GIFs already in database
    cursor.execute("SELECT gif_filename FROM alert_gifs WHERE gif_filename LIKE '%.gif'")
    db_files = set([row[0] for row in cursor.fetchall()])
    
    # Find missing GIFs
    missing_gifs = fs_files - db_files
    print(f"GIFs already in database: {len(db_files)}")
    print(f"Missing GIFs to add: {len(missing_gifs)}")
    
    if missing_gifs:
        print("\nAdding missing GIFs to database...")
        added_count = 0
        
        for gif_filename in sorted(missing_gifs):
            try:
                gif_path = os.path.join(alerts_dir, gif_filename)
                file_size = os.path.getsize(gif_path)
                
                # Parse timestamp from filename
                match = re.search(r'alert_(\d{8})_(\d{6})', gif_filename)
                if match:
                    date_str = match.group(1)
                    time_str = match.group(2)
                    created_at = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                else:
                    created_at = datetime.now()
                
                # Determine channel based on pattern or use generic
                channel_id = "generic_camera"
                
                # Insert into alert_gifs table
                cursor.execute("""
                    INSERT INTO alert_gifs (
                        channel_id, alert_type, gif_filename, gif_path, 
                        alert_message, frame_count, file_size, duration_seconds, 
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    channel_id,
                    'generic_alert',
                    gif_filename,
                    gif_path,
                    f'GIF recording: {gif_filename}',
                    0,  # frame count unknown
                    file_size,
                    0.0,  # duration unknown
                    created_at.isoformat()
                ))
                
                added_count += 1
                if added_count % 10 == 0:
                    print(f"  Added {added_count} GIFs...")
                    
            except Exception as e:
                print(f"  ⚠️ Error adding {gif_filename}: {e}")
        
        conn.commit()
        print(f"\n✅ Successfully added {added_count} GIFs to database")
    
    # Verify the update
    cursor.execute("SELECT COUNT(*) FROM alert_gifs WHERE gif_path LIKE '%.gif'")
    final_count = cursor.fetchone()[0]
    print(f"\nFinal count of alert_gifs with GIF files: {final_count}")
    
else:
    print(f"Error: Alerts directory not found: {alerts_dir}")

conn.close()
print("\n" + "=" * 80)
