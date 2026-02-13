"""
Check alert GIFs in database vs filesystem
"""
import os
import sqlite3
from pathlib import Path

# Connect to database
db_path = "data/sakshi.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check alert_gifs table
print("=" * 80)
print("ALERT_GIFS TABLE ANALYSIS")
print("=" * 80)

cursor.execute("SELECT COUNT(*) FROM alert_gifs")
total_records = cursor.fetchone()[0]
print(f"\nTotal records in alert_gifs table: {total_records}")

# Check for records with empty/null gif_path
cursor.execute("SELECT COUNT(*) FROM alert_gifs WHERE gif_path = '' OR gif_path IS NULL")
empty_path_count = cursor.fetchone()[0]
print(f"Records with empty/NULL gif_path: {empty_path_count}")

# Check for records with gif_path
cursor.execute("SELECT COUNT(*) FROM alert_gifs WHERE gif_path != '' AND gif_path IS NOT NULL")
filled_path_count = cursor.fetchone()[0]
print(f"Records with gif_path filled: {filled_path_count}")

# Sample some records
print("\n" + "-" * 80)
print("SAMPLE RECORDS FROM alert_gifs TABLE:")
print("-" * 80)
cursor.execute("""
    SELECT id, channel_id, alert_type, gif_filename, gif_path, created_at 
    FROM alert_gifs 
    LIMIT 10
""")
for row in cursor.fetchall():
    print(f"ID: {row[0]}, Channel: {row[1]}, Type: {row[2]}")
    print(f"  Filename: {row[3]}")
    print(f"  Path: {row[4]}")
    print(f"  Created: {row[5]}")
    print()

# Check filesystem
print("=" * 80)
print("FILESYSTEM ANALYSIS")
print("=" * 80)

alerts_dir = "static/alerts"
if os.path.exists(alerts_dir):
    files = [f for f in os.listdir(alerts_dir) if f.endswith('.gif')]
    print(f"\nTotal GIF files in {alerts_dir}: {len(files)}")
    
    # Show sample files
    print("\nSample GIF files:")
    for f in files[-10:]:  # Last 10 files
        filepath = os.path.join(alerts_dir, f)
        size = os.path.getsize(filepath)
        print(f"  {f} ({size} bytes)")
else:
    print(f"Alerts directory not found: {alerts_dir}")

# Compare: GIFs in filesystem but not properly recorded in database
print("\n" + "=" * 80)
print("COMPARISON ANALYSIS")
print("=" * 80)

cursor.execute("SELECT gif_filename FROM alert_gifs WHERE gif_path != '' AND gif_path IS NOT NULL")
db_files = set([row[0] for row in cursor.fetchall()])

if os.path.exists(alerts_dir):
    filesystem_files = set([f for f in os.listdir(alerts_dir) if f.endswith('.gif')])
    
    missing_in_db = filesystem_files - db_files
    print(f"\nGIFs in filesystem but NOT properly in database: {len(missing_in_db)}")
    if missing_in_db:
        print("Sample missing GIFs:")
        for f in list(missing_in_db)[:10]:
            print(f"  {f}")
    
    extra_in_db = db_files - filesystem_files
    print(f"\nGIFs in database but NOT in filesystem: {len(extra_in_db)}")
    if extra_in_db:
        print("Sample extra in DB:")
        for f in list(extra_in_db)[:5]:
            print(f"  {f}")
    
    print(f"\nMatching GIFs: {len(db_files & filesystem_files)}")

conn.close()
print("\n" + "=" * 80)
