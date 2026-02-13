"""
Fix alert_gifs table - remove records with empty gif_path and consolidate entries
This script will:
1. Delete all alert_gifs records with empty/NULL gif_path (these are orphaned log_alert entries)
2. Verify filesystem matches database entries
"""
import sqlite3
import os
from datetime import datetime

db_path = "data/sakshi.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 80)
print("FIXING ALERT_GIFS TABLE")
print("=" * 80)

# Show before stats
cursor.execute("SELECT COUNT(*) FROM alert_gifs")
before_total = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM alert_gifs WHERE gif_path = '' OR gif_path IS NULL")
before_empty = cursor.fetchone()[0]

print(f"\nBEFORE FIX:")
print(f"  Total records: {before_total}")
print(f"  Records with empty/NULL gif_path: {before_empty}")
print(f"  Records with gif_path: {before_total - before_empty}")

# Delete records with empty gif_path (these are orphaned log_alert entries)
print(f"\nDeleting {before_empty} records with empty/NULL gif_path...")
cursor.execute("DELETE FROM alert_gifs WHERE gif_path = '' OR gif_path IS NULL")
conn.commit()

# Show after stats
cursor.execute("SELECT COUNT(*) FROM alert_gifs")
after_total = cursor.fetchone()[0]

print(f"\nAFTER DELETION:")
print(f"  Total records: {after_total}")
print(f"  Records deleted: {before_empty}")

# Now check for inconsistencies (GIFs in DB but not in filesystem)
print("\n" + "=" * 80)
print("CHECKING DATABASE VS FILESYSTEM CONSISTENCY")
print("=" * 80)

cursor.execute("SELECT id, gif_filename, gif_path FROM alert_gifs WHERE gif_path != '' AND gif_path IS NOT NULL")
db_records = cursor.fetchall()

bad_records = []
for record_id, filename, path in db_records:
    if path and not os.path.exists(path):
        bad_records.append((record_id, filename, path))
        print(f"\n⚠️ Record ID {record_id}: File missing: {path}")

if bad_records:
    print(f"\n\nFound {len(bad_records)} records with missing files")
    print("Deleting records with missing files...")
    for record_id, _, _ in bad_records:
        cursor.execute("DELETE FROM alert_gifs WHERE id = ?", (record_id,))
    conn.commit()
    print(f"Deleted {len(bad_records)} records")

# Final stats
cursor.execute("SELECT COUNT(*) FROM alert_gifs")
final_total = cursor.fetchone()[0]

print(f"\n" + "=" * 80)
print(f"FINAL STATS")
print("=" * 80)
print(f"  Total records in alert_gifs: {final_total}")
print(f"  Records deleted in total: {before_total - final_total}")
print(f"  Summary: Removed {before_empty} empty records + {len(bad_records)} missing file records")

print(f"\n✅ Alert GIFs table cleaned up successfully!")

conn.close()
