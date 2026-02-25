"""Remove TableServiceMonitor and ServiceDisciplineMonitor from camera_21 in the database."""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=os.getenv('DB_PORT', '5432'),
    dbname=os.getenv('DB_NAME', 'sakshiai'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD')
)
cur = conn.cursor()

channel_id = 'camera_21'
modules_to_remove = ['TableServiceMonitor', 'ServiceDisciplineMonitor']

# Show current state
cur.execute("SELECT id, app_name, config_type FROM channel_config WHERE channel_id = %s", (channel_id,))
print(f"channel_config for {channel_id}:")
for row in cur.fetchall():
    print(f"  id={row[0]}, app_name={row[1]}, config_type={row[2]}")

cur.execute("SELECT id, module_name FROM channel_modules WHERE channel_id = %s", (channel_id,))
print(f"\nchannel_modules for {channel_id}:")
for row in cur.fetchall():
    print(f"  id={row[0]}, module_name={row[1]}")

# Delete from channel_config
for mod in modules_to_remove:
    cur.execute("DELETE FROM channel_config WHERE channel_id = %s AND app_name = %s", (channel_id, mod))
    print(f"\nDeleted {cur.rowcount} rows from channel_config for {channel_id} - {mod}")

# Delete from channel_modules
for mod in modules_to_remove:
    cur.execute("DELETE FROM channel_modules WHERE channel_id = %s AND module_name = %s", (channel_id, mod))
    print(f"Deleted {cur.rowcount} rows from channel_modules for {channel_id} - {mod}")

conn.commit()
print("\nDone. Database updated successfully.")

cur.close()
conn.close()
