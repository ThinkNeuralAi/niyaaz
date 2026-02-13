"""
Alert GIFs Verification Script
Monitors alert_gifs table health and consistency
"""
import sqlite3
import os
from datetime import datetime, timedelta

db_path = "data/sakshi.db"

def verify_alert_gifs_health():
    """Comprehensive health check of alert_gifs table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("=" * 80)
    print("ALERT_GIFS TABLE HEALTH CHECK")
    print("=" * 80)
    
    # 1. Check total records
    cursor.execute("SELECT COUNT(*) FROM alert_gifs")
    total = cursor.fetchone()[0]
    print(f"\n✅ Total records: {total}")
    
    # 2. Check for empty gif_path
    cursor.execute("SELECT COUNT(*) FROM alert_gifs WHERE gif_path = '' OR gif_path IS NULL")
    empty_count = cursor.fetchone()[0]
    if empty_count == 0:
        print(f"✅ Records with valid gif_path: {total} (100%)")
    else:
        print(f"❌ Records with empty/NULL gif_path: {empty_count}")
    
    # 3. Check alert type distribution
    print("\n📊 Alert Types Distribution:")
    cursor.execute("""
        SELECT alert_type, COUNT(*) as count 
        FROM alert_gifs 
        GROUP BY alert_type 
        ORDER BY count DESC
    """)
    for alert_type, count in cursor.fetchall():
        print(f"   - {alert_type}: {count}")
    
    # 4. Check recent alerts (last 24 hours)
    yesterday = datetime.now() - timedelta(hours=24)
    cursor.execute("""
        SELECT COUNT(*) FROM alert_gifs 
        WHERE created_at > datetime(?)
    """, (yesterday.isoformat(),))
    recent_count = cursor.fetchone()[0]
    print(f"\n📈 Recent alerts (last 24h): {recent_count}")
    
    # 5. Check filesystem consistency
    print("\n🔍 Filesystem Consistency Check:")
    alerts_dir = "static/alerts"
    if os.path.exists(alerts_dir):
        fs_gifs = set([f for f in os.listdir(alerts_dir) if f.endswith('.gif')])
        fs_count = len(fs_gifs)
        print(f"   GIF files in filesystem: {fs_count}")
        
        # Get GIFs in database
        cursor.execute("SELECT gif_filename FROM alert_gifs WHERE alert_type LIKE '%alert%'")
        db_gifs = set([row[0] for row in cursor.fetchall() if row[0]])
        
        missing_in_db = fs_gifs - db_gifs
        if missing_in_db:
            print(f"   ⚠️ GIFs missing in database: {len(missing_in_db)}")
        else:
            print(f"   ✅ All filesystem GIFs are in database")
    else:
        print(f"   ⚠️ Alerts directory not found: {alerts_dir}")
    
    # 6. Data quality metrics
    print("\n📋 Data Quality Metrics:")
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN alert_message IS NOT NULL THEN 1 END) as with_message,
            COUNT(CASE WHEN alert_data IS NOT NULL THEN 1 END) as with_data,
            COUNT(CASE WHEN frame_count > 0 THEN 1 END) as with_frame_count
        FROM alert_gifs
    """)
    total, with_msg, with_data, with_frames = cursor.fetchone()
    print(f"   Records with message: {with_msg}/{total} ({100*with_msg/total:.1f}%)")
    print(f"   Records with data: {with_data}/{total} ({100*with_data/total:.1f}%)")
    print(f"   Records with frame_count: {with_frames}/{total} ({100*with_frames/total:.1f}%)")
    
    # 7. Sample recent records
    print("\n📿 Recent Sample Records:")
    cursor.execute("""
        SELECT id, channel_id, alert_type, gif_filename, created_at
        FROM alert_gifs
        ORDER BY created_at DESC
        LIMIT 5
    """)
    for rec_id, channel, alert_type, filename, created in cursor.fetchall():
        print(f"   ID: {rec_id} | {channel} | {alert_type} | {created}")
    
    # 8. Health status
    print("\n" + "=" * 80)
    print("HEALTH STATUS")
    print("=" * 80)
    
    health_status = "✅ HEALTHY" if empty_count == 0 and total > 0 else "⚠️ NEEDS ATTENTION"
    print(f"\n{health_status}")
    
    if empty_count > 0:
        print(f"   ACTION: Run 'python fix_alert_gifs_cleanup.py' to remove empty records")
    
    if fs_count > total * 0.1:  # More than 10% of GIFs are not in database
        print(f"   ACTION: Run 'python add_missing_gifs_to_db.py' to recover missing records")
    
    conn.close()

if __name__ == "__main__":
    verify_alert_gifs_health()
