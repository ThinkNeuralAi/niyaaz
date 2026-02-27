"""
Diagnostic script to check why alerts are missing from the dashboard.
Run this on the server: python scripts/check_pipeline_status.py
"""
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import psycopg2

def get_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_NAME', 'sakshiai'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'Postgres123')
    )

def main():
    conn = get_connection()
    cur = conn.cursor()

    print('=' * 70)
    print('  DASHBOARD ALERT DIAGNOSTIC REPORT')
    print('=' * 70)

    # 1. Check which modules are assigned to channels
    print('\n[1] MODULE ASSIGNMENTS (channel_modules table)')
    print('-' * 70)
    # First check what columns exist
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'channel_modules' ORDER BY ordinal_position")
    columns = [r[0] for r in cur.fetchall()]
    print(f'  Table columns: {columns}')
    print()

    target_modules = [
        'DressCodeMonitoring', 'PPEMonitoring',
        'TableServiceMonitor', 'ServiceDisciplineMonitor',
        'TableCleanlinessMonitor'
    ]
    cur.execute('''
        SELECT channel_id, module_name, created_at
        FROM channel_modules
        WHERE module_name IN %s
        ORDER BY module_name, channel_id
    ''', (tuple(target_modules),))
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f'  {r[0]:25s} | {r[1]:30s} | Created: {r[2]}')
    else:
        print('  >> NONE of these modules are assigned to any channels!')
        print('  >> This is likely the root cause - modules need to be assigned first.')

    # 2. All distinct module names
    print('\n[2] ALL MODULE TYPES IN channel_modules')
    print('-' * 70)
    cur.execute('''
        SELECT module_name, COUNT(*) as cnt
        FROM channel_modules
        GROUP BY module_name
        ORDER BY module_name
    ''')
    for r in cur.fetchall():
        print(f'  {r[0]:35s} | {r[1]:3d} channels')

    # 3. Active channels - discover table name first
    print('\n[3] ACTIVE CHANNELS')
    print('-' * 70)
    try:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name LIKE '%channel%'")
        channel_tables = [r[0] for r in cur.fetchall()]
        print(f'  Channel-related tables: {channel_tables}')
        # Try common table names
        for tbl in ['channels', 'channel_configs', 'camera_channels']:
            if tbl in channel_tables:
                try:
                    cur.execute(f'SELECT * FROM {tbl} LIMIT 1')
                    cols = [desc[0] for desc in cur.description]
                    print(f'  Using table: {tbl} (columns: {cols})')
                    cur.execute(f'SELECT channel_id, store_id FROM {tbl} ORDER BY store_id, channel_id')
                    rows = cur.fetchall()
                    print(f'  Total channels: {len(rows)}')
                    for r in rows:
                        print(f'    {r[0]:25s} | Store: {r[1]}')
                    break
                except Exception as e2:
                    conn.rollback()
                    print(f'  Error reading {tbl}: {e2}')
        # Also check channel_modules for store mapping
        print('\n  Channel-to-Store from channel_modules:')
        cur.execute('SELECT DISTINCT channel_id, store_id FROM channel_modules ORDER BY store_id, channel_id')
        for r in cur.fetchall():
            print(f'    {r[0]:25s} | Store: {r[1]}')
    except Exception as e:
        conn.rollback()
        print(f'  ERROR: {e}')

    # 4. Recent data in alert tables
    print('\n[4] DATABASE TABLES - Record counts and last entry')
    print('-' * 70)
    tables = [
        ('ppe_alerts', 'created_at'),
        ('dresscode_alerts', 'created_at'),
        ('table_cleanliness_violations', 'created_at'),
        ('table_service_violations', 'created_at'),
        ('alert_gifs', 'created_at'),
        ('queue_violations', 'created_at'),
        ('cash_snapshots', 'created_at'),
    ]
    for tbl, col in tables:
        try:
            cur.execute(f'SELECT COUNT(*), MAX({col}) FROM {tbl}')
            r = cur.fetchone()
            count = r[0] or 0
            last = r[1] or 'N/A'
            marker = ' << EMPTY!' if count == 0 else ''
            print(f'  {tbl:35s} | Count: {count:6d} | Last: {last}{marker}')
        except Exception as e:
            conn.rollback()
            print(f'  {tbl:35s} | ERROR: {e}')

    # 5. Alert types in alert_gifs
    print('\n[5] ALERT TYPES in alert_gifs (breakdown)')
    print('-' * 70)
    try:
        cur.execute('''
            SELECT alert_type, COUNT(*) as cnt, MAX(created_at) as last_entry
            FROM alert_gifs
            GROUP BY alert_type
            ORDER BY cnt DESC
        ''')
        for r in cur.fetchall():
            print(f'  {str(r[0]):35s} | Count: {r[1]:6d} | Last: {r[2]}')
    except Exception as e:
        conn.rollback()
        print(f'  ERROR: {e}')

    # 6. Check if stores have excluded modules
    print('\n[6] STORE CONFIGURATIONS (excluded_modules)')
    print('-' * 70)
    try:
        cur.execute('SELECT store_id, store_name, excluded_modules FROM stores ORDER BY store_id')
        for r in cur.fetchall():
            excluded = r[2] if r[2] else 'None'
            print(f'  {r[0]:15s} | {r[1]:20s} | Excluded: {excluded}')
    except Exception as e:
        conn.rollback()
        print(f'  ERROR: {e}')

    # 7. Check enabled status of our target modules
    print('\n[7] ENABLED STATUS of target modules')
    print('-' * 70)
    try:
        cur.execute('''
            SELECT channel_id, store_id, module_name, enabled
            FROM channel_modules
            WHERE module_name IN ('DressCodeMonitoring', 'PPEMonitoring', 'TableServiceMonitor', 'ServiceDisciplineMonitor')
            ORDER BY module_name, channel_id
        ''')
        for r in cur.fetchall():
            status = 'ENABLED' if r[3] else 'DISABLED'
            print(f'  {r[0]:20s} | {r[1]:10s} | {r[2]:30s} | {status}')
    except Exception as e:
        conn.rollback()
        print(f'  ERROR: {e}')

    conn.close()

    print('\n' + '=' * 70)
    print('  DIAGNOSIS COMPLETE')
    print('=' * 70)
    print('''
KEY THINGS TO CHECK:
  - If [1] shows NO modules assigned, that's why there's no data
  - If [2] is missing the target modules entirely, they were never configured
  - If [4] shows EMPTY tables, the pipeline hasn't been writing data
  - If [6] shows modules as "excluded", they won't appear on the dashboard
''')

if __name__ == '__main__':
    main()
