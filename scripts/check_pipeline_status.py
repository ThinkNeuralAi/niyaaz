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
    target_modules = [
        'DressCodeMonitoring', 'PPEMonitoring',
        'TableServiceMonitor', 'ServiceDisciplineMonitor',
        'TableCleanlinessMonitor'
    ]
    cur.execute('''
        SELECT channel_id, module_name, is_active, created_at
        FROM channel_modules
        WHERE module_name IN %s
        ORDER BY module_name, channel_id
    ''', (tuple(target_modules),))
    rows = cur.fetchall()
    if rows:
        for r in rows:
            status = 'ACTIVE' if r[2] else 'INACTIVE'
            print(f'  {r[0]:25s} | {r[1]:30s} | {status:10s} | {r[3]}')
    else:
        print('  >> NONE of these modules are assigned to any channels!')
        print('  >> This is likely the root cause - modules need to be assigned first.')

    # 2. All distinct module names
    print('\n[2] ALL MODULE TYPES IN channel_modules')
    print('-' * 70)
    cur.execute('''
        SELECT module_name, COUNT(*) as cnt,
               SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active_cnt
        FROM channel_modules
        GROUP BY module_name
        ORDER BY module_name
    ''')
    for r in cur.fetchall():
        print(f'  {r[0]:35s} | {r[1]:3d} total | {r[2]:3d} active')

    # 3. Active channels
    print('\n[3] ACTIVE CHANNELS')
    print('-' * 70)
    cur.execute('''
        SELECT channel_id, store_id, is_active
        FROM channels
        WHERE is_active = true
        ORDER BY store_id, channel_id
    ''')
    rows = cur.fetchall()
    print(f'  Total active channels: {len(rows)}')
    for r in rows:
        print(f'  {r[0]:25s} | Store: {r[1]}')

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
